"""PyTorch models for STREAM and standard CFM baselines."""

from __future__ import annotations

import torch
from torch import nn


class StandardCFM(nn.Module):
    """CFM vector field mapping a cell-state representation to expression velocity."""

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
        state_dim: int | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dim = state_dim or n_genes
        for _ in range(n_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            dim = hidden_dim
        layers.append(nn.Linear(dim, n_genes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StreamModel(nn.Module):
    """Sequence-conditioned STREAM vector field.

    The per-gene output is read from the promoter token, which must be placed at
    token index 0 in the packed CRE tensors.
    """

    def __init__(
        self,
        n_genes: int,
        cre_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        variant: str = "film",
        positional_encoding: str = "rope",
        n_context_tokens: int = 8,
        state_dim: int | None = None,
        output_dim: int = 1,
    ):
        super().__init__()
        if variant not in {"film", "cross_attention"}:
            raise ValueError("variant must be 'film' or 'cross_attention'")
        if positional_encoding not in {"none", "learned", "rope"}:
            raise ValueError("positional_encoding must be none, learned, or rope")
        self.n_genes = n_genes
        self.state_dim = state_dim or n_genes
        self.d_model = d_model
        self.variant = variant
        self.positional_encoding = positional_encoding
        self.input_proj = nn.Linear(cre_dim, d_model)
        self.distance_proj = nn.Sequential(nn.Linear(1, d_model), nn.Tanh(), nn.Linear(d_model, d_model))
        self.promoter_embed = nn.Embedding(2, d_model)
        self.learned_pos = nn.Embedding(2049, d_model) if positional_encoding == "learned" else None
        self.cre_encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )

        if variant == "film":
            self.cell_context = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.state_dim, d_model),
                        nn.GELU(),
                        nn.Linear(d_model, 2 * d_model),
                    )
                    for _ in range(n_layers)
                ]
            )
            self.cross_attn = None
        else:
            self.cell_context = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.state_dim, d_model * n_context_tokens),
                        nn.GELU(),
                        nn.Linear(d_model * n_context_tokens, d_model * n_context_tokens),
                    )
                    for _ in range(n_layers)
                ]
            )
            self.n_context_tokens = n_context_tokens
            self.cross_attn = nn.ModuleList(
                [nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True) for _ in range(n_layers)]
            )

        self.norm = nn.LayerNorm(d_model)
        self.output_dim = int(output_dim)
        self.head = nn.Linear(d_model, self.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        cre_embeddings: torch.Tensor,
        cre_mask: torch.Tensor,
        signed_distance: torch.Tensor,
        is_promoter: torch.Tensor,
        gene_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        promoter = self.encode_gene_features(
            x, cre_embeddings, cre_mask, signed_distance, is_promoter, gene_indices=gene_indices
        )
        output = self.head(promoter)
        return output.squeeze(-1) if self.output_dim == 1 else output

    def encode_gene_features(
        self,
        x: torch.Tensor,
        cre_embeddings: torch.Tensor,
        cre_mask: torch.Tensor,
        signed_distance: torch.Tensor,
        is_promoter: torch.Tensor,
        gene_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return conditioned promoter features before the gene-level readout."""

        if gene_indices is not None:
            cre_embeddings = cre_embeddings.index_select(0, gene_indices)
            cre_mask = cre_mask.index_select(0, gene_indices)
            signed_distance = signed_distance.index_select(0, gene_indices)
            is_promoter = is_promoter.index_select(0, gene_indices)
        h, padding_mask = self.embed_cre(cre_embeddings, cre_mask, signed_distance, is_promoter)
        batch_size, n_genes, n_tokens = x.shape[0], h.shape[0], h.shape[1]
        h = h.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        layer_padding_mask = padding_mask.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * n_genes, n_tokens)
        for layer_index, layer in enumerate(self.cre_encoder_layers):
            h = h.reshape(batch_size * n_genes, n_tokens, self.d_model)
            h = layer(h, src_key_padding_mask=layer_padding_mask)
            h = h.reshape(batch_size, n_genes, n_tokens, self.d_model)
            if self.variant == "film":
                gamma_beta = self.cell_context[layer_index](x)
                gamma, beta = gamma_beta.chunk(2, dim=-1)
                h = h * (1.0 + gamma[:, None, None, :]) + beta[:, None, None, :]
            else:
                context = self.cell_context[layer_index](x).reshape(batch_size, self.n_context_tokens, self.d_model)
                q = h.reshape(batch_size, n_genes * n_tokens, self.d_model)
                attn_out, _ = self.cross_attn[layer_index](q, context, context, need_weights=False)
                h = (q + attn_out).reshape(batch_size, n_genes, n_tokens, self.d_model)
            h = h.masked_fill(padding_mask[None, :, :, None], 0.0)
        return self.norm(h[:, :, 0, :])

    def embed_cre(
        self,
        cre_embeddings: torch.Tensor,
        cre_mask: torch.Tensor,
        signed_distance: torch.Tensor,
        is_promoter: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist_scaled = signed_distance.float().unsqueeze(-1) / 100_000.0
        h = self.input_proj(cre_embeddings.float())
        h = h + self.distance_proj(dist_scaled)
        h = h + self.promoter_embed(is_promoter.long())
        if self.learned_pos is not None:
            bins = torch.clamp(torch.round(signed_distance.float() / 100.0).long() + 1024, 0, 2048)
            h = h + self.learned_pos(bins)
        elif self.positional_encoding == "rope":
            h = apply_rope(h, signed_distance.float())
        padding_mask = ~cre_mask.bool()
        return h.masked_fill(padding_mask.unsqueeze(-1), 0.0), padding_mask


class ScoreFlowStreamModel(nn.Module):
    """STREAM with autonomous and coupled stochastic-interpolant readouts.

    ``tau`` is stochastic-interpolant path position, not developmental time.
    """

    def __init__(self, *, state_dim: int, time_dim: int = 32, **stream_kwargs):
        super().__init__()
        if time_dim < 4 or time_dim % 2:
            raise ValueError("time_dim must be an even integer of at least four")
        self.time_dim = time_dim
        d_model = int(stream_kwargs.get("d_model", 256))
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, d_model), nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.stream = StreamModel(state_dim=state_dim, output_dim=1, **stream_kwargs)
        self.stream.head = nn.Identity()
        self.autonomous_velocity_head = nn.Linear(d_model, 1)
        self.conditional_velocity_head = nn.Linear(d_model, 1)
        self.noise_head = nn.Linear(d_model, 1)

    def forward(self, state: torch.Tensor, tau: torch.Tensor, **cre_inputs) -> torch.Tensor:
        tau = tau.reshape(-1, 1)
        frequencies = torch.exp(
            torch.linspace(0.0, -6.0, self.time_dim // 2, device=tau.device, dtype=tau.dtype)
        )
        angles = 2.0 * torch.pi * tau * frequencies
        gamma, beta = self.time_mlp(torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)).chunk(2, dim=1)
        features = self.stream.encode_gene_features(state, **cre_inputs)
        conditional_features = features * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        autonomous_velocity = self.autonomous_velocity_head(features).squeeze(-1)
        conditional_velocity = self.conditional_velocity_head(conditional_features).squeeze(-1)
        noise = self.noise_head(conditional_features).squeeze(-1)
        return torch.stack([autonomous_velocity, conditional_velocity, noise], dim=-1)

def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10_000.0) -> torch.Tensor:
    """Apply RoPE to token features using signed genomic positions."""

    dim = x.shape[-1]
    half = dim // 2
    if half == 0:
        return x
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    freq = torch.arange(half, device=x.device, dtype=x.dtype)
    inv_freq = base ** (-freq / max(half - 1, 1))
    theta = positions.to(device=x.device, dtype=x.dtype).unsqueeze(-1) * inv_freq / 1000.0
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if dim % 2:
        rotated = torch.cat([rotated, x[..., -1:]], dim=-1)
    return rotated


def mse_cfm_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)
