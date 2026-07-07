"""PyTorch models for STREAM and standard CFM baselines."""

from __future__ import annotations

import math

import torch
from torch import nn


class StandardCFM(nn.Module):
    """Expression-only CFM baseline using the same selected genes."""

    def __init__(self, n_genes: int, hidden_dim: int = 512, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        dim = n_genes
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
    ):
        super().__init__()
        if variant not in {"film", "cross_attention"}:
            raise ValueError("variant must be 'film' or 'cross_attention'")
        if positional_encoding not in {"none", "learned", "rope"}:
            raise ValueError("positional_encoding must be none, learned, or rope")
        self.n_genes = n_genes
        self.d_model = d_model
        self.variant = variant
        self.positional_encoding = positional_encoding
        self.input_proj = nn.Linear(cre_dim, d_model)
        self.distance_proj = nn.Sequential(nn.Linear(1, d_model), nn.Tanh(), nn.Linear(d_model, d_model))
        self.promoter_embed = nn.Embedding(2, d_model)
        self.learned_pos = nn.Embedding(2049, d_model) if positional_encoding == "learned" else None
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cre_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        if variant == "film":
            self.cell_context = nn.Sequential(
                nn.Linear(n_genes, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2 * d_model),
            )
            self.cross_attn = None
        else:
            self.cell_context = nn.Sequential(
                nn.Linear(n_genes, d_model * n_context_tokens),
                nn.GELU(),
                nn.Linear(d_model * n_context_tokens, d_model * n_context_tokens),
            )
            self.n_context_tokens = n_context_tokens
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        cre_embeddings: torch.Tensor,
        cre_mask: torch.Tensor,
        signed_distance: torch.Tensor,
        is_promoter: torch.Tensor,
        gene_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if gene_indices is not None:
            cre_embeddings = cre_embeddings.index_select(0, gene_indices)
            cre_mask = cre_mask.index_select(0, gene_indices)
            signed_distance = signed_distance.index_select(0, gene_indices)
            is_promoter = is_promoter.index_select(0, gene_indices)
        gene_tokens = self.encode_cre(cre_embeddings, cre_mask, signed_distance, is_promoter)
        h = gene_tokens.unsqueeze(0).expand(x.shape[0], -1, -1, -1).contiguous()
        if self.variant == "film":
            gamma_beta = self.cell_context(x)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            h = h * (1.0 + gamma[:, None, None, :]) + beta[:, None, None, :]
        else:
            context = self.cell_context(x).reshape(x.shape[0], self.n_context_tokens, self.d_model)
            q = h.reshape(x.shape[0], -1, self.d_model)
            attn_out, _ = self.cross_attn(q, context, context, need_weights=False)
            h = (q + attn_out).reshape_as(h)
        promoter = self.norm(h[:, :, 0, :])
        return self.head(promoter).squeeze(-1)

    def encode_cre(
        self,
        cre_embeddings: torch.Tensor,
        cre_mask: torch.Tensor,
        signed_distance: torch.Tensor,
        is_promoter: torch.Tensor,
    ) -> torch.Tensor:
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
        encoded = self.cre_encoder(h, src_key_padding_mask=padding_mask)
        return encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)


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
