"""Configuration helpers for STREAM mouse development experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StreamConfig:
    project_root: Path = Path("/gpfs/commons/home/daknowles/projects/stream")
    adata_dir: Path = Path("downloads/adata")
    cell_metadata_csv: Path = Path("downloads/adata/df_cell.csv")
    gene_metadata_csv: Path = Path("downloads/adata/df_gene.csv")
    hvg_csv: Path = Path("outputs/jax_adata_eda/streaming_hvg_genes.csv")
    ccre_bed: Path = Path("downloads/screen_mouse_cre/mm10-cCREs.bed.gz")
    gtf: Path = Path("~/knowles_lab/index/nonhuman/mm10/Mus_musculus.GRCm38.95.gtf.gz")
    fasta: Path = Path("~/knowles_lab/index/nonhuman/mm10/GRCm38.primary_assembly.genome.fa.gz")
    out_dir: Path = Path("outputs/stream")

    gene_set: str = "protein_coding_hvg"
    n_hvg: int = 2000
    cre_window_bp: int = 100_000
    promoter_window_bp: int = 1_000
    synthetic_promoter_bp: int = 512
    max_cres_per_gene: int = 32

    alphagenome_repo: Path = Path("/gpfs/commons/home/daknowles/projects/alphagenome-pytorch")
    alphagenome_checkpoint: Path | None = None
    alphagenome_sequence_bp: int = 131_072
    alphagenome_batch_size: int = 2
    alphagenome_organism_index: int = 1

    expression_layer: str | None = None
    batch_size: int = 64
    ot_epsilon: float = 0.05
    ot_iterations: int = 80
    learning_rate: float = 1e-4
    epochs: int = 10
    heldout_days: list[str] = field(default_factory=lambda: ["E9.5", "E10.5"])

    model_variant: str = "film"
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    n_context_tokens: int = 8
    dropout: float = 0.1
    positional_encoding: str = "rope"

    seed: int = 1337
    device: str = "cuda"

    use_wandb: bool = True
    wandb_project: str = "stream"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    wandb_run_name: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StreamConfig":
        import yaml

        path = Path(path)
        with path.open() as handle:
            raw = yaml.safe_load(handle) or {}
        cfg = cls(**raw)
        cfg.project_root = Path(cfg.project_root).expanduser().resolve()
        for name in (
            "adata_dir",
            "cell_metadata_csv",
            "gene_metadata_csv",
            "hvg_csv",
            "ccre_bed",
            "gtf",
            "fasta",
            "out_dir",
            "alphagenome_repo",
        ):
            value = getattr(cfg, name)
            setattr(cfg, name, cfg.resolve_path(value))
        if cfg.alphagenome_checkpoint is not None:
            cfg.alphagenome_checkpoint = cfg.resolve_path(cfg.alphagenome_checkpoint)
        return cfg

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                out[key] = str(value)
            else:
                out[key] = value
        return out
