# stream

Workflow code for EDA and STREAM modeling of the JAX AnnData files from the Shendure lab public backup.

## Contents

- `notebooks/jax_adata_eda.ipynb` - source notebook for dataset inventory, metadata exploration, embryonic staging summaries, and full-data UMAP.
- `notebooks/jax_adata_eda_executed.ipynb` - executed notebook from the completed Slurm run.
- `src/jax_adata_streaming.py` - reusable streaming helpers for metadata summaries and memory-bounded full-data UMAP.
- `src/stream_model/` - PyTorch STREAM model, standard CFM baseline, CRE/TSS preprocessing, AlphaGenome embedding helpers, and minibatch OT utilities.
- `slurm/run_jax_adata_eda.sbatch` - Slurm batch script for running the notebook on a compute node.
- `slurm/run_stream_*.sbatch` - Slurm batch scripts for STREAM preprocessing, CRE embedding, training, and evaluation.
- `notebooks/stream_scvelo_velocity_stream.ipynb` - scVelo velocity stream plots for standard CFM, FiLM STREAM, and cross-attention STREAM checkpoints.
- `requirements-jax-adata-eda.txt` - Python dependencies for the analysis environment.
- `requirements-stream.txt` - Python dependencies for the STREAM workflow.
- `docs/main.tex` - model notes for STREAM.
- `AGENTS.md` - notes for future coding agents working in this repo.

Downloaded data, generated outputs, Slurm logs, and local virtual environments are intentionally not tracked by Git.

## Data

The expected local data directory is:

```text
downloads/adata/
```

Expected files:

```text
adata_JAX_dataset_1.h5ad
adata_JAX_dataset_2.h5ad
adata_JAX_dataset_3.h5ad
adata_JAX_dataset_4.h5ad
df_cell.csv
df_gene.csv
```

These files are large and are ignored by `.gitignore`.

## Environment

On the NYGC cluster, create the local virtual environment with the same module used by the Slurm script:

```bash
module load Python/3.11.5-GCCcore-13.2.0
python -m venv .venv-jax-adata-eda
source .venv-jax-adata-eda/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-jax-adata-eda.txt
```

## Run

Submit the notebook execution through Slurm:

```bash
sbatch slurm/run_jax_adata_eda.sbatch
```

The script writes logs to `logs/` and outputs to:

```text
outputs/jax_adata_eda/
```

## STREAM Model

STREAM learns a sequence-conditioned CFM vector field for mouse development. cCREs are linked to genes within 100 kb of each TSS, every gene receives a promoter token, and AlphaGenome embeddings are cached before training. The model compares:

- standard expression-only CFM;
- STREAM with FiLM conditioning on cell state;
- STREAM with cross-attention conditioning on cell state.

Prepare gene/TSS/CRE links:

```bash
sbatch slurm/run_stream_prepare.sbatch
```

Set `alphagenome_checkpoint` in `configs/stream_mouse_dev.yaml`, then embed CREs:

```bash
sbatch slurm/run_stream_embed_cre.sbatch
```

Train model variants:

```bash
VARIANT=standard_cfm sbatch slurm/run_stream_train.sbatch
VARIANT=film sbatch slurm/run_stream_train.sbatch
VARIANT=cross_attention sbatch slurm/run_stream_train.sbatch
```

Evaluate on held-out timepoint intervals:

```bash
VARIANT=film sbatch slurm/run_stream_evaluate.sbatch
```

Generate scVelo velocity stream plots from the trained checkpoints:

```bash
sbatch slurm/run_stream_scvelo_notebook.sbatch
```

STREAM outputs are written to:

```text
outputs/stream/
```

## UMAP Strategy

The full dataset has 11,441,407 cells. Exact in-memory UMAP over the concatenated matrix exceeded a 512 GB allocation, so the notebook uses a scalable workflow:

1. Stream over the `.h5ad` files to select highly variable genes.
2. Fit sparse SVD/PCA on a representative sample.
3. Transform all cells into the PCA space in chunks.
4. Fit UMAP on a representative PCA sample.
5. Project all cells into the learned UMAP space.

Key generated outputs include full UMAP coordinates, density plots, UMAP colored by `day`, and UMAP colored by `embryo_id`.

Tracked example figures:

- `figures/full_umap_by_major_trajectory.png`
- `figures/full_umap_by_celltype_update_top30.png`

## Metadata Notes

The AnnData `.obs` tables contain `cell_id`, `keep`, `day`, `embryo_id`, and `experimental_batch`.

The companion `df_cell.csv` additionally contains cell labels:

- `major_trajectory`
- `celltype_update`

It also contains per-cell `day` and `embryo_id`, which can be summarized into per-embryo staging counts.
