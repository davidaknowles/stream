# stream

Workflow code for basic EDA of the JAX AnnData files from the Shendure lab public backup.

## Contents

- `notebooks/jax_adata_eda.ipynb` - source notebook for dataset inventory, metadata exploration, embryonic staging summaries, and full-data UMAP.
- `notebooks/jax_adata_eda_executed.ipynb` - executed notebook from the completed Slurm run.
- `slurm/run_jax_adata_eda.sbatch` - Slurm batch script for running the notebook on a compute node.
- `requirements-jax-adata-eda.txt` - Python dependencies for the analysis environment.
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

## UMAP Strategy

The full dataset has 11,441,407 cells. Exact in-memory UMAP over the concatenated matrix exceeded a 512 GB allocation, so the notebook uses a scalable workflow:

1. Stream over the `.h5ad` files to select highly variable genes.
2. Fit sparse SVD/PCA on a representative sample.
3. Transform all cells into the PCA space in chunks.
4. Fit UMAP on a representative PCA sample.
5. Project all cells into the learned UMAP space.

Key generated outputs include full UMAP coordinates, density plots, UMAP colored by `day`, and UMAP colored by `embryo_id`.
