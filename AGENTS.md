# Agent Notes

This repository contains workflow code for downloading and exploring the JAX AnnData files from the Shendure lab public backup.

## Repository Hygiene

- Do not commit downloaded data, generated outputs, Slurm logs, or virtual environments.
- Keep reusable workflow files under version control: notebooks, Slurm scripts, requirements, and documentation.
- Large artifacts live locally under `downloads/`, `outputs/`, and `logs/`, which are ignored by Git.

## Analysis Workflow

- Main notebook: `notebooks/jax_adata_eda.ipynb`
- Streaming/metadata helper module: `src/jax_adata_streaming.py`
- STREAM model package: `src/stream_model/`
- Slurm runner: `slurm/run_jax_adata_eda.sbatch`
- STREAM Slurm runners: `slurm/run_stream_*.sbatch`
- Python environment requirements: `requirements-jax-adata-eda.txt`
- STREAM environment requirements: `requirements-stream.txt`

The notebook is designed for the full 11.4M-cell dataset. Exact in-memory UMAP is not viable at this scale, so the workflow uses streaming HVG selection, sampled sparse SVD/PCA, UMAP fitting on a representative sample, and projection of all cells.

STREAM uses a protein-coding HVG panel, cCREs within 100 kb of each gene TSS, an explicit promoter token for each gene, AlphaGenome CRE embeddings cached under `outputs/stream/`, minibatch OT between neighboring time points, and held-out-timepoint evaluation. The standard CFM baseline uses the same selected genes and minibatch OT setup.

## Cluster Notes

- Use Slurm for notebook execution; do not run full-data analysis on the login node.
- The current Slurm script targets `bigmem` and loads `Python/3.11.5-GCCcore-13.2.0`.
- The local virtual environment is expected at `.venv-jax-adata-eda/` and is intentionally ignored.
