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

STREAM learns a continuous vector field for mouse development, but unlike a standard expression-only CFM model it conditions each gene's predicted velocity on that gene's regulatory sequence context. The core idea is:

1. Represent each cell as expression over a selected protein-coding HVG panel.
2. Link cCREs to each gene if they fall within 100 kb of the gene TSS.
3. Embed each linked CRE/promoter sequence with AlphaGenome.
4. Treat the CREs for a gene as regulatory tokens processed by a shared transformer.
5. Condition those regulatory tokens on the current global cell state.
6. Read each gene's velocity from the final promoter-token representation.

Every gene receives an explicit promoter token. If no linked cCRE is within 1 kb of the TSS, a synthetic promoter CRE centered on the TSS is inserted. This gives the model a consistent promoter readout location while still allowing distal CRE tokens to influence the promoter representation through self-attention.

The model learns by conditional flow matching between neighboring developmental time points. Each minibatch contains cells from one adjacent interval, minibatch optimal transport couples cells across that interval, and the model regresses the velocity required to move from the earlier cell state to the later cell state.

The implemented comparison includes:

- standard expression-only CFM;
- STREAM with FiLM conditioning on cell state;
- STREAM with cross-attention conditioning on cell state.

The FiLM variant maps cell state to feature-wise scale/shift parameters applied to regulatory token states. The cross-attention variant maps cell state to context tokens that regulatory tokens attend to. Both STREAM variants use the same CRE links, AlphaGenome embeddings, promoter-token readout, selected genes, and minibatch OT setup as the baseline comparison.

The GPU Slurm scripts use `STREAM_PYTHON=$HOME/venv/torchfix/bin/python` by
default when that venv exists, because the local `PyTorch/2.1.2-foss-2023b`
module can fail to import torch on current nodes due to a missing
`libibverbs.so.1` runtime. Override `STREAM_PYTHON` if a different working
torch environment is desired.

Prepare gene/TSS/CRE links:

```bash
sbatch slurm/run_stream_prepare.sbatch
```

For larger gene-panel experiments, first export streaming gene variances across
all genes rather than reusing the default 2,000-row HVG table:

```bash
HVG_OUTPUT=outputs/jax_adata_eda/streaming_gene_variances.csv \
  sbatch slurm/run_stream_select_hvgs.sbatch
```

Then run each panel into its own output directory:

```bash
HVG_CSV=outputs/jax_adata_eda/streaming_gene_variances.csv N_HVG=5000 OUT_DIR=outputs/stream_hvg5000 \
  sbatch slurm/run_stream_prepare.sbatch

HVG_CSV=outputs/jax_adata_eda/streaming_gene_variances.csv N_HVG=10000 OUT_DIR=outputs/stream_hvg10000 \
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

For larger STREAM panels, training and evaluation predict genes in chunks to
avoid materializing the full `[batch, genes, CRE tokens, hidden]` activation
tensor. The default `gene_chunk_size` is 512 and can be overridden in Slurm with
`GENE_CHUNK_SIZE`; `BATCH_SIZE` can also be set per job without editing the YAML.

Evaluate on held-out timepoint intervals:

```bash
VARIANT=film sbatch slurm/run_stream_evaluate.sbatch
```

Evaluation always writes metrics for the model's full selected gene panel. Add
`EVAL_GENE_SUBSET=legacy_1984:outputs/stream/selected_genes.csv` to also score
the same held-out timepoint batches on the legacy <2k gene panel for fair
comparison against earlier runs.

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
