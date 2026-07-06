# Lab Notebook

## 2026-07-06

- Added larger STREAM panel support for 5k and 10k protein-coding HVG runs.
- Existing `streaming_hvg_genes.csv` only contains 2,000 genes, so larger runs need a fresh streaming gene-variance table. Added `scripts/select_streaming_hvgs.py` and `slurm/run_stream_select_hvgs.sbatch` to compute this without rerunning the full UMAP workflow.
- STREAM scripts now accept `--hvg-csv`, `--n-hvg`, and `--out-dir` overrides, with matching Slurm environment variables. This keeps 5k/10k checkpoints and metrics in separate output directories instead of overwriting `outputs/stream/`.
- Gene selection now sorts by variance before taking protein-coding HVGs and selects a buffer before GTF matching, then trims after TSS annotation. This preserves the legacy <2k panel behavior when only the 2,000-row HVG file is available, while allowing exact larger panels when the full variance table is present.
- Held-out evaluation now reports `eval_gene_set`, `n_eval_genes`, full-panel metrics, and optional subset-panel metrics from a reference `selected_genes.csv`. This supports scoring 5k/10k models on both their full modeled gene sets and the legacy `outputs/stream/selected_genes.csv` panel.
- Compute-node unit test job `18708907` passed: 4 tests passed, 5 torch-dependent tests skipped because the cluster PyTorch module could not import torch without `libibverbs.so.1`.
- GPU check job `18708941` confirmed `~/venv/torchfix/bin/python` can import torch 2.11.0+cu128 and see an NVIDIA L40S. The GPU STREAM Slurm scripts now default to this venv via `STREAM_PYTHON`, with the PyTorch module retained only as a fallback.
- GPU smoke test job `18708953` exercised `evaluate_intervals` with full and legacy gene-set metrics on CUDA and passed.
