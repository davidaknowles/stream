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
- Submitted 5k/10k rerun dependency chain. Streaming variance job: `18709007`. 5k jobs: prepare `18709008`, embed `18709009`, train/eval standard CFM `18709010`/`18709011`, train/eval FiLM `18709012`/`18709013`, train/eval cross-attention `18709014`/`18709015`. 10k jobs: prepare `18709016`, embed `18709017`, train/eval standard CFM `18709018`/`18709019`, train/eval FiLM `18709020`/`18709021`, train/eval cross-attention `18709022`/`18709023`.

## 2026-07-07

- Checked the larger-panel Slurm chain. The HVG, prepare, CRE embedding, and standard CFM training jobs completed for both 5k and 10k panels. The 5k standard CFM eval also completed; 10k standard CFM eval `18709019` is still pending on GPU priority.
- FiLM and cross-attention STREAM training failed quickly for both 5k and 10k panels with CUDA OOM. The failure occurred when the model expanded regulatory token states to `[batch, genes, tokens, hidden]` for all genes at once.
- Canceled stale dependent eval jobs `18709013`, `18709015`, `18709021`, and `18709023`, which were pending with `DependencyNeverSatisfied`.
- Added gene-chunked STREAM prediction and loss computation. STREAM training and evaluation now process genes in configurable chunks (`gene_chunk_size`, default 512), preserving the same full-panel MSE objective while avoiding the full batch-by-gene activation tensor.
- Submitted guarded STREAM restart chain with `BATCH_SIZE=32` and `GENE_CHUNK_SIZE=512`. Smoke jobs: FiLM `18712595`, cross-attention `18712596`. Full jobs depend on those smoke jobs: 5k FiLM train/eval `18712617`/`18712618`, 5k cross-attention train/eval `18712619`/`18712620`, 10k FiLM train/eval `18712621`/`18712622`, and 10k cross-attention train/eval `18712623`/`18712624`.

## 2026-07-08

- Updated STREAM architecture so both conditioning mechanisms are applied at every transformer layer. FiLM now uses layer-specific cell-state MLPs to produce scale/shift vectors after each CRE self-attention block. Cross-attention now uses layer-specific cell-state context-token MLPs and cross-attention modules after each CRE self-attention block.
- Updated chunked STREAM training to backpropagate each gene chunk immediately. This avoids retaining autograd graphs for every gene chunk before a single backward pass.
