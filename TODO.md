# TODO

## Improve OT Pair Reuse Without Losing Interval Diversity

- Replace the single sequential pair queue with an interval-stratified pair bank.
- Construct a large OT plan independently for every training interval.
- Sample 256-1,024 endpoint pairs from each plan and store paired expression endpoints on CPU.
- Shuffle model microbatches across intervals instead of exhausting one interval before moving to the next.
- Refresh an interval's OT plan after its pair bank is exhausted.
- Keep the model/UCE microbatch small while allowing large OT endpoint pools.
- Balance intervals explicitly so densely sampled developmental stages do not dominate training.
- Record plan diagnostics by interval: transported mass, pair cost, effective edges, maximum edge mass, and marginal imbalance.
- Compare pair-bank size and refresh frequency while holding the number of optimizer examples fixed.

## Denoise Expression Before Fitting Dynamics

Most CFM applications fit dynamics in a lower-dimensional or denoised state space. Determine whether sparse count noise is overwhelming the developmental velocity signal.

- **PCA coupling cost:** compute OT in train-fitted log-normalized PCA space while retaining expression-space interpolation, targets, and gene-level outputs. Test this first because it changes pairing without changing the STREAM output contract.
- **PCA dynamics:** fit CFM velocities in train-fitted PC space and decode predicted endpoints to genes. Quantify reconstruction loss and determine how CRE-conditioned per-gene prediction would be retained or recovered.
- **Meta-cells:** aggregate cells within timepoint and local state neighborhoods before OT/CFM. Preserve rare populations and avoid combining distinct lineages.
- **KNN smoothing:** smooth expression within each timepoint using neighbors defined from training-only PCA coordinates. Never use held-out-stage cells or cross-time neighbors during preprocessing.
- Compare raw counts, PCA-cost-only OT, meta-cells, and KNN-smoothed endpoints using the same held-out blocks and persistence baseline.
- Report whether denoising improves observed-interval validation, not only held-out endpoint metrics.

## Train With Non-Adjacent Timepoints

Non-adjacent endpoints can provide larger, less noise-dominated displacement targets and expose the autonomous field to multiple temporal scales. They should supplement adjacent intervals rather than replace them initially: a straight OT chord over a long interval may skip intermediate branches or conflict with local velocities.

- Add skip-1, skip-2, and skip-4 intervals alongside adjacent intervals.
- Balance sampling across temporal gap sizes and developmental regions.
- Normalize validation by interval-specific velocity scale so short noisy intervals and long smooth intervals are comparable.
- Compare adjacent-only training against adjacent-plus-skip curricula at a fixed number of cells and optimizer updates.
- Measure consistency between velocities learned from adjacent and non-adjacent intervals at overlapping states.
- Evaluate forecasts at several horizons using contiguous held-out blocks.
- For a held-out block, exclude every training interval whose temporal span crosses that block, even when both interval endpoints are observed. Otherwise a long training chord would weaken the intended forecasting test.
- If long chords degrade local forecasts, progressively introduce larger gaps after fitting adjacent intervals or reduce their loss weight.
