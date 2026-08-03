# TODO

## Improve OT Pair Reuse Without Losing Interval Diversity (Implemented)

- Replace the single sequential pair queue with an interval-stratified pair bank.
- Construct a large OT plan independently for every training interval.
- Sample 1,024 endpoint pairs from each production plan and store paired expression endpoints on CPU.
- Shuffle model microbatches across intervals instead of exhausting one interval before moving to the next.
- Refresh an interval's OT plan after its pair bank is exhausted.
- Keep the model/UCE microbatch small while allowing large OT endpoint pools.
- Balance intervals explicitly so densely sampled developmental stages do not dominate training.
- Record plan diagnostics by interval: transported mass, pair cost, effective edges, maximum edge mass, and marginal imbalance.
- Compare pair-bank size and refresh frequency while holding the number of optimizer examples fixed.

## Denoise Expression Before Fitting Dynamics (Ablations Running)

Most CFM applications fit dynamics in a lower-dimensional or denoised state space. Determine whether sparse count noise is overwhelming the developmental velocity signal.

- **PCA coupling cost:** implemented using train-fitted log-normalized PCA coordinates while retaining expression-space interpolation and gene-level outputs.
- **PCA endpoint denoising:** implemented by reconstructing selected gene-space velocity targets from the train-fitted PCA model.
- Do not fit dynamics or predict velocities directly in PC space. PC outputs do not have a natural gene-specific CRE representation and would change the STREAM model contract.
- **Meta-cells:** implemented as within-timepoint PCA clusters whose log-normalized expression centroids replace selected endpoint targets while retaining each cell's library size.
- **KNN smoothing:** implemented using within-timepoint neighbors from train-fitted PCA coordinates; held-out-stage and cross-time neighbors are never used.
- **Smoothed UCE input:** implemented an autonomous PCA preprocessing arm evaluating `UCE(D(x))` at every training interpolation and Euler rollout step. Also test KNN/metacell-smoothed UCE input during fitting and fixed validation, reverting explicitly to raw expression during rollout because no timepoint-specific reference population is available.
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

## Learn Developmental Pseudotime

- Infer lineage-aware pseudotime within broad annotated stages and use it as a continuous ordering variable for OT and stochastic-interpolant training.
- Fit pseudotime on training stages only; do not use held-out-stage expression when constructing the coordinate.
- Compare stage time, pseudotime, and pseudotime-within-stage while holding endpoint pairs and model capacity fixed.
- Check whether pseudotime reduces multimodality within intervals and improves contiguous held-out-block forecasts over persistence.
