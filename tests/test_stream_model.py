from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stream_model.data import (
    adjacent_intervals,
    build_time_coordinates,
    canonical_day_label,
    incoming_heldout_intervals,
    intervals_with_skips,
)
from stream_model.genome import build_token_arrays, build_token_arrays_from_matrix, link_cres_to_genes, parse_gtf_tss


def test_parse_gtf_tss_strand_coordinates(tmp_path: Path):
    gtf = tmp_path / "test.gtf"
    gtf.write_text(
        'chr1\tsrc\tgene\t11\t20\t.\t+\t.\tgene_id "g1"; gene_name "G1"; gene_type "protein_coding";\n'
        'chr1\tsrc\tgene\t31\t40\t.\t-\t.\tgene_id "g2"; gene_name "G2"; gene_type "protein_coding";\n'
    )
    out = parse_gtf_tss(gtf)
    assert dict(zip(out["gene_id"], out["tss0"])) == {"g1": 10, "g2": 39}


def test_link_cres_marks_nearest_promoter_within_1kb():
    tss = pd.DataFrame(
        [{"gene_id": "g1", "gene_name": "G1", "chrom": "chr1", "strand": "+", "tss0": 1000}]
    )
    cres = pd.DataFrame(
        [
            {"ccre_id": "far", "element_id": "far", "state": "CA", "chrom": "chr1", "start": 2500, "end": 2600, "midpoint": 2550, "is_synthetic": False},
            {"ccre_id": "near", "element_id": "near", "state": "CA", "chrom": "chr1", "start": 900, "end": 950, "midpoint": 925, "is_synthetic": False},
        ]
    )
    links = link_cres_to_genes(tss, cres, window_bp=5000, promoter_window_bp=1000)
    promoter = links[links["is_promoter"]].iloc[0]
    assert promoter["ccre_id"] == "near"
    assert promoter["token_rank"] == 0


def test_link_cres_adds_synthetic_promoter_when_closest_is_far():
    tss = pd.DataFrame(
        [{"gene_id": "g1", "gene_name": "G1", "chrom": "chr1", "strand": "+", "tss0": 1000}]
    )
    cres = pd.DataFrame(
        [{"ccre_id": "far", "element_id": "far", "state": "CA", "chrom": "chr1", "start": 2500, "end": 2600, "midpoint": 2550, "is_synthetic": False}]
    )
    links = link_cres_to_genes(tss, cres, window_bp=5000, promoter_window_bp=1000)
    promoter = links[links["is_promoter"]].iloc[0]
    assert promoter["ccre_id"] == "synthetic_promoter:g1"
    assert bool(promoter["is_synthetic"])
    assert promoter["token_rank"] == 0


def test_matrix_token_packing_matches_embedding_table():
    links = pd.DataFrame(
        {
            "gene_id": ["g1", "g1", "g2"],
            "gene_name": ["G1", "G1", "G2"],
            "ccre_id": ["c2", "c1", "c3"],
            "token_rank": [1, 0, 0],
            "signed_distance": [80, 0, 0],
            "is_promoter": [False, True, True],
        }
    )
    ids = np.asarray(["c1", "c2", "c3"])
    matrix = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    table = pd.DataFrame({"ccre_id": ids, "emb_0": matrix[:, 0], "emb_1": matrix[:, 1]})

    expected = build_token_arrays(links, table, max_tokens=2)
    actual = build_token_arrays_from_matrix(links, ids, matrix, max_tokens=2)

    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_tss_context_windows_cover_flanks_and_choose_internal_context():
    from stream_model.alphagenome_embed import (
        choose_context_window,
        overlapping_bin_slice,
        tss_context_window_starts,
    )

    starts = tss_context_window_starts(tss0=1_000_000, flank_bp=100_000, sequence_bp=131_072)
    assert starts == (900_000, 968_928)
    assert choose_context_window(905_000, 905_200, starts, 131_072) == 0
    assert choose_context_window(1_095_000, 1_095_200, starts, 131_072) == 1
    assert choose_context_window(999_900, 1_000_100, starts, 131_072) == 0
    bins = overlapping_bin_slice(900_128, 900_384, starts[0], resolution=128, n_bins=1024)
    assert bins == slice(1, 3)


def test_tss_context_windows_reject_regions_too_wide_for_two_inputs():
    from stream_model.alphagenome_embed import tss_context_window_starts

    with pytest.raises(ValueError, match="cannot span"):
        tss_context_window_starts(tss0=1_000_000, flank_bp=200_000, sequence_bp=131_072)


def test_adjacent_intervals_excludes_heldout_days():
    days = ["E8.5", "E9.0", "E9.5", "E10.0"]
    assert adjacent_intervals(days, {"E9.5"}) == [("E8.5", "E9.0")]
    assert incoming_heldout_intervals(days, {"E9.5"}) == [("E9.0", "E9.5")]


def test_skipped_intervals_exclude_spans_crossing_heldout_stages():
    days = ["E8.5", "E8.75", "E9.0", "E9.25", "E9.5"]
    intervals = intervals_with_skips(days, {"E9.0"}, max_skip=2)

    assert ("E8.5", "E8.75") in intervals
    assert ("E9.25", "E9.5") in intervals
    assert ("E8.5", "E9.25") not in intervals
    assert all("E9.0" not in days[days.index(start) : days.index(end) + 1] for start, end in intervals)


def test_time_coordinates_support_physical_days_and_relative_scaling():
    stages = ["18", "36", "72"]
    physical = build_time_coordinates(stages, "physical_days", value_scale=1 / 24)
    relative = build_time_coordinates(stages, "relative", value_scale=1 / 24)
    assert physical == {"18": 0.75, "36": 1.5, "72": 3.0}
    assert relative == {"18": 0.0, "36": 1 / 3, "72": 1.0}
    assert canonical_day_label(36.0) == "36"
    assert canonical_day_label("E9.0") == "E9.0"


def test_interval_sampler_validation_split_is_stage_stratified_and_disjoint():
    from stream_model.data import H5adIntervalSampler

    manifest = pd.DataFrame(
        [
            {"path": "atlas.h5ad", "row_idx": row, "day": day}
            for day_index, day in enumerate(["E8.5", "E8.75", "E9.0"])
            for row in range(day_index * 10, day_index * 10 + 10)
        ]
    )
    sampler = H5adIntervalSampler(
        manifest, np.arange(3), [("E8.5", "E8.75"), ("E8.75", "E9.0")], batch_size=2
    )
    train, validation = sampler.split_validation(0.2, seed=11)
    train_cells = set(zip(train.manifest["path"], train.manifest["row_idx"]))
    validation_cells = set(zip(validation.manifest["path"], validation.manifest["row_idx"]))

    assert train_cells.isdisjoint(validation_cells)
    assert validation.manifest.groupby("day").size().to_dict() == {"E8.5": 2, "E8.75": 2, "E9.0": 2}
    assert train.manifest.groupby("day").size().to_dict() == {"E8.5": 8, "E8.75": 8, "E9.0": 8}


def test_ot_and_cfm_shapes():
    torch = pytest.importorskip("torch")
    from stream_model.ot import cfm_interpolate, pairwise_squared_cost, sample_coupling_pairs, sinkhorn_coupling

    torch.manual_seed(3)
    x0 = torch.randn(5, 3)
    x1 = torch.randn(7, 3)
    coupling = sinkhorn_coupling(pairwise_squared_cost(x0, x1), epsilon=0.2, iterations=200)
    assert coupling.shape == (5, 7)
    assert torch.allclose(coupling.sum(1), torch.full((5,), 1 / 5), atol=2e-3)
    assert torch.allclose(coupling.sum(0), torch.full((7,), 1 / 7), atol=2e-3)
    i, j = sample_coupling_pairs(coupling, 4)
    xt, target, tau = cfm_interpolate(x0[i], x1[j], 8.5, 9.0)
    assert xt.shape == target.shape == (4, 3)
    assert tau.shape == (4, 1)


def test_coupling_pair_sampling_is_reproducible_and_respects_support():
    torch = pytest.importorskip("torch")
    from stream_model.ot import sample_coupling_pairs

    coupling = torch.tensor([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]])
    first = sample_coupling_pairs(coupling, 100, torch.Generator().manual_seed(23))
    second = sample_coupling_pairs(coupling, 100, torch.Generator().manual_seed(23))

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.all(coupling[first] > 0)


def test_ot_pool_can_emit_smaller_reproducible_pair_minibatch():
    torch = pytest.importorskip("torch")
    from stream_model.ot import ot_cfm_batch

    x0 = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    x1 = x0.flip(0)
    first = ot_cfm_batch(
        x0, x1, 0.0, 1.0, n_pairs=3, generator=torch.Generator().manual_seed(17)
    )
    second = ot_cfm_batch(
        x0, x1, 0.0, 1.0, n_pairs=3, generator=torch.Generator().manual_seed(17)
    )

    assert first[0].shape == first[1].shape == (3, 3)
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)


def test_partial_ot_transports_requested_mass_and_drops_expensive_match():
    torch = pytest.importorskip("torch")
    from stream_model.ot import partial_sinkhorn_coupling

    cost = torch.tensor([[0.0, 2.0], [2.0, 2.0]])
    coupling = partial_sinkhorn_coupling(cost, transported_mass=0.5, epsilon=0.05, iterations=500)

    assert float(coupling.sum()) == pytest.approx(0.5, abs=2e-3)
    assert float(coupling[0, 0] / coupling.sum()) > 0.99


def test_kl_unbalanced_ot_relaxes_marginals_around_expensive_cells():
    torch = pytest.importorskip("torch")
    from stream_model.ot import unbalanced_sinkhorn_coupling

    cost = torch.tensor([[0.0, 2.0], [2.0, 2.0]])
    coupling = unbalanced_sinkhorn_coupling(cost, marginal_relaxation=0.1, epsilon=0.05, iterations=500)

    assert torch.isfinite(coupling).all()
    assert float(coupling.sum()) > 0
    assert float(coupling[0, 0] / coupling.sum()) > 0.99
    assert not torch.allclose(coupling.sum(1), torch.full((2,), 0.5), atol=0.05)


def test_ot_cfm_interpolates_auxiliary_state_with_expression_pairs():
    torch = pytest.importorskip("torch")
    from stream_model.ot import ot_cfm_batch_with_state

    x0 = torch.tensor([[0.0, 1.0]])
    x1 = torch.tensor([[2.0, 3.0]])
    state0 = torch.tensor([[10.0, 20.0, 30.0]])
    state1 = torch.tensor([[30.0, 40.0, 50.0]])
    xt, target, tau, state_t = ot_cfm_batch_with_state(x0, x1, state0, state1, 8.5, 9.0)
    assert xt.shape == target.shape == (1, 2)
    assert state_t.shape == (1, 3)
    assert torch.allclose(state_t, (1 - tau) * state0 + tau * state1)


@pytest.mark.parametrize("variant", ["standard_cfm", "film", "cross_attention"])
def test_model_forward_variants(variant):
    torch = pytest.importorskip("torch")
    from stream_model.models import StandardCFM, StreamModel

    batch = 2
    genes = 4
    state_dim = genes + 1
    x = torch.randn(batch, state_dim)
    if variant == "standard_cfm":
        model = StandardCFM(n_genes=genes, hidden_dim=16, n_layers=1, state_dim=state_dim)
        out = model(x)
    else:
        model = StreamModel(
            n_genes=genes,
            cre_dim=8,
            d_model=16,
            n_heads=4,
            n_layers=1,
            variant="cross_attention" if variant == "cross_attention" else "film",
            positional_encoding="rope",
            n_context_tokens=2,
            state_dim=state_dim,
        )
        cre_embeddings = torch.randn(genes, 3, 8)
        mask = torch.ones(genes, 3, dtype=torch.bool)
        signed_distance = torch.tensor([[0, 1000, -2000]] * genes, dtype=torch.float32)
        is_promoter = torch.zeros(genes, 3, dtype=torch.bool)
        is_promoter[:, 0] = True
        out = model(x, cre_embeddings, mask, signed_distance, is_promoter)
    assert out.shape == (batch, genes)


@pytest.mark.parametrize("variant", ["film", "cross_attention"])
def test_stream_chunked_prediction_matches_full_forward(variant):
    torch = pytest.importorskip("torch")
    from stream_model.models import StreamModel
    from stream_model.train import predict_stream_chunked, stream_chunked_loss

    batch = 2
    genes = 7
    model = StreamModel(
        n_genes=genes,
        cre_dim=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        variant=variant,
        positional_encoding="rope",
        n_context_tokens=2,
    )
    model.eval()
    x = torch.randn(batch, genes)
    target = torch.randn(batch, genes)
    cre_inputs = {
        "cre_embeddings": torch.randn(genes, 3, 8),
        "cre_mask": torch.ones(genes, 3, dtype=torch.bool),
        "signed_distance": torch.tensor([[0, 1000, -2000]] * genes, dtype=torch.float32),
        "is_promoter": torch.zeros(genes, 3, dtype=torch.bool),
    }
    cre_inputs["is_promoter"][:, 0] = True

    full = model(x, **cre_inputs)
    chunked = predict_stream_chunked(model, x, cre_inputs, gene_chunk_size=3)
    chunked_loss = stream_chunked_loss(model, x, target, cre_inputs, gene_chunk_size=3)
    full_loss = torch.mean((full - target) ** 2)
    subset_indices = torch.tensor([0, 2, 5], dtype=torch.long)
    subset_loss = stream_chunked_loss(
        model,
        x,
        target,
        cre_inputs,
        gene_chunk_size=2,
        loss_gene_indices=subset_indices,
    )
    manual_subset_loss = torch.mean((full.index_select(1, subset_indices) - target.index_select(1, subset_indices)) ** 2)

    assert torch.allclose(chunked, full, atol=1e-5)
    assert torch.allclose(chunked_loss, full_loss, atol=1e-5)
    assert torch.allclose(subset_loss, manual_subset_loss, atol=1e-5)


def test_fixed_validation_reports_scale_normalized_loss():
    torch = pytest.importorskip("torch")
    from stream_model.train import FixedValidationBatch, evaluate_fixed_validation, validation_velocity_scale

    class ZeroModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x):
            return torch.zeros_like(x) + self.weight

    batch = FixedValidationBatch(
        day0="E8.5",
        day1="E8.75",
        t0=8.5,
        t1=8.75,
        state_t=torch.zeros((2, 2)),
        target=torch.tensor([[1.0, 10.0], [1.0, 10.0]]),
        x0=torch.zeros((2, 2)),
        x1=torch.ones((2, 2)),
    )
    scale = validation_velocity_scale([batch])
    metrics = evaluate_fixed_validation(ZeroModel(), [batch], scale, None, gene_chunk_size=2)

    assert metrics["val_loss_raw"] == pytest.approx(50.5)
    assert metrics["val_loss_normalized"] == pytest.approx(1.0)


def test_pca_reconstruction_is_nonnegative_and_gene_valued():
    from stream_model.denoise import PCADenoiser

    pca = PCADenoiser(
        components=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        mean=np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
        explained_variance=np.ones(2, dtype=np.float32),
    )
    counts = np.asarray([[10.0, 2.0, 0.0], [0.0, 4.0, 8.0]], dtype=np.float32)
    coordinates = pca.transform_counts(counts)
    reconstructed = pca.reconstruct_counts(counts)

    assert coordinates.shape == (2, 2)
    assert reconstructed.shape == counts.shape
    assert np.isfinite(reconstructed).all()
    assert (reconstructed >= 0).all()


def test_interval_pair_bank_interleaves_intervals_before_repeating():
    torch = pytest.importorskip("torch")
    from stream_model.data import IntervalBatch
    from stream_model.pair_bank import IntervalPairBank

    class Config:
        seed = 7
        batch_size = 2
        ot_pairs_per_pool = 4
        ot_method = "balanced"
        ot_epsilon = 0.1
        ot_iterations = 20
        ot_partial_mass = 0.95
        ot_marginal_relaxation = 0.1
        ot_cost_space = "expression"
        endpoint_denoising = "none"

    class Sampler:
        intervals = [("0", "1"), ("1", "2"), ("2", "3")]

        def __init__(self):
            self.calls = {interval: 0 for interval in self.intervals}

        def sample_interval(self, day0, day1):
            self.calls[(day0, day1)] += 1
            return IntervalBatch(
                x0=np.zeros((6, 3), dtype=np.float32),
                x1=np.ones((6, 3), dtype=np.float32),
                t0=float(day0),
                t1=float(day1),
                day0=day0,
                day1=day1,
            )

    sampler = Sampler()
    bank = IntervalPairBank(Config(), sampler, torch.device("cpu"))
    first_round = []
    for _ in range(3):
        batch = bank.next()
        first_round.append((batch.day0, batch.day1))
    second_round = []
    for _ in range(3):
        batch = bank.next()
        second_round.append((batch.day0, batch.day1))

    assert set(first_round) == set(sampler.intervals)
    assert set(second_round) == set(sampler.intervals)
    assert sampler.calls == {interval: 1 for interval in sampler.intervals}


def test_pair_bank_keeps_raw_state_separate_from_pca_denoised_targets():
    torch = pytest.importorskip("torch")
    from stream_model.data import IntervalBatch
    from stream_model.pair_bank import IntervalPairBank

    class Config:
        seed = 9
        batch_size = 2
        ot_pairs_per_pool = 2
        ot_method = "balanced"
        ot_epsilon = 0.1
        ot_iterations = 20
        ot_partial_mass = 0.95
        ot_marginal_relaxation = 0.1
        ot_cost_space = "pca"
        endpoint_denoising = "pca"

    class PCA:
        def transform_counts(self, values):
            return np.asarray(values[:, :2], dtype=np.float32)

        def reconstruct_counts(self, values):
            return np.asarray(values + 10.0, dtype=np.float32)

    class Sampler:
        intervals = [("0", "1")]

        def sample_interval(self, day0, day1):
            return IntervalBatch(
                x0=np.zeros((4, 3), dtype=np.float32),
                x1=np.ones((4, 3), dtype=np.float32),
                t0=0.0,
                t1=1.0,
                day0=day0,
                day1=day1,
            )

    batch = IntervalPairBank(Config(), Sampler(), torch.device("cpu"), pca=PCA()).next()

    assert torch.all(batch.raw_x0 == 0)
    assert torch.all(batch.raw_x1 == 1)
    assert torch.all(batch.target_x0 == 10)
    assert torch.all(batch.target_x1 == 11)


def test_train_steps_stops_after_validation_patience():
    torch = pytest.importorskip("torch")
    from stream_model.data import IntervalBatch
    from stream_model.train import FixedValidationBatch, train_steps

    class Config:
        epochs = 10
        seed = 3
        batch_size = 2
        ot_epsilon = 0.1
        ot_iterations = 5
        gene_chunk_size = 2
        model_variant = "standard_cfm"
        cell_state = "expression"

    class Sampler:
        def sample(self):
            return IntervalBatch(
                x0=np.zeros((2, 2), dtype=np.float32),
                x1=np.ones((2, 2), dtype=np.float32),
                t0=0.0,
                t1=1.0,
                day0="0",
                day1="1",
            )

    class ConstantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x):
            return torch.zeros_like(x) + self.value

    fixed = FixedValidationBatch(
        day0="0",
        day1="1",
        t0=0.0,
        t1=1.0,
        state_t=torch.zeros((2, 2)),
        target=torch.ones((2, 2)),
        x0=torch.zeros((2, 2)),
        x1=torch.ones((2, 2)),
    )
    model = ConstantModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    result = train_steps(
        Config(),
        Sampler(),
        model,
        optimizer,
        steps_per_epoch=1,
        validation_batches=[fixed],
        early_stopping_patience=2,
        rollout_every_validations=0,
    )

    assert result.stopped_early
    assert result.best_epoch == 0
    assert len(result.train_metrics) == 3
    assert len(result.validation_metrics) == 3


def test_train_steps_reuses_large_ot_pool_across_model_minibatches():
    torch = pytest.importorskip("torch")
    from stream_model.data import IntervalBatch
    from stream_model.train import train_steps

    class Config:
        epochs = 1
        seed = 5
        batch_size = 2
        ot_epsilon = 0.1
        ot_iterations = 10
        ot_method = "balanced"
        ot_partial_mass = 0.95
        ot_marginal_relaxation = 0.1
        ot_pairs_per_pool = 4
        gene_chunk_size = 2
        model_variant = "standard_cfm"
        cell_state = "expression"

    class Sampler:
        calls = 0

        def sample(self):
            self.calls += 1
            return IntervalBatch(
                x0=np.zeros((6, 2), dtype=np.float32),
                x1=np.ones((6, 2), dtype=np.float32),
                t0=0.0,
                t1=1.0,
                day0="0",
                day1="1",
            )

    class ConstantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x):
            return torch.zeros_like(x) + self.value

    sampler = Sampler()
    model = ConstantModel()
    result = train_steps(
        Config(),
        sampler,
        model,
        torch.optim.SGD(model.parameters(), lr=0.0),
        steps_per_epoch=4,
    )

    assert sampler.calls == 2
    assert [row["ot_pool_refill"] for row in result.train_metrics] == [0, 0, 1, 1]


@pytest.mark.parametrize("variant", ["film", "cross_attention"])
def test_stream_conditioning_is_layerwise(variant):
    torch = pytest.importorskip("torch")
    from torch import nn
    from stream_model.models import StreamModel

    model = StreamModel(
        n_genes=5,
        cre_dim=8,
        d_model=16,
        n_heads=4,
        n_layers=3,
        variant=variant,
        n_context_tokens=2,
    )

    assert isinstance(model.cre_encoder_layers, nn.ModuleList)
    assert len(model.cre_encoder_layers) == 3
    assert isinstance(model.cell_context, nn.ModuleList)
    assert len(model.cell_context) == 3
    if variant == "cross_attention":
        assert isinstance(model.cross_attn, nn.ModuleList)
        assert len(model.cross_attn) == 3
    else:
        assert model.cross_attn is None


def test_uce_sentence_uses_chromosome_delimiters_and_sorted_positions():
    sparse = pytest.importorskip("scipy.sparse")
    from stream_model.uce import UCE_CHROM_CLOSE_TOKEN, UCE_CHROM_TOKEN_OFFSET, UCE_CLS_TOKEN, UCEGeneMetadata, sample_uce_sentence

    row = sparse.csr_matrix([[1.0, 8.0, 2.0, 4.0]])[0]
    metadata = UCEGeneMetadata(
        token_ids=np.array([10, 11, 12, -1]),
        chrom_ids=np.array([1, 0, 1, -1]),
        starts=np.array([30, 20, 10, -1]),
    )
    sentence = sample_uce_sentence(row, metadata, np.random.default_rng(2), sample_size=12)
    assert sentence is not None
    assert sentence[0] == UCE_CLS_TOKEN
    assert np.count_nonzero(sentence == UCE_CHROM_CLOSE_TOKEN) == 2
    assert set(sentence[1:]) >= {UCE_CHROM_TOKEN_OFFSET, UCE_CHROM_TOKEN_OFFSET + 1, 10, 11, 12}


def test_dense_uce_sentence_accepts_fractional_counts_and_is_reproducible():
    from stream_model.uce import UCEGeneMetadata, sample_dense_uce_sentence

    metadata = UCEGeneMetadata(
        token_ids=np.array([10, 11, 12]),
        chrom_ids=np.array([0, 0, 1]),
        starts=np.array([30, 10, 20]),
    )
    values = np.array([0.25, 3.5, 1.25], dtype=np.float32)
    first = sample_dense_uce_sentence(values, metadata, np.random.default_rng(7), sample_size=16)
    second = sample_dense_uce_sentence(values, metadata, np.random.default_rng(7), sample_size=16)
    np.testing.assert_array_equal(first, second)
    assert first is not None
    assert len(first) >= 17


def test_projected_euler_rollout_is_autonomous_and_nonnegative():
    torch = pytest.importorskip("torch")
    from stream_model.rollout import projected_euler_rollout

    seen_seeds = []

    def velocity(x, seed):
        seen_seeds.append(seed)
        return torch.tensor([[-2.0, 1.0]], dtype=x.dtype)

    result = projected_euler_rollout(
        torch.tensor([[0.5, 0.0]]),
        0.0,
        1.0,
        velocity,
        steps=2,
        seed=19,
    )
    assert torch.allclose(result, torch.tensor([[0.0, 1.0]]))
    assert seen_seeds == [19, 19]


def test_endpoint_metrics_are_exact_for_identical_predictions():
    from stream_model.rollout import mean_shift_metrics, sinkhorn_divergence

    x0 = np.array([[0.0, 1.0], [1.0, 0.0]])
    x1 = np.array([[1.0, 3.0], [2.0, 2.0]])
    metrics = mean_shift_metrics(x0, x1, x1)
    assert metrics["mean_shift_r2"] == pytest.approx(1.0)
    assert metrics["mean_shift_mae"] == pytest.approx(0.0)
    assert metrics["mean_gene_wasserstein1"] == pytest.approx(0.0)
    assert sinkhorn_divergence(x1, x1, epsilon=0.1) == pytest.approx(0.0, abs=1e-8)


def test_evaluate_intervals_reports_full_and_subset_gene_sets():
    torch = pytest.importorskip("torch")
    from stream_model.data import IntervalBatch
    from stream_model.evaluate import evaluate_intervals

    class Config:
        ot_epsilon = 0.1
        ot_iterations = 10

    class Sampler:
        def sample(self):
            return IntervalBatch(
                x0=np.zeros((4, 3), dtype=np.float32),
                x1=np.ones((4, 3), dtype=np.float32),
                t0=8.5,
                t1=9.0,
                day0="E8.5",
                day1="E9.0",
            )

    class ZeroModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x):
            return torch.zeros_like(x) + self.weight

    metrics = evaluate_intervals(
        Config(),
        Sampler(),
        ZeroModel(),
        n_batches=2,
        eval_gene_sets={"full": None, "legacy": [0, 2]},
    )
    assert set(metrics["eval_gene_set"]) == {"full", "legacy"}
    assert set(metrics.groupby("eval_gene_set")["n_eval_genes"].first().to_dict().items()) == {
        ("full", 3),
        ("legacy", 2),
    }
    assert len(metrics) == 4
    assert {"displacement_mse", "displacement_mae", "displacement_r2"}.issubset(metrics.columns)


def test_evaluate_intervals_reports_displacement_r2():
    torch = pytest.importorskip("torch")
    from stream_model.data import IntervalBatch
    from stream_model.evaluate import evaluate_intervals

    class Config:
        ot_epsilon = 0.1
        ot_iterations = 10

    class Sampler:
        def sample(self):
            return IntervalBatch(
                x0=np.zeros((2, 3), dtype=np.float32),
                x1=np.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float32),
                t0=0.0,
                t1=1.0,
                day0="0",
                day1="1",
            )

    class ZeroModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x):
            return torch.zeros_like(x) + self.weight

    metrics = evaluate_intervals(Config(), Sampler(), ZeroModel(), n_batches=1)
    assert metrics.loc[0, "displacement_r2"] == pytest.approx(-1.5)


def test_evaluate_intervals_uses_auxiliary_state_with_expression_target():
    torch = pytest.importorskip("torch")
    from stream_model.data import IntervalBatch
    from stream_model.evaluate import evaluate_intervals

    class Config:
        ot_epsilon = 0.1
        ot_iterations = 10
        cell_state = "uce"

    class Sampler:
        def sample(self):
            return IntervalBatch(
                x0=np.zeros((2, 3), dtype=np.float32),
                x1=np.ones((2, 3), dtype=np.float32),
                state0=np.zeros((2, 5), dtype=np.float32),
                state1=np.ones((2, 5), dtype=np.float32),
                t0=8.5,
                t1=9.0,
                day0="E8.5",
                day1="E9.0",
            )

    model = torch.nn.Linear(5, 3)
    metrics = evaluate_intervals(Config(), Sampler(), model, n_batches=1)
    assert metrics.loc[0, "cell_state"] == "uce"
    assert metrics.loc[0, "n_eval_genes"] == 3


def test_zero_velocity_baseline_is_a_persistence_comparator():
    from stream_model.data import IntervalBatch
    from stream_model.evaluate import ZeroVelocityBaseline, evaluate_intervals

    class Config:
        ot_epsilon = 0.1
        ot_iterations = 10

    class Sampler:
        def sample(self):
            return IntervalBatch(
                x0=np.zeros((2, 3), dtype=np.float32),
                x1=np.ones((2, 3), dtype=np.float32),
                t0=0.0,
                t1=1.0,
                day0="0",
                day1="1",
            )

    metrics = evaluate_intervals(Config(), Sampler(), ZeroVelocityBaseline(3), n_batches=1)
    assert metrics.loc[0, "displacement_mae"] == pytest.approx(1.0)
