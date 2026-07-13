from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stream_model.data import adjacent_intervals, build_time_coordinates, canonical_day_label
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


def test_adjacent_intervals_excludes_heldout_days():
    days = ["E8.5", "E9.0", "E9.5", "E10.0"]
    assert adjacent_intervals(days, {"E9.5"}) == [("E8.5", "E9.0")]


def test_time_coordinates_support_physical_days_and_relative_scaling():
    stages = ["18", "36", "72"]
    physical = build_time_coordinates(stages, "physical_days", value_scale=1 / 24)
    relative = build_time_coordinates(stages, "relative", value_scale=1 / 24)
    assert physical == {"18": 0.75, "36": 1.5, "72": 3.0}
    assert relative == {"18": 0.0, "36": 1 / 3, "72": 1.0}
    assert canonical_day_label(36.0) == "36"
    assert canonical_day_label("E9.0") == "E9.0"


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

    assert torch.allclose(chunked, full, atol=1e-5)
    assert torch.allclose(chunked_loss, full_loss, atol=1e-5)


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
    assert {"displacement_mse", "displacement_mae"}.issubset(metrics.columns)


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
