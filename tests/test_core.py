"""Automated tests for PromptOpt-SLM core modules.

Runnable with pytest (`pytest tests/`) or without it (`python tests/run_tests.py`).
Tests avoid fixtures so the bundled runner can call them directly.
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
try:  # pytest is optional: a minimal shim keeps the suite runnable without it
    import pytest
except ImportError:  # pragma: no cover
    import contextlib

    class _Raises(contextlib.AbstractContextManager):
        def __init__(self, exc):
            self.exc = exc

        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError(f"expected {self.exc.__name__}")
            return issubclass(et, self.exc)

    class pytest:  # type: ignore[no-redef]
        raises = staticmethod(lambda exc: _Raises(exc))

        @staticmethod
        def approx(x, abs=1e-6):
            return _Approx(x, abs)

    class _Approx:
        def __init__(self, x, tol):
            self.x, self.tol = x, tol

        def __eq__(self, other):
            return abs(other - self.x) <= self.tol

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import ConfigError, load_profile                            # noqa: E402
from core.adaptive_evolution import AdaptiveEvolver                             # noqa: E402
from core.calibration import contextual_calibrate, prior_match, softmax_rows, estimate_bias  # noqa: E402
from core.dataset_engine import DatasetEngine, DatasetError, normalise_text     # noqa: E402
from core.demo_selector import DemoSelector                                     # noqa: E402
from core.evaluation import (bootstrap_ci, brier_score, classification_metrics,  # noqa: E402
                             expected_calibration_error)
from core.experiment_store import ExperimentStore                               # noqa: E402
from core.model_engine import ModelUnavailableError, HFBackend, SimulatedBackend, build_lexicon  # noqa: E402
from core.prompt_generator import PromptConfig, build_prompt, generate_instructions, generate_verbalizers  # noqa: E402
from core.prompt_optimizer import Budget, BudgetExceededError, ObjectiveWeights, PromptOptimizer, objective  # noqa: E402
from core.robustness import RobustnessLab, UncertaintyGate, typo, word_dropout  # noqa: E402

CSV = ROOT / "data" / "demo_support_tickets.csv"
CLASSES = ["billing", "technical", "account_access"]


def _engine() -> DatasetEngine:
    return DatasetEngine.from_csv(CSV)


def _cfg(calibration="none", demos=None) -> PromptConfig:
    return PromptConfig(instruction="Classify the ticket.",
                        verbalizer={0: "billing", 1: "technical", 2: "account_access"},
                        demos=demos or [("refund for invoice", 0), ("app crashes", 1), ("locked out", 2)],
                        calibration=calibration)


# ------------------------------------------------------------------ config
def test_profiles_load_and_validate():
    for name in ("demo", "research", "full_benchmark"):
        assert load_profile(name).name == name
    with pytest.raises(ConfigError):
        load_profile("nonexistent")


# ----------------------------------------------------------- dataset engine
def test_quality_report_detects_issues():
    q = _engine().quality_report()
    assert q.n_missing_text == 6 and q.n_missing_label == 4
    assert q.n_duplicates > 0
    assert 0 < q.quality_score < 100
    assert set(q.class_counts) == set(CLASSES)


def test_clean_removes_missing_and_duplicates():
    e = _engine().clean()
    assert e.data["text"].notna().all() and (e.data["text"] != "").all()
    assert not e.data["text"].str.lower().duplicated().any()
    assert set(e.classes) == set(CLASSES)


def test_split_is_stratified_and_disjoint():
    e = _engine()
    s = e.split(val_per_class=16, test_fraction=0.3, seed=0)
    assert s["val"].groupby("label").size().eq(16).all()
    val_t, test_t = set(s["val"]["text"]), set(s["test"]["text"])
    assert not (val_t & set(s["pool"]["text"])), "validation must not overlap support pool"
    assert not (val_t & test_t), "test must not overlap validation"


def test_few_shot_sample_is_balanced():
    e = _engine()
    s = e.split(16, seed=1)
    demos = e.few_shot_sample(s["pool"], shots_per_class=4, seed=1)
    labels = [y for _, y in demos]
    assert len(demos) == 12 and all(labels.count(c) == 4 for c in range(3))


def test_missing_columns_raise():
    with pytest.raises(DatasetError):
        DatasetEngine(pd.DataFrame({"a": [1], "b": [2]}))


def test_json_loader_accepts_wrapped_records():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        p.write_text(json.dumps({"data": [{"text": "hi there", "label": "a"}]}), encoding="utf-8")
        assert len(DatasetEngine.from_json(p).clean().data) == 1


def test_normalise_text_handles_missing():
    assert normalise_text(None) == "" and normalise_text("  a   b ") == "a b"


# ---------------------------------------------------------------- calibration
def test_contextual_calibration_uniformises_content_free_input():
    bias = np.array([0.8, 0.2])
    out = contextual_calibrate(np.array([bias]), bias)
    assert np.allclose(out, [[0.5, 0.5]])


def test_softmax_rows_sum_to_one():
    p = softmax_rows(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_prior_match_gives_uniform_mean():
    rng = np.random.default_rng(0)
    p = softmax_rows(rng.normal(size=(200, 3)) + np.array([1.5, 0, 0]))
    q = prior_match(p)
    assert np.allclose(q.mean(axis=0), 1 / 3, atol=1e-2)


def test_estimate_bias_is_a_distribution():
    b = estimate_bias(lambda texts: np.tile([2.0, 0.0], (len(texts), 1)))
    assert np.isclose(b.sum(), 1.0) and b[0] > b[1]


# ------------------------------------------------------------ prompt generator
def test_build_prompt_ends_with_open_answer_slot():
    prompt = build_prompt(_cfg(), "my card was charged twice")
    assert prompt.endswith("Label:")
    assert "my card was charged twice" in prompt


def test_signature_is_stable_and_sensitive():
    assert _cfg().signature() == _cfg().signature()
    assert _cfg().signature() != _cfg(calibration="prior").signature()


def test_verbalizer_words_must_be_distinct():
    with pytest.raises(ValueError):
        PromptConfig(instruction="x", verbalizer={0: "a", 1: "a"})


def test_generate_verbalizers_puts_default_first_and_filters_collisions():
    syn = {"billing": ["invoice"], "technical": ["tech"], "account_access": ["login"]}
    cands = generate_verbalizers(CLASSES, syn, lambda w: w[:2], max_candidates=8)
    assert cands[0] == {0: "billing", 1: "technical", 2: "account_access"}
    for c in cands:
        firsts = [w[:2] for w in c.values()]
        assert len(set(firsts)) == len(firsts)


def test_generate_instructions_include_label_list():
    ins = generate_instructions("the category", CLASSES)
    assert any("billing" in s for s in ins) and len(ins) >= 8


# ----------------------------------------------------------------- selectors
def _selector():
    e = _engine()
    s = e.split(16, seed=0)
    support = e.few_shot_sample(s["pool"], 4, 0)
    return DemoSelector(support, 3), s


def test_balanced_selection_is_class_balanced():
    sel, _ = _selector()
    chosen = sel.balanced(6, random.Random(0))
    assert sorted(y for _, y in chosen) == [0, 0, 1, 1, 2, 2]


def test_mmr_respects_class_cap_and_size():
    sel, s = _selector()
    q = s["val"]["text"].iloc[0]
    chosen = sel.mmr(q, m=6, alpha=0.7)
    assert len(chosen) == 6
    labels = [y for _, y in chosen]
    assert max(labels.count(c) for c in range(3)) <= 2


def test_mmr_orders_most_similar_last():
    sel, s = _selector()
    q = s["val"]["text"].iloc[0]
    chosen = sel.mmr(q, m=6)
    qv = sel.embedder.encode([q])[0]
    sims = [float(sel.embedder.encode([t])[0] @ qv) for t, _ in chosen]
    assert sims[-1] == max(sims)


def test_mmr_rejects_bad_alpha():
    sel, _ = _selector()
    with pytest.raises(ValueError):
        sel.mmr("x", m=3, alpha=1.5)


# -------------------------------------------------------------- model engine
def _sim_backend():
    e = _engine()
    s = e.split(16, seed=0)
    lex = build_lexicon(s["pool"]["text"].tolist(),
                        [CLASSES.index(l) for l in s["pool"]["label"]], 3)
    return SimulatedBackend(lex), s


def test_simulated_backend_is_deterministic():
    b, s = _sim_backend()
    cfg = _cfg()
    prompts = [build_prompt(cfg, t) for t in s["val"]["text"][:5]]
    a = b.label_logits(prompts, cfg.verbalizer)
    c = b.label_logits(prompts, cfg.verbalizer)
    assert np.array_equal(a, c) and a.shape == (5, 3)


def test_simulated_backend_flags_itself_as_simulated():
    b, _ = _sim_backend()
    assert b.is_simulated and b.describe()["simulated"] is True


def test_hf_backend_raises_clear_error_without_torch():
    saved = sys.modules.get("torch")
    sys.modules["torch"] = None  # forces ImportError on import
    try:
        with pytest.raises(ModelUnavailableError):
            HFBackend("any/model")
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


# ------------------------------------------------------------------ optimiser
def test_objective_penalises_instability_and_tokens():
    w = ObjectiveWeights(stability_lambda=0.5, token_mu=0.02, token_ref=200, latency_nu=0)
    assert objective(0.9, 0.0, 100, 0, w) > objective(0.9, 0.1, 100, 0, w)
    assert objective(0.9, 0.0, 100, 0, w) > objective(0.9, 0.0, 400, 0, w)


def test_budget_is_enforced():
    b = Budget(max_forward_passes=10)
    b.charge(8)
    with pytest.raises(BudgetExceededError):
        b.charge(5)
    assert b.used == 8


def _optimizer(budget=10_000):
    be, s = _sim_backend()
    texts = s["val"]["text"].tolist()
    labels = [CLASSES.index(l) for l in s["val"]["label"]]
    return PromptOptimizer(be, texts, labels, Budget(budget), n_permutations=2, seed=0), s


def test_successive_halving_returns_a_candidate():
    opt, s = _optimizer()
    sel, _ = _selector()
    rng = random.Random(0)
    pool = [PromptConfig(instruction=f"Classify {i}.", verbalizer={0: "billing", 1: "technical", 2: "account_access"},
                         demos=sel.balanced(6, rng), calibration=rng.choice(["none", "contextual"]))
            for i in range(8)]
    res = opt.successive_halving(pool, fractions=(0.25, 0.5, 1.0))
    assert res.best.signature() in {c.signature() for c in pool}
    assert res.leaderboard[0]["rank"] == 1 and res.forward_passes > 0


def test_optimizer_stops_when_budget_is_exceeded():
    opt, _ = _optimizer(budget=20)
    with pytest.raises(BudgetExceededError):
        opt.full_evaluate(_cfg())


# ------------------------------------------------------------ evolution
def test_evolution_children_are_valid_configs():
    sel, _ = _selector()
    ev = AdaptiveEvolver(["Classify x."], [{0: "billing", 1: "technical", 2: "account_access"}],
                         sel.support, n_demos=6, seed=0, mutation_rate=0.9)
    parent = _cfg()
    for _ in range(20):
        child = ev.mutate(parent)
        assert len(set(child.verbalizer.values())) == 3
    other = _cfg(calibration="prior")
    child = ev.recombine(parent, other)
    assert child.template_id == "recomb" and len(child.demos) <= 6


def test_evolver_rejects_bad_rate():
    with pytest.raises(ValueError):
        AdaptiveEvolver(["x"], [{0: "a"}], [("t", 0)], n_demos=1, mutation_rate=2.0)


# ------------------------------------------------------------- robustness
def test_perturbations_keep_text_non_empty():
    rng = random.Random(0)
    assert typo("the invoice is wrong", 1.0, rng).split()
    assert word_dropout("a b c", 1.0, rng)  # never returns empty


def test_fragility_index_zero_when_unchanged():
    df = pd.DataFrame([{"perturbation": "clean", "level": 0, "macro_f1": 0.9, "drop": 0.0},
                       {"perturbation": "typo", "level": 0.1, "macro_f1": 0.9, "drop": 0.0}])
    pfi, _ = RobustnessLab.fragility_index(df)
    assert pfi == 0.0


def test_uncertainty_gate_flags_low_margin():
    gate = UncertaintyGate(tau=0.6, delta=0.2)
    probs = np.array([[0.9, 0.05, 0.05], [0.45, 0.4, 0.15]])
    assert gate.decide(probs).tolist() == [True, False]
    res = gate.evaluate(probs, [0, 0])
    assert res["review_rate"] == 0.5


# ------------------------------------------------------------- evaluation
def test_perfect_predictions_have_zero_brier_and_ece():
    probs = np.eye(3)
    y = [0, 1, 2]
    assert brier_score(probs, y) == 0.0
    ece, _ = expected_calibration_error(probs, y)
    assert ece == pytest.approx(0.0)


def test_classification_metrics_keys():
    m = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0])
    assert {"accuracy", "f1_macro", "f1_weighted", "recall_macro"} <= set(m)
    assert m["accuracy"] == pytest.approx(0.75)


def test_bootstrap_ci_brackets_point_estimate():
    y = np.array([0, 1] * 50)
    pred = y.copy()
    pred[:10] = 1 - pred[:10]
    point, lo, hi = bootstrap_ci(y, pred, n_boot=300, seed=0)
    assert lo <= point <= hi


# ------------------------------------------------------------- store
def test_experiment_store_round_trip():
    with tempfile.TemporaryDirectory() as d:
        with ExperimentStore(Path(d) / "t.db") as store:
            rid = store.log_run("demo", "x.csv", "simulated", "m", {"a": 1}, {"f1": 0.5})
            store.log_prompt_version(rid, {"signature": "abc", "template_id": "t", "calibration": "none",
                                           "n_demos": 3, "f1": 0.8, "std": 0.01, "J": 0.79, "config": {}})
            board = store.leaderboard(run_id=rid)
            assert board[0]["signature"] == "abc"
            assert store.runs()[0]["id"] == rid
            out = store.export_csv(Path(d) / "lb.csv")
            assert out.exists()
