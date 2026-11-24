"""DEMO-profile experiment runner.

Produces, from the bundled synthetic corpus:
  * MEASURED   : TF-IDF + logistic regression baseline (real sklearn fit)
  * MEASURED   : dataset quality report, class distribution, text lengths
  * SIMULATED  : prompt-optimisation pipeline run on the SimulatedBackend
                 (plumbing check only - NOT an SLM result)

Every output file name and figure title carries its provenance tag.
Run:  python -m experiments.run_demo
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import load_profile                       # noqa: E402
from core.calibration import estimate_bias                     # noqa: E402
from core.dataset_engine import DatasetEngine                  # noqa: E402
from core.demo_selector import DemoSelector                    # noqa: E402
from core.evaluation import classification_metrics, expected_calibration_error, brier_score  # noqa: E402
from core.experiment_store import ExperimentStore              # noqa: E402
from core.model_engine import SimulatedBackend, SLMClassifier, build_lexicon  # noqa: E402
from core.prompt_generator import (PromptConfig, build_prompt, generate_instructions,  # noqa: E402
                                   generate_verbalizers)
from core.prompt_optimizer import Budget, PromptOptimizer     # noqa: E402
from core.robustness import RobustnessLab, UncertaintyGate    # noqa: E402

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
SYNONYMS = {"billing": ["invoice", "payment"], "technical": ["tech", "bug"],
            "account_access": ["login", "access"]}
TASK_NOUN = "the category of the support ticket"


def tag_fig(ax_title: str, tag: str) -> str:
    return f"[{tag}] {ax_title}"


def run_seed(seed: int, prof, engine: DatasetEngine, classes: list) -> dict:
    split = engine.split(prof.val_per_class, test_fraction=0.3, seed=seed)
    support = engine.few_shot_sample(split["pool"], prof.shots_per_class, seed)
    val_df, test_df = split["val"], split["test"]
    name_to_id = {c: i for i, c in enumerate(classes)}
    val_texts = val_df["text"].tolist()
    val_labels = [name_to_id[l] for l in val_df["label"]]
    test_texts = test_df["text"].tolist()
    test_labels = [name_to_id[l] for l in test_df["label"]]

    sup_texts = [t for t, _ in support]
    sup_labels = [y for _, y in support]
    lexicon = build_lexicon(split["pool"]["text"].tolist(),
                            [name_to_id[l] for l in split["pool"]["label"]], len(classes))
    backend = SimulatedBackend(lexicon, prior=np.array([0.6, 0.0, 0.0]))
    selector = DemoSelector(support, len(classes))
    rng = random.Random(seed)
    instructions = generate_instructions(TASK_NOUN, classes)
    verbalizers = generate_verbalizers(classes, SYNONYMS, lambda w: w.split()[0].lower(),
                                       max_candidates=8, seed=seed)
    pool = []
    while len(pool) < prof.n_candidates:
        pool.append(PromptConfig(
            instruction=rng.choice(instructions),
            verbalizer=dict(rng.choice(verbalizers)),
            demos=selector.balanced(2 * len(classes), rng),
            calibration=rng.choice(["none", "contextual", "prior"]),
            template_id="grammar"))
    budget = Budget(prof.max_forward_passes)
    opt = PromptOptimizer(backend, val_texts, val_labels, budget,
                          n_permutations=prof.stability_permutations, seed=seed)
    t0 = time.perf_counter()
    halving = opt.successive_halving(pool, fractions=prof.halving_fractions)
    halving_secs = time.perf_counter() - t0
    halving_passes = halving.forward_passes

    budget2 = Budget(prof.max_forward_passes)
    opt2 = PromptOptimizer(backend, val_texts, val_labels, budget2,
                           n_permutations=prof.stability_permutations, seed=seed)
    full = opt2.random_search_full(pool)

    manual = PromptConfig(instruction=instructions[0],
                          verbalizer={i: c for i, c in enumerate(classes)},
                          demos=selector.random(2 * len(classes), random.Random(seed)),
                          calibration="none", template_id="manual")
    zero_shot = PromptConfig(instruction=instructions[1],
                             verbalizer={i: c for i, c in enumerate(classes)},
                             demos=[], calibration="none", template_id="zero")
    best = halving.best
    uncal = PromptConfig.from_dict({**best.to_dict(), "calibration": "none"})
    variants = {"zero_shot": zero_shot, "manual_random": manual,
                "manual_calibrated": PromptConfig.from_dict({**manual.to_dict(), "calibration": "contextual"}),
                "optimised_uncalibrated": uncal, "optimised": best}
    results = {}
    probs_by_name = {}
    for name, cfg in variants.items():
        clf = SLMClassifier(backend, cfg)
        probs = clf.predict_proba(test_texts)
        probs_by_name[name] = probs
        m = classification_metrics(test_labels, probs.argmax(axis=1), labels=list(range(len(classes))))
        ece, _ = expected_calibration_error(probs, test_labels)
        m["ece"] = ece
        m["brier"] = brier_score(probs, test_labels)
        results[name] = m

    # TF-IDF + logistic regression baseline (MEASURED, real sklearn fit on support only)
    sup_df = pd.DataFrame({"text": sup_texts, "label": [classes[y] for y in sup_labels]})
    clf_lr = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
    clf_lr.fit(sup_df["text"], sup_df["label"])
    lr_pred = [name_to_id[p] for p in clf_lr.predict(test_texts)]
    results["tfidf_logreg"] = classification_metrics(test_labels, lr_pred, labels=list(range(len(classes))))

    # robustness + uncertainty on the optimised configuration (SIMULATED backend)
    lab = RobustnessLab(SLMClassifier(backend, best), test_texts, test_labels, seed=seed)
    rob = lab.run()
    pfi, per_type = RobustnessLab.fragility_index(rob)
    gate = UncertaintyGate(tau=0.6, delta=0.1).evaluate(probs_by_name["optimised"], test_labels)
    return {
        "seed": seed, "results": results, "best_config": best.to_dict(),
        "best_signature": best.signature(),
        "search": {"halving_forward_passes": halving_passes,
                   "full_forward_passes": full.forward_passes,
                   "halving_seconds": round(halving_secs, 3),
                   "full_seconds": round(full.wall_seconds, 3),
                   "halving_J": round(halving.best_eval.J, 4),
                   "full_J": round(full.best_eval.J, 4),
                   "halving_history": halving.history,
                   "leaderboard": halving.leaderboard[:10]},
        "robustness": {"pfi": pfi, "per_type": per_type.round(4).to_dict(),
                       "table": rob.to_dict(orient="records")},
        "uncertainty": gate,
        "_probs": probs_by_name["optimised"], "_labels": test_labels,
        "_lr_pred": lr_pred, "_test_texts": test_texts,
        "_forward_passes_total": budget.used + budget2.used,
    }


def main() -> None:
    prof = load_profile("demo")
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    engine = DatasetEngine.from_csv(ROOT / prof.dataset).clean()
    quality = DatasetEngine.from_csv(ROOT / prof.dataset).quality_report()
    classes = engine.classes
    sim_summary = engine.semantic_similarity_summary()

    runs = [run_seed(s, prof, engine, classes) for s in prof.seeds]

    # ----- aggregate MEASURED baseline and SIMULATED pipeline
    agg = {}
    for name in runs[0]["results"]:
        f1s = [r["results"][name]["f1_macro"] for r in runs]
        accs = [r["results"][name]["accuracy"] for r in runs]
        agg[name] = {"f1_macro_mean": float(np.mean(f1s)), "f1_macro_std": float(np.std(f1s)),
                     "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs))}
    provenance = {
        "tfidf_logreg": "MEASURED (sklearn fit on bundled synthetic corpus)",
        "zero_shot": "SIMULATED backend", "manual_random": "SIMULATED backend",
        "manual_calibrated": "SIMULATED backend", "optimised_uncalibrated": "SIMULATED backend",
        "optimised": "SIMULATED backend",
    }
    summary = {
        "profile": prof.name, "backend": "simulated (lexicon heuristic, NOT an SLM)",
        "dataset": "bundled synthetic support-ticket corpus", "quality": quality.to_dict(),
        "similarity": sim_summary, "aggregate": agg, "provenance": provenance,
        "fragility_index_mean": float(np.mean([r["robustness"]["pfi"] for r in runs])),
        "uncertainty_mean": {k: float(np.nanmean([r["uncertainty"][k] for r in runs]))
                             for k in ("review_rate", "coverage", "accuracy_all", "selective_accuracy")},
        "seeds": list(prof.seeds),
    }
    (RESULTS / "demo_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    pd.DataFrame([{"method": k, **v, "provenance": provenance[k]} for k, v in agg.items()]) \
        .to_csv(RESULTS / "demo_method_comparison.csv", index=False)
    per_seed = [{"seed": r["seed"], **{f"{m}_f1": r["results"][m]["f1_macro"] for m in r["results"]}}
                for r in runs]
    pd.DataFrame(per_seed).to_csv(RESULTS / "demo_per_seed.csv", index=False)
    rob_all = pd.concat([pd.DataFrame(r["robustness"]["table"]).assign(seed=r["seed"]) for r in runs])
    rob_all.to_csv(RESULTS / "demo_robustness.csv", index=False)
    with ExperimentStore(RESULTS / "experiments.db") as store:
        run_id = store.log_run("demo", prof.dataset, "simulated", prof.model_name,
                               prof.to_dict(), summary["aggregate"])
        for r in runs:
            for lb in r["search"]["leaderboard"]:
                store.log_prompt_version(run_id, {**lb, "n_demos": None,
                                                  "f1": lb.get("f1"), "std": lb.get("std"),
                                                  "J": lb.get("J"), "config": {}})
        store.export_csv(RESULTS / "leaderboard_export.csv")

    # ----- figures (every title carries provenance)
    counts = engine.data["label"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(counts.index, counts.values, color="#B22222")
    ax.set_title(tag_fig("Class distribution after cleaning", "MEASURED"))
    ax.set_ylabel("tickets")
    fig.tight_layout(); fig.savefig(FIGS / "fig_class_distribution.png", dpi=200); plt.close(fig)

    lengths = engine.data["text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(lengths, bins=range(int(lengths.min()), int(lengths.max()) + 2), color="#8B1A1A")
    ax.set_title(tag_fig("Text-length distribution (words)", "MEASURED"))
    fig.tight_layout(); fig.savefig(FIGS / "fig_text_length.png", dpi=200); plt.close(fig)

    names = list(agg)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, [agg[n]["f1_macro_mean"] for n in names],
           yerr=[agg[n]["f1_macro_std"] for n in names], capsize=4,
           color=["#4D4D4D" if provenance[n].startswith("MEASURED") else "#E8A0A0" for n in names])
    ax.set_ylim(0, 1.05); ax.set_ylabel("macro-F1 (mean +/- std over seeds)")
    ax.set_title(tag_fig("Method comparison on held-out test split (grey = measured, pink = simulated)", "MIXED"))
    plt.xticks(rotation=20, ha="right"); fig.tight_layout()
    fig.savefig(FIGS / "fig_method_comparison.png", dpi=200); plt.close(fig)

    labs = [str(c) for c in classes]
    cm = np.zeros((len(classes), len(classes)))
    from core.evaluation import confusion
    cm = confusion(runs[0]["_labels"], runs[0]["_probs"].argmax(axis=1), len(classes))
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Reds")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks(range(len(classes)), labs, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), labs)
    ax.set_title(tag_fig("Confusion matrix, seed 0", "SIMULATED"))
    fig.tight_layout(); fig.savefig(FIGS / "fig_confusion_sim.png", dpi=200); plt.close(fig)

    rob_mean = rob_all.groupby(["perturbation", "level"])["macro_f1"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, grp in rob_mean.groupby("perturbation"):
        ax.plot(grp["level"], grp["macro_f1"], marker="o", label=p)
    ax.set_xlabel("perturbation level"); ax.set_ylabel("macro-F1")
    ax.set_title(tag_fig("Robustness under controlled perturbations", "SIMULATED"))
    ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "fig_robustness_sim.png", dpi=200); plt.close(fig)

    ids = [r["seed"] for r in runs]
    fig, ax = plt.subplots(figsize=(7, 4))
    hist = runs[0]["search"]["halving_history"]
    stages = [h["stage"] for h in hist if h["stage"] != "final"]
    ax.plot(stages, [h["best"] for h in hist if h["stage"] != "final"], marker="o", label="best")
    ax.plot(stages, [h["median"] for h in hist if h["stage"] != "final"], marker="s", label="median")
    ax.set_xlabel("halving stage"); ax.set_ylabel("macro-F1 on validation subset")
    ax.set_title(tag_fig("Successive-halving convergence, seed 0", "SIMULATED"))
    ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "fig_halving_convergence_sim.png", dpi=200); plt.close(fig)

    print(json.dumps({"aggregate": agg, "fragility_index_mean": summary["fragility_index_mean"]},
                     indent=2, default=float))
    print(f"wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
