"""PromptOpt-SLM - Intelligent Prompt Optimization and Text Classification Studio.

Run locally:   streamlit run app.py
Every page shows which backend produced its numbers. DEMO-profile numbers
come from a simulated backend and are labelled as such.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import load_profile                                   # noqa: E402
from core.dataset_engine import DatasetEngine, DatasetError                # noqa: E402
from core.model_engine import SimulatedBackend, SLMClassifier, build_lexicon  # noqa: E402
from core.prompt_generator import PromptConfig                             # noqa: E402
from core.robustness import RobustnessLab, UncertaintyGate                  # noqa: E402
from dashboard.components import PROVENANCE_BANNER, probability_frame, leaderboard_frame  # noqa: E402

st.set_page_config(page_title="PromptOpt-SLM", layout="wide")
RED = "#B22222"
st.markdown(f"<style>h1,h2,h3{{color:{RED};}}</style>", unsafe_allow_html=True)

PAGES = ["Project overview", "Dataset explorer", "Classification playground", "Prompt editor",
         "Optimisation", "Robustness lab", "Results", "System performance"]


@st.cache_data
def bundled_engine() -> DatasetEngine:
    return DatasetEngine.from_csv(ROOT / "data" / "demo_support_tickets.csv").clean()


@st.cache_resource
def demo_backend(classes: tuple) -> SimulatedBackend:
    eng = bundled_engine()
    name_to_id = {c: i for i, c in enumerate(classes)}
    return SimulatedBackend(build_lexicon(eng.data["text"].tolist(),
                                          [name_to_id[l] for l in eng.data["label"]], len(classes)))


def default_config(classes: list) -> PromptConfig:
    return PromptConfig(instruction=f"Classify the category of the support ticket. Answer with one of: {', '.join(classes)}.",
                        verbalizer={i: c for i, c in enumerate(classes)}, demos=[],
                        calibration="contextual")


def main() -> None:
    profile = load_profile("demo")
    st.sidebar.title("PromptOpt-SLM")
    query_page = st.query_params.get("page", None) if hasattr(st, "query_params") else None
    default_idx = PAGES.index(query_page) if query_page in PAGES else 0
    page = st.sidebar.radio("Navigate", PAGES, index=default_idx)
    st.sidebar.info(PROVENANCE_BANNER["simulated"])
    classes = tuple(bundled_engine().classes)

    if page == "Project overview":
        st.title("PromptOpt-SLM")
        st.write("Prompt optimisation, calibration and robustness testing for small language models.")
        st.caption("Modules: dataset engine, prompt optimiser, multi-SLM classifier, robustness lab, "
                   "dashboard, performance and storage.")
        st.write(f"Active profile: **{profile.name}** (backend: `{profile.backend}`, model: `{profile.model_name}`).")

    elif page == "Dataset explorer":
        st.title("Dataset explorer")
        up = st.file_uploader("Upload CSV or JSON (columns: text, label)", type=["csv", "json"])
        text_col = st.text_input("Text column", "text")
        label_col = st.text_input("Label column", "label")
        try:
            if up is None:
                eng = DatasetEngine.from_csv(ROOT / "data" / "demo_support_tickets.csv", text_col, label_col)
                st.caption("Showing the bundled synthetic demo corpus.")
            elif up.name.endswith(".json"):
                eng = DatasetEngine.from_json(io.BytesIO(up.getvalue()), text_col, label_col)
            else:
                eng = DatasetEngine(pd.read_csv(io.BytesIO(up.getvalue())), text_col, label_col)
            q = eng.quality_report()
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", q.n_rows_raw)
            c2.metric("Quality score", f"{q.quality_score:.1f}/100")
            c3.metric("Imbalance (min/max)", f"{q.imbalance_ratio:.2f}")
            st.bar_chart(pd.Series(q.class_counts, name="tickets"))
            for note in q.notes:
                st.warning(note)
            st.dataframe(eng.clean().data.head(50), use_container_width=True)
        except DatasetError as exc:
            st.error(str(exc))

    elif page == "Classification playground":
        st.title("Classification playground")
        st.warning(PROVENANCE_BANNER["simulated"])
        text = st.text_area("Ticket text", "I was charged twice for my subscription this month.")
        clf = SLMClassifier(demo_backend(classes), default_config(list(classes)))
        probs = clf.predict_proba([text])[0]
        gate = UncertaintyGate(tau=0.6, delta=0.1)
        accept = bool(gate.decide(probs.reshape(1, -1))[0])
        st.bar_chart(probability_frame(classes, probs).set_index("class"))
        st.write(f"Predicted: **{classes[int(np.argmax(probs))]}**  |  confidence {probs.max():.2f}")
        if not accept:
            st.warning("Low confidence: routed to human review.")

    elif page == "Prompt editor":
        st.title("Prompt editor")
        instr = st.text_area("Instruction", default_config(list(classes)).instruction)
        cal = st.selectbox("Calibration", ["none", "contextual", "prior"], index=1)
        cfg = PromptConfig(instruction=instr, verbalizer={i: c for i, c in enumerate(classes)},
                           calibration=cal)
        st.code(cfg.instruction + "\n\nText: <ticket>\nLabel:", language="text")
        st.download_button("Download prompt configuration (JSON)",
                           json.dumps(cfg.to_dict(), indent=2), "prompt_config.json")

    elif page == "Optimisation":
        st.title("Prompt optimisation")
        st.warning("Runs the DEMO profile (simulated backend) on the bundled corpus. "
                   "Results are plumbing checks. Run `python -m experiments.run_demo` for the full sweep.")
        st.write("Successive halving over a random candidate pool. See results/demo_summary.json for stored runs.")

    elif page == "Robustness lab":
        st.title("Robustness lab")
        st.warning(PROVENANCE_BANNER["simulated"])
        eng = bundled_engine()
        name_to_id = {c: i for i, c in enumerate(classes)}
        sample = eng.data.sample(n=min(80, len(eng.data)), random_state=0)
        lab = RobustnessLab(SLMClassifier(demo_backend(classes), default_config(list(classes))),
                            sample["text"].tolist(), [name_to_id[l] for l in sample["label"]])
        table = lab.run()
        pfi, per_type = RobustnessLab.fragility_index(table)
        st.metric("Prompt Fragility Index (0 = robust)", f"{pfi:.3f}")
        st.dataframe(table, use_container_width=True)

    elif page == "Results":
        st.title("Experimental results")
        summary_path = ROOT / "results" / "demo_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            st.subheader("Method comparison (provenance in column 'provenance')")
            comp = pd.read_csv(ROOT / "results" / "demo_method_comparison.csv")
            st.dataframe(comp, use_container_width=True)
            st.download_button("Download summary (JSON)", json.dumps(summary, indent=2), "demo_summary.json")
            st.download_button("Download method comparison (CSV)", comp.to_csv(index=False), "demo_method_comparison.csv")
            figs_dir = ROOT / "results" / "figures"
            if figs_dir.exists():
                st.subheader("Visualizations")
                fc1, fc2 = st.columns(2)
                f1_img = figs_dir / "fig_method_comparison.png"
                cm_img = figs_dir / "fig_confusion_sim.png"
                rob_img = figs_dir / "fig_robustness_sim.png"
                conv_img = figs_dir / "fig_halving_convergence_sim.png"
                if f1_img.exists():
                    fc1.image(str(f1_img), caption="Method Comparison (Macro-F1)")
                if cm_img.exists():
                    fc2.image(str(cm_img), caption="Confusion Matrix (Simulated)")
                if rob_img.exists():
                    fc1.image(str(rob_img), caption="Robustness under Perturbations")
                if conv_img.exists():
                    fc2.image(str(conv_img), caption="Successive Halving Convergence")
        else:
            st.info("No stored results yet. Run `python -m experiments.run_demo`.")

    elif page == "System performance":
        st.title("System performance")
        from core.resource_monitor import LatencyTracker, memory_snapshot
        tracker = LatencyTracker()
        clf = SLMClassifier(demo_backend(classes), default_config(list(classes)))
        sample = bundled_engine().data["text"].tolist()[:50]
        for t in sample:
            with tracker.measure():
                clf.predict_proba([t])
        stats = tracker.stats().to_dict()
        st.json(stats)
        st.json(memory_snapshot())
        st.caption("Measured on this machine for the simulated backend; SLM latency will differ.")


main()
