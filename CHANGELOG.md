# Changelog

All notable changes to the **PromptOpt-SLM** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-11-24

### Added
- **Core Architecture & Modules**:
  - `DatasetEngine` (Module 1): Automated CSV/JSON loading, normalisation, duplicate removal, quality scoring ($Q \in [0, 100]$), class imbalance ratio, and stratified train/val/test splitting.
  - `PromptGenerator` (Module 2, Part 1): Grammar-based prompt generation, synonym verbalizer expansion with tokenizer collision filtering, and stable SHA-1 configuration signatures.
  - `DemoSelector` (Module 2, Part 2): Class-balanced random and Class-Balanced Maximal Marginal Relevance (MMR) exemplar selection with pluggable TF-IDF and Sentence-Transformer embedders.
  - `PromptOptimizer` (Module 2, Part 3): Budgeted multi-stage Successive Halving search over candidate prompt configurations with stability-aware objective $J(\pi) = \text{MacroF1} - \lambda \cdot \sigma - \mu \cdot \text{tokens} - \nu \cdot \text{latency}$.
  - `AdaptiveEvolver` (Module 2 Extension): Evolutionary prompt optimization loop with rule-based mutation and demonstration recombination.
  - `ScoringBackend` & Polymorphism (Module 3): Dual-backend architecture supporting `SimulatedBackend` (deterministic lexicon heuristic with prior and recency bias) and `HFBackend` (teacher-forcing causal LM scoring via PyTorch & Hugging Face Transformers).
  - `Calibration`: Output-side contextual calibration (Zhao et al., 2021) and uniform prior matching.
  - `RobustnessLab` & `UncertaintyGate` (Module 4): Controlled perturbations (typos, word dropout, noise tokens, synonym replacement), Prompt Fragility Index (PFI), and dual-threshold abstention gate $(\tau, \delta)$.
  - `ExperimentStore` (Module 6): Thread-safe SQLite repository logging experimental runs, prompt versions, leaderboards, and CSV exports.
  - `ResourceMonitor`: Wall-clock latency tracker (P50, P95, throughput) and tracemalloc / GPU memory snapshotting.
- **Interactive Research Studio (`app.py`)**:
  - Full multi-page Streamlit application covering Project Overview, Dataset Explorer, Classification Playground, Prompt Editor, Optimization, Robustness Lab, Experimental Results, and System Performance.
  - URL query-parameter navigation (`?page=...`) for programmatic view selection and automated test reporting.
  - Native display of generated research figures, confusion matrices, and latency telemetry.
- **Testing & Tooling**:
  - Comprehensive unit test suite in `tests/test_core.py` (37 tests covering all modules).
  - Standalone zero-dependency test runner `tests/run_tests.py`.
  - Synthetic corpus generator `data/make_demo_data.py` (200 support tickets, 3 classes).
  - Google Colab notebook builder `notebooks/build_notebook.py`.
  - GitHub Actions CI workflow running across Ubuntu and Windows on Python 3.10, 3.11, and 3.12.
- **Research Documentation**:
  - Complete academic report `report/report.md` accompanied by compiled DOCX and PDF versions in `results/`.
  - System architecture Graphviz diagram (`assets/architecture.dot` and `assets/fig_architecture.png`).
  - Genuine high-resolution application screenshots in `assets/`.
