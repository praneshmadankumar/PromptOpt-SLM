# Contributing to PromptOpt-SLM

Thank you for your interest in contributing to **PromptOpt-SLM: Prompt Optimization Framework for Few-Shot Text Classification Using Small Language Models**.

This repository serves as both a reproducible academic research artifact and an extensible Python toolkit for prompt optimization across edge and small language models.

---

## Code of Conduct

We are committed to providing a welcoming, constructive, and harassment-free environment for all contributors. Please ensure that all interactions remain civil, evidence-based, and respectful.

---

## Research Integrity & Provenance Guidelines

This repository strictly separates **empirical measurements** from **heuristic simulations**:

1. **Do not fabricate results**: Any benchmark claims must be produced by executable code and logged in reproducible artifacts (`results/`).
2. **Explicit Provenance**:
   - `MEASURED`: Real machine learning model fit on real data (e.g. Scikit-learn TF-IDF + Logistic Regression).
   - `SIMULATED`: Heuristic mock scorer (`SimulatedBackend`) used for offline verification and plumbing checks.
   - `RESEARCH`: Actual Small Language Model run (e.g., Qwen2.5-0.5B-Instruct via `HFBackend`) executed on hardware.
3. Every figure, table, and log entry must display its provenance category clearly.

---

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/praneshmadankumar/PromptOpt-SLM.git
   cd PromptOpt-SLM
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Using standard venv
   python -m venv .venv
   source .venv/bin/activate  # On Linux/macOS
   .venv\Scripts\activate     # On Windows (PowerShell)

   # Alternatively using uv (recommended for ultra-fast setup)
   uv venv --python 3.12 .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # For core demo profile (offline, CPU friendly)
   pip install -r requirements.txt

   # For GPU research profile (real Hugging Face SLMs)
   pip install torch transformers sentence-transformers
   ```

---

## Testing & Quality Assurance

Before submitting any pull request, ensure that all tests pass:

```bash
# Run test suite with pytest
pytest tests/ -v

# Run the standalone zero-dependency test runner
python tests/run_tests.py

# Verify the full demo experiment pipeline
python -m experiments.run_demo
```

All 37 unit tests must pass with zero warnings or regressions.

---

## Git Workflow & Commit Guidelines

We follow **Conventional Commits**:

- `feat:` A new feature or capability (e.g., new demonstration selection heuristic)
- `fix:` A bug fix in existing modules
- `docs:` Documentation additions or corrections
- `test:` Adding or refining unit tests
- `refactor:` Code restructuring without functional alterations
- `perf:` Performance or memory optimizations

### Pull Request Process

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Commit your changes with clear, descriptive commit messages.
3. Verify tests and linting.
4. Push your branch to GitHub and open a Pull Request against `main`.
5. Provide a summary of changes, rationale, and sample outputs in the PR description.
