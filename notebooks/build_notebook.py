"""Builds PromptOpt-SLM_Research_Notebook.ipynb (Google Colab compatible).

Run once locally:  python notebooks/build_notebook.py
The notebook clones nothing: upload the project zip in the first cell,
or mount Drive and point PROJECT_DIR at the unzipped folder.
"""
import json
from pathlib import Path

CELLS = [
    ("markdown", ["# PromptOpt-SLM - Research Workflow (Google Colab)\n",
                  "Runs the DEMO profile end to end. The DEMO profile uses a **simulated** backend "
                  "(not a language model). To run real small language models, set `PROFILE = 'research'` "
                  "on a GPU runtime after installing `requirements.txt`."]),
    ("code", ["# 1. Unzip the uploaded project (upload PromptOpt-SLM_Complete_Project.zip first)\n",
              "import zipfile, os, pathlib\n",
              "if os.path.exists('/content/PromptOpt-SLM.zip'):\n",
              "    zipfile.ZipFile('/content/PromptOpt-SLM.zip').extractall('/content')\n",
              "PROJECT_DIR = '/content/PromptOpt-SLM'\n",
              "os.chdir(PROJECT_DIR)\n",
              "print(os.listdir(PROJECT_DIR))"]),
    ("code", ["# 2. Dependencies (numpy, pandas, scikit-learn and matplotlib are preinstalled on Colab)\n",
              "!pip -q install streamlit==1.* 2>/dev/null || true"]),
    ("code", ["# 3. Automated tests\n",
              "!python tests/run_tests.py"]),
    ("code", ["# 4. Regenerate the bundled corpus (deterministic, seed 42)\n",
              "!python data/make_demo_data.py"]),
    ("code", ["# 5. Run the DEMO experiment (writes results/ and results/figures/)\n",
              "!python -m experiments.run_demo"]),
    ("code", ["# 6. Inspect measured and simulated outputs, with provenance tags\n",
              "import json, pandas as pd\n",
              "summary = json.load(open('results/demo_summary.json'))\n",
              "print('Backend:', summary['backend'])\n",
              "display(pd.read_csv('results/demo_method_comparison.csv'))"]),
    ("code", ["# 7. Show figures\n",
              "from IPython.display import Image, display\n",
              "for f in ['results/figures/fig_class_distribution.png','results/figures/fig_method_comparison.png']:\n",
              "    display(Image(f))"]),
    ("code", ["# 8. Optional research profile: real SLM (needs GPU and internet). Off by default.\n",
              "PROFILE = 'demo'  # change to 'research' deliberately\n",
              "from config.settings import load_profile\n",
              "print(load_profile(PROFILE))"]),
]


def build(out: Path) -> None:
    nb = {"cells": [], "metadata": {"colab": {"name": "PromptOpt-SLM_Research_Notebook.ipynb"},
                                    "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                    "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "PromptOpt-SLM_Research_Notebook.ipynb"
    build(target)
    print(f"wrote {target}")
