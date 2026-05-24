# %%
import subprocess
import sys

import pandas as pd

from src.config import EVALUATION_METRICS_FILE, EXPERIMENTAL_RESULTS_FILE, TEST_PREDICTIONS_FILE

# %%
subprocess.run([sys.executable, "scripts/evaluate_model.py"], check=False)
subprocess.run([sys.executable, "scripts/make_experimental_results.py"], check=False)

# %%
if TEST_PREDICTIONS_FILE.exists():
    pd.read_csv(TEST_PREDICTIONS_FILE).head()

# %%
print(f"Evaluation metrics file: {EVALUATION_METRICS_FILE}")
print(f"Experimental workbook: {EXPERIMENTAL_RESULTS_FILE}")
