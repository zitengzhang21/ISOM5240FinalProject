# %%
import subprocess
import sys

# %%
subprocess.run(
    [
        sys.executable,
        "scripts/train_distilbert.py",
        "--epochs",
        "1",
        "--batch_size",
        "8",
    ],
    check=False,
)

# %%
print("Training command completed. Review console logs and the saved model under models/phishing-distilbert-local/.")
