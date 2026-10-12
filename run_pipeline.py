"""Run every step in order. Takes about 3 minutes on a laptop."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
STEPS = [
    "step1_load_data.py",
    "step2_make_labels.py",
    "step3_build_features.py",
    "step4_train_baseline.py",
    "step5_compare_models.py",
    "step6_choose_thresholds.py",
]

for step in STEPS:
    print(f"\n##### {step} #####", flush=True)
    subprocess.run([sys.executable, step], cwd=SRC, check=True)
