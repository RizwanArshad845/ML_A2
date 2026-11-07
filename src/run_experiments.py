"""Task 3.5: run the experiment grid. Each run differs from the reference in ONE change.

    python src/run_experiments.py            # all runs
    python src/run_experiments.py vg_img160  # only the named ones

All runs share the same budget (20 frozen + 10 fine-tune epochs, early stopping) so
the comparison is fair. Output of each run goes to logs/<run_name>.log.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON = ["--epochs", "20", "--epochs_ft", "10"]

RUNS = [
    # (run name, tracking mode, extra args, the one change)
    ("vg_ref", "manual", ["--model", "transfer"], "reference"),
    ("vg_lr3e-4", "manual", ["--model", "transfer", "--lr", "3e-4"], "lr 1e-3 -> 3e-4"),
    ("vg_aug_none", "manual", ["--model", "transfer", "--augment", "none"], "augment basic -> none"),
    ("vg_aug_strong", "manual", ["--model", "transfer", "--augment", "strong"], "augment basic -> strong"),
    ("vg_img160", "manual", ["--model", "transfer", "--img_size", "160"], "img_size 224 -> 160"),
    ("vg_frozen", "manual", ["--model", "transfer", "--epochs_ft", "0"], "frozen base, no fine-tuning"),
    ("vg_baseline_cnn", "manual", ["--model", "baseline"], "backbone MobileNetV2 -> baseline CNN"),
    ("vg_ref_autolog", "autolog", ["--model", "transfer"], "reference, logged with autolog"),
]


def main() -> None:
    wanted = set(sys.argv[1:])
    (ROOT / "logs").mkdir(exist_ok=True)
    for name, mode, extra, change in RUNS:
        if wanted and name not in wanted:
            continue
        cmd = [sys.executable, str(ROOT / "src" / "train.py"), *COMMON, *extra,
               "--run_name", name, "--tracking", mode, "--change", change]
        with open(ROOT / "logs" / f"{name}.log", "w", encoding="utf-8") as log:
            code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        lines = (ROOT / "logs" / f"{name}.log").read_text(encoding="utf-8").splitlines()
        summary = [line for line in lines if line.startswith("MLflow run")]
        print(f"{name}: exit {code} {summary[-1] if summary else ''}", flush=True)


if __name__ == "__main__":
    main()
