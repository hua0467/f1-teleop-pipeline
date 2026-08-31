"""Train a Behavior Cloning (BC) policy on a LeRobot dataset.

Thin, single-command wrapper around LeRobot's ``lerobot-train`` CLI:

    python scripts/train_bc.py --config configs/train_bc.yaml

Reads hyperparameters from the YAML config (default
``configs/train_bc.yaml``) and forwards them to the training CLI.
Any ``--key=value`` argument overrides the corresponding config entry,
so experiments stay reproducible from one command line:

    python scripts/train_bc.py --training.offline_steps=50000 --seed=7

The dataset is validated (meta/info.json must exist) before training
starts. Checkpoints are written to ``outputs/train`` by default.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "train_bc.yaml"


def load_config(path: Path) -> dict:
    """Load the YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return cfg


def flatten(cfg: dict, prefix: str = "") -> dict:
    """Flatten a nested config into dotted CLI keys (train-bc args)."""
    flat = {}
    for key, value in cfg.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten(value, full_key))
        else:
            flat[full_key] = value
    return flat


def build_train_command(flat: dict) -> list[str]:
    """Build the ``lerobot-train`` argument list from flattened config."""
    args = []
    for key, value in flat.items():
        if value is None:
            continue
        # Booleans become the strings LeRobot's CLI expects.
        if isinstance(value, bool):
            value = str(value).lower()
        args.append(f"--{key}={value}")
    return args


def validate_dataset(repo_id: str) -> None:
    """Warn early if the dataset does not look like a LeRobot dataset."""
    local = Path(repo_id)
    if not local.exists():
        print(f"[DS] NOTE: {repo_id} is not a local path; assuming a "
              f"Hugging Face repo id.")
        return
    info = local / "meta" / "info.json"
    if info.exists():
        print(f"[DS] Dataset OK: {repo_id}")
    else:
        sys.exit(
            f"[DS] ERROR: {repo_id} does not contain meta/info.json.\n"
            f"     Convert recordings first:\n"
            f"     python scripts/03_convert_to_lerobot.py "
            f"--input ./recordings --output {repo_id}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a BC policy on a LeRobot dataset (lerobot-train "
                    "wrapper).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"YAML config (default: {DEFAULT_CONFIG})")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config)
    flat = flatten(cfg)

    # Apply --key=value overrides from the command line.
    for override in overrides:
        if not override.startswith("--"):
            continue
        item = override.lstrip("--")
        if "=" not in item:
            sys.exit(f"[CFG] Override must be --key=value, got: {override}")
        key, value = item.split("=", 1)
        flat[key] = value

    repo_id = str(flat.get("dataset.repo_id", ""))
    if not repo_id:
        sys.exit("[CFG] dataset.repo_id is not set in the config.")
    validate_dataset(repo_id)

    command = build_train_command(flat)

    # Prefer the console script; fall back to the module entry point.
    if shutil.which("lerobot-train"):
        full_cmd = ["lerobot-train"] + command
    else:
        full_cmd = [sys.executable, "-m", "lerobot.scripts.train"] + command

    print("[TRAIN] Launching:")
    print("  " + " ".join(full_cmd))
    print()
    raise SystemExit(subprocess.call(full_cmd))


if __name__ == "__main__":
    main()
