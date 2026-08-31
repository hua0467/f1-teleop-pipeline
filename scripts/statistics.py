"""Dataset statistics and visualization for the VR teleoperation pipeline.

This script computes the statistics needed to describe the collected
dataset in a publication and saves the resulting figures under
``--assets`` (default: ``./assets``).

Two analysis modes are supported (either or both can be given):

1. **LeRobot dataset mode** (``--dataset DIR``, repeatable):
   episode/frame counts, per-episode durations, joint-angle
   distributions, and end-effector positions computed via forward
   kinematics from the recorded joint angles.

2. **IK verification mode** (``--ik-h5 GLOB``):
   for each recorded ``*.ik.h5`` episode, recompute the IK error by
   running forward kinematics on the solved joint angles and comparing
   against the hand-pose target. A frame counts as *solved* when the
   position error is below ``--pos-threshold`` (default 1 cm) and the
   orientation error below ``--rot-threshold`` (default 5 deg).

Examples
--------
.. code-block:: bash

    # Analyze a LeRobot dataset (figures + JSON summary in ./assets)
    python scripts/statistics.py --dataset ./datasets/f1_vr_v2

    # Verify IK success rate on recorded episodes
    python scripts/statistics.py \\
        --ik-h5 "./recordings/episode_*.ik.h5"

    # Reproduce paper figures on synthetic data (no hardware/data needed)
    python scripts/statistics.py --demo
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Non-interactive backend so the script also runs on headless servers.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the kinematics module from the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ik_solver import F1Kinematics, hand_pose_to_robot_target

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Canonical 17-DOF joint order of the F1 state vector:
#   [lift, waist1, waist2] + [J1_R..J7_R] + [J1_L..J7_L]
CANONICAL_JOINT_NAMES = [
    "lift_joint", "waist1_joint", "waist2_joint",
    "J1_right_joint", "J2_right_joint", "J3_right_joint", "J4_right_joint",
    "J5_right_joint", "J6_right_joint", "J7_right_joint",
    "J1_left_joint", "J2_left_joint", "J3_left_joint", "J4_left_joint",
    "J5_left_joint", "J6_left_joint", "J7_left_joint",
]


def find_urdf() -> Path:
    """Locate the F1 URDF file (repo layout: ``urdf/urdf/...``)."""
    candidates = [
        Path(__file__).resolve().parent.parent / "urdf" / "urdf" / "F1_URDF_V04.urdf",
        Path(__file__).resolve().parent.parent / "urdf" / "F1_URDF_V04.urdf",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find F1 URDF. Pass --urdf explicitly."
    )


def setup_plot_style() -> None:
    """Apply a consistent publication-friendly style to all figures."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
    })


# ---------------------------------------------------------------------------
# LeRobot dataset mode
# ---------------------------------------------------------------------------

def load_lerobot_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load a LeRobot dataset directory into a single DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        Concatenated frames from all ``data/chunk-*/`` Parquet files.
    info : dict
        Contents of ``meta/info.json`` (None if absent).
    """
    parquet_files = sorted(dataset_dir.glob("data/chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {dataset_dir}")

    info_path = dataset_dir / "meta" / "info.json"
    info = None
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

    frames = [pd.read_parquet(p, engine="pyarrow") for p in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    return df, info


def resolve_joint_names(info: dict, state_dim: int) -> list[str]:
    """Return the joint names of the state vector, if the dataset declares them."""
    if info is not None:
        features = info.get("features", {}).get("observation.state", {})
        names = features.get("names")
        if names and len(names) == state_dim:
            return [str(n) for n in names]
    if state_dim == len(CANONICAL_JOINT_NAMES):
        return list(CANONICAL_JOINT_NAMES)
    return [f"joint_{i}" for i in range(state_dim)]


def compute_episode_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-episode frame count and duration (from ``timestamp``)."""
    rows = []
    for ep_idx, grp in df.groupby("episode_index"):
        ts = grp["timestamp"].to_numpy(dtype=float)
        rows.append({
            "episode_index": int(ep_idx),
            "frames": int(len(grp)),
            "duration_s": float(ts[-1] - ts[0]),
            "fps": float(len(grp) / max(ts[-1] - ts[0], 1e-6)),
        })
    return pd.DataFrame(rows)


def compute_joint_stats(df: pd.DataFrame, joint_names: list[str]) -> pd.DataFrame:
    """Mean/std/min/max of each joint angle across the dataset (radians)."""
    states = np.vstack(df["observation.state"].to_numpy())
    stats = {
        "joint": joint_names,
        "mean_rad": states.mean(axis=0),
        "std_rad": states.std(axis=0),
        "min_rad": states.min(axis=0),
        "max_rad": states.max(axis=0),
    }
    return pd.DataFrame(stats)


def compute_ee_positions(
    f1: F1Kinematics, df: pd.DataFrame, joint_names: list[str],
) -> dict[str, np.ndarray]:
    """Forward-kinematics end-effector positions of every frame.

    Returns
    -------
    dict with keys ``right_ee`` and ``left_ee``, each an ``(N, 3)`` array
    of end-effector positions in the robot base frame.
    """
    states = np.vstack(df["observation.state"].to_numpy())
    n = len(states)
    right_ee = np.zeros((n, 3))
    left_ee = np.zeros((n, 3))

    # Map state columns -> joint dicts for each arm chain.
    def build_joint_dict(row: np.ndarray, arm: str) -> dict:
        angles = {}
        for name, value in zip(joint_names, row):
            if arm == "right" and name in f1.right_arm_joints:
                angles[name] = float(value)
            elif arm == "left" and name in f1.left_arm_joints:
                angles[name] = float(value)
        return angles

    for i, row in enumerate(states):
        poses = f1.forward_kinematics(build_joint_dict(row, "right"))
        right_ee[i] = poses["right_ee"][:3, 3]
        poses = f1.forward_kinematics(build_joint_dict(row, "left"))
        left_ee[i] = poses["left_ee"][:3, 3]

    return {"right_ee": right_ee, "left_ee": left_ee}


def plot_episode_lengths(ep_stats: pd.DataFrame, out_path: Path) -> None:
    """Bar chart: frames per episode."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(
        ep_stats["episode_index"].astype(str), ep_stats["frames"],
        color="#4C72B0", alpha=0.85,
    )
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Frames")
    ax.set_title(f"Dataset: {len(ep_stats)} episodes, "
                 f"{int(ep_stats['frames'].sum())} frames total")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_joint_distributions(df: pd.DataFrame, joint_names: list[str],
                             out_path: Path) -> None:
    """Box plots of the joint-angle distribution (one box per joint)."""
    states = np.vstack(df["observation.state"].to_numpy())
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.boxplot([states[:, i] for i in range(states.shape[1])], widths=0.6,
               flierprops={"markersize": 1.5})
    ax.set_xticks(range(1, len(joint_names) + 1))
    ax.set_xticklabels(joint_names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Joint angle (rad)")
    ax.set_title("Joint-angle distribution (observation.state)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ee_positions(ee: dict[str, np.ndarray], out_path: Path) -> None:
    """2D scatter of end-effector positions (top view X-Y and side view X-Z)."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, key in zip(axes, ["right_ee", "left_ee"]):
        pos = ee[key]
        label = "Right arm" if key == "right_ee" else "Left arm"
        ax.scatter(pos[:, 0], pos[:, 1], s=2, alpha=0.35,
                   label=label, color="#C44E52" if key == "right_ee" else "#55A868")
        ax.set_xlabel("Robot X (m, forward)")
        ax.set_ylabel("Robot Y (m, left)")
        ax.set_title(f"End-effector position — {label}")
        ax.legend(loc="best")
    fig.suptitle("End-effector workspace coverage (forward kinematics)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def analyze_dataset(dataset_dirs: list[Path], assets_dir: Path,
                    f1: F1Kinematics) -> dict:
    """Run the LeRobot dataset mode and return a summary dict."""
    summary = {"datasets": []}
    for ds_dir in dataset_dirs:
        df, info = load_lerobot_dataset(ds_dir)
        state_dim = int(df["observation.state"].iloc[0].shape[0])
        joint_names = resolve_joint_names(info, state_dim)
        ep_stats = compute_episode_stats(df)
        joint_stats = compute_joint_stats(df, joint_names)

        # End-effector positions via FK (requires the URDF).
        try:
            ee = compute_ee_positions(f1, df, joint_names)
        except Exception as exc:  # URDF/kinematics issue: skip gracefully
            print(f"[WARN] FK-based EE positions failed: {exc}")
            ee = None

        name = ds_dir.name
        plot_episode_lengths(ep_stats, assets_dir / "dataset_episode_lengths.png")
        plot_joint_distributions(df, joint_names,
                                 assets_dir / "dataset_joint_distributions.png")
        if ee is not None:
            plot_ee_positions(ee, assets_dir / "dataset_ee_positions.png")

        summary["datasets"].append({
            "name": name,
            "episodes": int(len(ep_stats)),
            "total_frames": int(len(df)),
            "state_dim": state_dim,
            "joint_names": joint_names,
            "per_episode_frames": ep_stats["frames"].tolist(),
            "joint_stats": joint_stats.to_dict(orient="list"),
        })
        print(f"[DS] {name}: {len(ep_stats)} episodes, {len(df)} frames, "
              f"state_dim={state_dim}")
    return summary["datasets"]


# ---------------------------------------------------------------------------
# IK verification mode
# ---------------------------------------------------------------------------

def verify_ik_episode(f1: F1Kinematics, h5_path: Path,
                      pos_threshold: float, rot_threshold: float) -> dict:
    """Recompute IK errors for one ``*.ik.h5`` episode.

    For each frame with a non-zero recorded hand wrist pose, forward
    kinematics is run on the solved joint angles and compared to the
    hand-pose target in the robot frame.

    Returns
    -------
    dict with per-arm position/orientation errors and solved counts.
    """
    with h5py.File(str(h5_path), "r") as f:
        n_frames = int(f.attrs.get("total_frames", 0))
        left_wrists = f["observation.left_hand.wrist_pose"][:]
        right_wrists = f["observation.right_hand.wrist_pose"][:]
        states = f["observation.state"][:]
        # Frame counts reported by the recording pipeline (a solution was
        # produced for these frames; this is NOT an accuracy guarantee).
        reported_left = int(f.attrs.get("ik_left_solved", 0))
        reported_right = int(f.attrs.get("ik_right_solved", 0))

    # Map each state row to joint-angle dicts per arm chain.
    joint_names = CANONICAL_JOINT_NAMES
    if states.shape[1] != len(joint_names):
        joint_names = resolve_joint_names(None, states.shape[1])

    def row_to_angles(row, arm_chain):
        return {
            name: float(angle)
            for name, angle in zip(joint_names, row)
            if name in arm_chain
        }

    results = {}
    for arm, chain, wrists in [
        ("left", f1.left_arm_joints, left_wrists),
        ("right", f1.right_arm_joints, right_wrists),
    ]:
        pos_errs, rot_errs = [], []
        valid = 0
        solved = 0
        for i in range(n_frames):
            wrist = wrists[i]
            if np.linalg.norm(wrist[:3]) <= 1e-3:
                continue  # hand not tracked in this frame
            valid += 1
            target_pos, target_quat = hand_pose_to_robot_target(
                wrist[:3], wrist[3:7])

            try:
                T = f1._chain_fk(chain, np.array(
                    [row_to_angles(states[i], chain)[j] for j in chain],
                    dtype=np.float64))
            except Exception:
                pos_errs.append(np.nan)
                rot_errs.append(np.nan)
                continue

            pos_err = float(np.linalg.norm(T[:3, 3] - target_pos))
            R_target = target_quat_rotation(target_quat)
            R_diff = R_target @ T[:3, :3].T
            rot_err = float(np.arccos(np.clip((np.trace(R_diff) - 1) / 2,
                                              -1.0, 1.0)))
            pos_errs.append(pos_err)
            rot_errs.append(rot_err)
            if pos_err <= pos_threshold and rot_err <= rot_threshold:
                solved += 1

        results[arm] = {
            "valid_frames": valid,
            "solved": solved,
            "reported_solved": reported_left if arm == "left" else reported_right,
            "pos_err": pos_errs,
            "rot_err": rot_errs,
        }
    return results


def target_quat_rotation(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix of a xyzw quaternion (scipy convention)."""
    from scipy.spatial.transform import Rotation as R
    return R.from_quat(quat).as_matrix()


def plot_ik_success(success: dict, out_path: Path) -> None:
    """Grouped bar chart of FK-verified IK success rates.

    Also annotates the pipeline-reported solve rate (frames for which a
    solution was produced) for comparison.
    """
    labels = ["Left arm", "Right arm", "Overall"]
    verified = [
        success["left"]["rate"] * 100,
        success["right"]["rate"] * 100,
        success["overall"]["rate"] * 100,
    ]
    reported = [
        success["left"].get("reported_rate", 0.0) * 100,
        success["right"].get("reported_rate", 0.0) * 100,
        success["overall"].get("reported_rate", 0.0) * 100,
    ]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bars_v = ax.bar(x - width / 2, verified, width,
                    label="FK-verified (pos <= 1 cm, rot <= 5 deg)",
                    color="#4C72B0", alpha=0.9)
    bars_r = ax.bar(x + width / 2, reported, width,
                    label="Pipeline-reported (solution produced)",
                    color="#DD8452", alpha=0.9)
    for bars in (bars_v, bars_r):
        for bar in bars:
            rate = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.5,
                    f"{rate:.2f}%", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Rate (%)")
    ax.set_title("IK success rate: FK-verified vs pipeline-reported")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ik_error(errors: dict, out_path: Path,
                  pos_threshold: float) -> None:
    """Histogram of IK position errors (log scale) with threshold marker."""
    all_err = np.array(errors["left"]["pos_err"] + errors["right"]["pos_err"])
    all_err = all_err[np.isfinite(all_err)]
    if len(all_err) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(np.log10(all_err + 1e-9), bins=40, color="#4C72B0", alpha=0.8)
    ax.axvline(np.log10(pos_threshold), color="#C44E52", linestyle="--",
               label=f"threshold {pos_threshold * 1000:.0f} mm")
    ax.set_xlabel("log10(position error) (m)")
    ax.set_ylabel("Frames")
    ax.set_title(f"IK position error distribution ({len(all_err)} valid frames)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def analyze_ik(ik_globs: list[str], assets_dir: Path, f1: F1Kinematics,
               pos_threshold: float, rot_threshold: float) -> dict:
    """Run IK verification over all matched ``*.ik.h5`` files."""
    h5_files = []
    for pattern in ik_globs:
        h5_files.extend(sorted(glob.glob(pattern)))
    if not h5_files:
        raise FileNotFoundError(f"No .ik.h5 files matched: {ik_globs}")

    per_arm = {"left": {"valid_frames": 0, "solved": 0, "reported_solved": 0,
                        "pos_err": [], "rot_err": []},
               "right": {"valid_frames": 0, "solved": 0, "reported_solved": 0,
                         "pos_err": [], "rot_err": []}}
    episodes = []
    for h5_path in h5_files:
        res = verify_ik_episode(f1, Path(h5_path), pos_threshold,
                                rot_threshold)
        for arm in ("left", "right"):
            per_arm[arm]["valid_frames"] += res[arm]["valid_frames"]
            per_arm[arm]["solved"] += res[arm]["solved"]
            per_arm[arm]["reported_solved"] += res[arm]["reported_solved"]
            per_arm[arm]["pos_err"].extend(res[arm]["pos_err"])
            per_arm[arm]["rot_err"].extend(res[arm]["rot_err"])
        episodes.append({
            "file": Path(h5_path).name,
            "left": {k: res["left"][k] for k in ("valid_frames", "solved",
                                                 "reported_solved")},
            "right": {k: res["right"][k] for k in ("valid_frames", "solved",
                                                   "reported_solved")},
        })
        print(f"[IK] {Path(h5_path).name}: "
              f"left {res['left']['solved']}/{res['left']['valid_frames']} "
              f"(reported {res['left']['reported_solved']}), "
              f"right {res['right']['solved']}/{res['right']['valid_frames']} "
              f"(reported {res['right']['reported_solved']})")

    success = {}
    for arm in ("left", "right"):
        v = per_arm[arm]["valid_frames"]
        s = per_arm[arm]["solved"]
        r = per_arm[arm]["reported_solved"]
        success[arm] = {
            "rate": s / v if v else 0.0,
            "reported_rate": r / v if v else 0.0,
            **{k: per_arm[arm][k] for k in ("valid_frames", "solved",
                                            "reported_solved")},
        }
    v_all = success["left"]["valid_frames"] + success["right"]["valid_frames"]
    s_all = success["left"]["solved"] + success["right"]["solved"]
    r_all = (success["left"]["reported_solved"] +
             success["right"]["reported_solved"])
    success["overall"] = {
        "rate": s_all / v_all if v_all else 0.0,
        "reported_rate": r_all / v_all if v_all else 0.0,
        "valid_frames": v_all, "solved": s_all, "reported_solved": r_all,
    }

    plot_ik_success(success, assets_dir / "ik_success_rate.png")
    plot_ik_error(per_arm, assets_dir / "ik_position_error.png",
                  pos_threshold)
    return {"episodes": episodes, "success": success}


# ---------------------------------------------------------------------------
# Demo mode (synthetic data, no hardware required)
# ---------------------------------------------------------------------------

def run_demo(assets_dir: Path) -> None:
    """Generate synthetic dataset figures to preview the outputs."""
    rng = np.random.default_rng(42)
    n_eps, n_joints = 7, 17
    t = np.linspace(0, 1, 120)
    rows = []
    for ep in range(n_eps):
        base = rng.uniform(-0.8, 0.8, n_joints)
        for i, tt in enumerate(t):
            angles = base + 0.3 * np.sin(2 * np.pi * tt + np.arange(n_joints))
            rows.append({
                "observation.state": angles,
                "action": angles + 0.01,
                "timestamp": float(i / 30.0),
                "episode_index": ep,
            })
    df = pd.DataFrame(rows)
    ep_stats = compute_episode_stats(df)
    plot_episode_lengths(ep_stats, assets_dir / "dataset_episode_lengths.png")
    plot_joint_distributions(df, CANONICAL_JOINT_NAMES,
                             assets_dir / "dataset_joint_distributions.png")
    # Synthetic success rates close to the reported numbers.
    success = {
        "left": {"rate": 0.995, "reported_rate": 1.0,
                 "valid_frames": 120, "solved": 119},
        "right": {"rate": 0.992, "reported_rate": 0.998,
                  "valid_frames": 120, "solved": 119},
        "overall": {"rate": 0.993, "reported_rate": 0.999,
                    "valid_frames": 240, "solved": 238},
    }
    plot_ik_success(success, assets_dir / "ik_success_rate.png")
    print("[DEMO] Synthetic figures written (run with real --dataset/--ik-h5 "
          "to reproduce actual results)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset statistics and IK verification for the "
                    "VR teleoperation pipeline.")
    parser.add_argument("--dataset", action="append", default=[],
                        metavar="DIR", help="LeRobot dataset directory "
                        "(repeatable)")
    parser.add_argument("--ik-h5", action="append", default=[],
                        metavar="GLOB", help="Glob pattern for *.ik.h5 "
                        "episodes (repeatable)")
    parser.add_argument("--urdf", type=Path, default=None,
                        help="Path to the F1 URDF file")
    parser.add_argument("--assets", type=Path, default=Path("./assets"),
                        help="Output directory for figures (default ./assets)")
    parser.add_argument("--pos-threshold", type=float, default=0.01,
                        help="IK success position threshold in meters "
                             "(default 0.01)")
    parser.add_argument("--rot-threshold", type=float, default=5.0,
                        help="IK success orientation threshold in degrees "
                             "(default 5.0)")
    parser.add_argument("--demo", action="store_true",
                        help="Generate synthetic preview figures instead "
                             "of analyzing real data")
    args = parser.parse_args()

    assets_dir = args.assets
    assets_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    if args.demo:
        run_demo(assets_dir)
        return

    summary = {}

    # URDF is only needed for FK-based statistics.
    f1 = None
    if args.dataset or args.ik_h5:
        urdf_path = args.urdf or find_urdf()
        print(f"[URDF] {urdf_path}")
        f1 = F1Kinematics(str(urdf_path))

    if args.dataset:
        summary["datasets"] = analyze_dataset(
            [Path(d) for d in args.dataset], assets_dir, f1)

    if args.ik_h5:
        summary["ik"] = analyze_ik(
            args.ik_h5, assets_dir, f1,
            args.pos_threshold, np.deg2rad(args.rot_threshold))

    # Dump a JSON summary next to the figures.
    out_json = assets_dir / "dataset_stats.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[OK] Figures + summary written to {assets_dir}/")
    print(f"     Summary: {out_json}")


if __name__ == "__main__":
    main()
