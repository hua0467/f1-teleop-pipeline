# Data Collection Pipeline

This document describes the full demonstration-collection pipeline:
hardware setup, the step-by-step recording workflow, the inverse
kinematics (IK) method and its success metrics, the HDF5 → LeRobot
conversion, and data integrity validation.

## 1. Hardware Setup

| Component | Role | Notes |
|-----------|------|-------|
| Meta Quest 3 headset | Hand tracking (22 landmarks + wrist pose per hand) | Runs a hand-tracking-streamer APK that sends data over UDP |
| PC (data collector) | Receives UDP, solves IK, writes HDF5/Parquet | Windows or Linux; GPU not required for collection |
| F1 mobile manipulator (17 DOF) | Target robot; its URDF defines the kinematic chains | 3 torso joints (lift, waist1, waist2) + 7-DOF right arm + 7-DOF left arm |
| GPU workstation (Ubuntu 22.04 + RTX 4090) | Training and offline processing | LeRobot, PyTorch |
| Camera (optional) | Third-person view for `*_vision` datasets | Not required for state-only datasets |

### Network setup

The Quest 3 and the PC must be on the same LAN. The streamer APK is
configured inside the headset with the **PC's IP address** and **UDP port
9000** (both configurable). Each packet is a UTF-8 text line of the form:

```
Left wrist | f = <frame> | t = <ms>: x, y, z, qx, qy, qz, qw
Left landmarks | f = <frame> | t = <ms>: x0, y0, z0, x1, y1, z1, ... (66 values)
```

Coordinate conventions: Quest 3 uses X = right, Y = up, Z = backward.
See [Section 3.2](#32-hand-robot-coordinate-mapping) for the mapping to
the robot frame.

## 2. Recording Workflow (Step by Step)

1. **Verify the environment** — `python scripts/01_test_environment.py`.
2. **Prepare the headset** — start the hand-tracking-streamer APK, enter
   the PC's IP and port 9000, press Start. Colored hand-tracking points
   should appear in the headset view.
3. **Record live with IK** (single pass produces HDF5 + LeRobot):

   ```bash
   python scripts/06_vr_ik_record.py --duration 30 --fps 30 \
       --output ./recordings --dataset f1_vr_v2 --task "pick up the cube"
   ```

   Per frame: parse UDP packets → solve IK for each tracked hand
   ([Section 3](#3-inverse-kinematics)) → append `observation.state` /
   `action` (17-D joints) plus raw hand data → save
   `recordings/episode_XXXXXX.h5` → export Parquet + metadata to
   `datasets/f1_vr_v2`.

4. **Alternative: record raw, solve IK offline** — record with
   `05_quick_record.py` / `04_vr_hand_record.py`, then:

   ```bash
   python scripts/07_offline_ik.py \
       --input recordings/episode_000000.h5 --dataset f1_vr_v2
   ```

   This writes `episode_000000.ik.h5` (HDF5 + IK result) and appends the
   episode to the LeRobot dataset.

5. **Validate the dataset** — see [Section 5](#5-data-integrity-validation).
6. **Compute statistics** — `scripts/statistics.py` (frames, IK metrics,
   distributions, figures in `assets/`).
7. **Train** — `scripts/train_bc.py --config configs/train_bc.yaml`.

## 3. Inverse Kinematics

### 3.1 Method (`scripts/ik_solver.py`)

The solver is pure NumPy/SciPy (no ROS, no pybullet/pinocchio):

1. **URDF parsing** — `xml.etree` reads every joint's parent/child link,
   origin (xyz + rpy), axis, and limits into dictionaries.
2. **Kinematic chains** — links are walked from the end effector back to
   `base_link`, yielding the per-arm chains:

   ```
   right: base_link → lift → waist1 → waist2 → J1_R → J2_R → ... → J7_R
   left:  base_link → lift → waist1 → waist2 → J1_L → J2_L → ... → J7_L
   ```

   Only revolute/prismatic joints are controllable (fixed links and the
   continuous wheel joints are excluded) → 10 joints per arm chain, of
   which the 3 torso joints are shared by both arms (17 unique DOF).
3. **Forward kinematics** — the 4×4 transform of each joint is built from
   its URDF origin and rotation about its axis; chain transforms are
   multiplied to obtain the end-effector pose.
4. **Numerical IK** — scipy `L-BFGS-B` minimizes

   ```
   cost = || p_FK - p_target || + 0.5 * angle(R_FK, R_target)
   ```

   (1 m position error weighted like 1 rad rotation error) subject to the
   URDF joint limits. Two initial guesses are tried per solve and the
   better result is kept:

   - **warm start** — the previous frame's solution (temporal continuity;
     the standard case during recording),
   - **zero pose** — fallback that escapes the warm start when it stalls.

   Optimizer settings: `maxiter=30`, `ftol=1e-4`.
5. **Torso merging** — the torso is shared by both arms, so the recorded
   torso target is the average of the two arms' solutions:
   `torso = (torso_right + torso_left) / 2`.

### 3.2 Hand → robot coordinate mapping

Quest 3 coordinates are remapped to the robot base frame:

```
Quest:  X = right, Y = up, Z = backward
Robot:  X = forward, Y = left, Z = up

target_x = -hand_z * scale + offset_x
target_y = -hand_x * scale + offset_y
target_z =  hand_y * scale + offset_z
```

with `scale = 0.8`, `offset = [0.35, 0, 0.5]`.

> ⚠️ **These scale/offset values are a provisional visual estimate and
> have not been physically calibrated.** Hand orientations are currently
> forwarded without rotation into the robot frame. This is the primary
> open issue for quantitative accuracy claims — see the IK metrics below
> and `docs/paper_draft_notes.md`.

### 3.3 IK success-rate calculation

Two complementary metrics (both computed by `scripts/statistics.py`):

1. **Pipeline-reported solve rate** — the fraction of *tracked* hand
   frames (wrist position norm > 1 mm) for which the pipeline produced a
   joint solution without error. Stored in the HDF5 attributes
   `ik_left_solved` / `ik_right_solved` (values also stored in the
   LeRobot export). **Current value: 6,977 / 6,983 = 99.9%**
   (100% of tracked frames; the 6 untracked frames are the difference).
2. **FK-verified accuracy** — a stricter, paper-grade metric: for every
   recorded frame, forward kinematics is re-run on the stored joint
   angles and compared to the hand-pose target in the robot frame. A
   frame passes if

   ```
   position error ≤ 1 cm  AND  orientation error ≤ 5°
   ```

   **Current status: pending.** With the provisional mapping, FK errors
   are systematically above the thresholds (median tens of cm — see
   `assets/ik_position_error.png`). This means the recorded targets do
   not yet accurately track the operator's hand; physical calibration of
   `scale`/`offset` (or a fitted mapping) is required before accuracy
   numbers can be reported. Until then, only metric 1 should be quoted
   as the pipeline's solve rate.

## 4. Data Saving and LeRobot Conversion

### 4.1 HDF5 layout (`recordings/episode_XXXXXX.h5`)

| Dataset | Shape | Content |
|---------|-------|---------|
| `timestamp` | (N,) | seconds since episode start |
| `observation.state` | (N, 17) | current joint angles (IK output) |
| `action` | (N, 17) | target joint angles (same values; VLA-ready) |
| `observation.left_hand.wrist_pose` | (N, 7) | xyz + qxyzw |
| `observation.left_hand.landmarks` | (N, 66) | 22 joints × 3 |
| `observation.right_hand.*` | — | same as left |
| `observation.gripper_cmd` | (N,) | pinch-distance gripper estimate |
| `episode_index` / `task_index` / `index` | (N,) | LeRobot bookkeeping |
| `next.done` / `next.reward` | (N,) | episode boundary / success flag |

Attributes: `fps`, `total_frames`, `source`, `task`, `recorded_at`,
`ik_*_solved`, `n_joints`.

### 4.2 LeRobot v3.0 conversion (`scripts/03_convert_to_lerobot.py`)

The converter writes, per episode, a Parquet file containing the eight
required LeRobot columns (`observation.state`, `action`, `timestamp`,
`episode_index`, `index`, `task_index`, `next.done`, `next.reward`) and
generates `meta/info.json` (with the mandatory `features` dictionary),
`meta/episodes.jsonl`, `meta/tasks.jsonl`, and `meta/modality.json`
(state/action segment names for VLA models such as π₀.₅). Full format
details: [lerobot_data_format.md](lerobot_data_format.md).

## 5. Data Integrity Validation

Run these checks after every recording session:

```bash
# 1. Environment + dependencies
python scripts/01_test_environment.py

# 2. Statistics: episode/frame counts, IK metrics, distributions
python scripts/statistics.py \
    --dataset ./datasets/f1_vr_v2 \
    --ik-h5 "./recordings/episode_*.ik.h5" \
    --assets ./assets

# 3. LeRobot library loads the dataset without errors
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; \
           print(LeRobotDataset('./datasets/f1_vr_v2'))"

# 4. Every recorded frame has a finite, within-limits joint vector
#    (checked implicitly by statistics.py; joint limits come from the URDF)
```

Common failure modes and checks:

| Symptom | Check |
|---------|-------|
| Zero UDP packets | Headset on same LAN? APK configured with the PC IP? Firewall? |
| All-zero hand data | Is hand tracking enabled in the headset (tracking dots visible)? |
| `UnicodeEncodeError` | Avoid non-ASCII characters in paths |
| LeRobot load failure | `meta/info.json` present with a valid `features` dict? |
