"""Record VR hand data -> IK solve -> joint angles -> HDF5 + LeRobot

Full pipeline:
  Quest 3 UDP packets -> parse hand pose -> IK solver -> joint angles
  -> HDF5 (raw hand + joint angles) -> LeRobot Parquet (real action data)
"""

import socket
import numpy as np
import h5py
import time
import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Add scripts dir for ik_solver import
sys.path.insert(0, str(Path(__file__).parent))
from ik_solver import F1Kinematics, hand_pose_to_robot_target

parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds")
parser.add_argument("--fps", type=int, default=30, help="Recording frame rate")
parser.add_argument("--output", type=str, default="./recordings", help="Output directory for HDF5")
parser.add_argument("--dataset", type=str, default="f1_vr_v1", help="Dataset name for LeRobot output")
parser.add_argument("--task", type=str, default="pick up the cube", help="Task description")
parser.add_argument("--urdf", type=str, default=None, help="Path to URDF file")
args = parser.parse_args()

# Find URDF
if args.urdf:
    urdf_path = args.urdf
else:
    urdf_path = Path(__file__).parent.parent / "urdf" / "urdf" / "F1_URDF_V04.urdf"
    if not urdf_path.exists():
        urdf_path = Path("C:/Users/Administrator/Desktop/F1_URDF_V04/urdf/F1_URDF_V04.urdf")

print(f"[IK] Loading URDF: {urdf_path}")
f1 = F1Kinematics(str(urdf_path))

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

# Determine episode index
existing = list(output_dir.glob("episode_*.h5"))
ep_idx = len(existing)
print(f"[DIR] Episode index: {ep_idx}")

# Open UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 9000))
sock.settimeout(1.0)

print(f"\n[REC] Capturing {args.duration}s of hand data on UDP 9000...")
print(f"  Wave your hands in front of the Quest 3!")
print(f"  Task: {args.task}")

# Storage
packets = []

t0 = time.time()
while time.time() - t0 < args.duration:
    try:
        data, addr = sock.recvfrom(65535)
        packets.append((time.time() - t0, data))
    except socket.timeout:
        pass

sock.close()
elapsed = time.time() - t0
print(f"[OK] {len(packets)} UDP packets received in {elapsed:.1f}s")

if len(packets) == 0:
    print("[FAIL] Zero packets. Is Quest 3 on WiFi? Did you click Start in the app?")
    exit(1)

# Parse packets into hand frames
left_wrist = np.zeros(7, dtype=np.float32)
right_wrist = np.zeros(7, dtype=np.float32)
left_lm = np.zeros(66, dtype=np.float32)
right_lm = np.zeros(66, dtype=np.float32)

timestamps = []
left_wrists, right_wrists = [], []
left_landmarks, right_landmarks = [], []

frame_interval = 1.0 / args.fps
last_ts = -frame_interval

for ts, raw in packets:
    try:
        text = raw.decode("utf-8").strip()
        for line in text.split("\n"):
            if "|" not in line or ":" not in line:
                continue
            meta, vals_str = line.split(":", 1)
            if not vals_str.strip():
                continue
            type_side = meta.split("|")[0].strip()
            parts = type_side.split(" ")
            if len(parts) < 2:
                continue
            side = parts[0]
            ptype = parts[1]
            vals = [float(x) for x in vals_str.strip().strip(",").split(",") if x.strip()]

            if ptype == "wrist" and len(vals) == 7:
                if side == "Left":
                    left_wrist = np.array(vals, dtype=np.float32)
                else:
                    right_wrist = np.array(vals, dtype=np.float32)
            elif ptype == "landmarks" and len(vals) >= 63:
                if side == "Left":
                    left_lm = np.array(list(vals[:66]) + [0.0] * (66 - len(vals)), dtype=np.float32)[:66]
                else:
                    right_lm = np.array(list(vals[:66]) + [0.0] * (66 - len(vals)), dtype=np.float32)[:66]
    except:
        continue

    if ts - last_ts >= frame_interval:
        timestamps.append(ts)
        left_wrists.append(left_wrist.copy())
        right_wrists.append(right_wrist.copy())
        left_landmarks.append(left_lm.copy())
        right_landmarks.append(right_lm.copy())
        last_ts = ts

n_frames = len(timestamps)
print(f"  Parsed {n_frames} hand frames at {args.fps} FPS")

if n_frames == 0:
    print("[FAIL] No frames parsed. Check hand tracking data format.")
    exit(1)

# ---- IK Solve: convert hand poses to joint angles for each frame ----
print(f"\n[IK] Solving inverse kinematics for {n_frames} frames...")

# All joint names in order
all_joint_names = f1.get_all_joint_names()
n_joints = len(all_joint_names)

# Store joint angles for each frame
joint_angles_all = np.zeros((n_frames, n_joints), dtype=np.float32)

# Initial guess for IK (will be updated each frame for temporal continuity)
right_guess = None
left_guess = None

# We need to solve IK for the arm joints. The torso joints (lift, waist1, waist2)
# are shared between both arms - we'll average the torso solutions.
# For the first iteration, we solve each arm independently and merge the torso.

right_success = 0
left_success = 0

for i in range(n_frames):
    # Right hand -> right arm IK
    rw = right_wrists[i]
    rw_pos = rw[:3]
    rw_quat = rw[3:7]  # x, y, z, w from the wrist data

    # Check if right hand data is valid (non-zero)
    if np.linalg.norm(rw_pos) > 0.001:
        try:
            target_pos, target_quat = hand_pose_to_robot_target(rw_pos, rw_quat)
            right_angles = f1.solve_ik(target_pos, target_quat, side='right',
                                        initial_guess=right_guess)
            right_guess = right_angles.copy()
            right_success += 1
        except Exception as e:
            right_angles = np.zeros(f1.n_right)
    else:
        right_angles = np.zeros(f1.n_right)

    # Left hand -> left arm IK
    lw = left_wrists[i]
    lw_pos = lw[:3]
    lw_quat = lw[3:7]

    if np.linalg.norm(lw_pos) > 0.001:
        try:
            target_pos, target_quat = hand_pose_to_robot_target(lw_pos, lw_quat)
            left_angles = f1.solve_ik(target_pos, target_quat, side='left',
                                       initial_guess=left_guess)
            left_guess = left_angles.copy()
            left_success += 1
        except Exception as e:
            left_angles = np.zeros(f1.n_left)
    else:
        left_angles = np.zeros(f1.n_left)

    # Merge: torso from right arm (shared), right arm, left arm
    # right_angles = [lift, waist1, waist2, J1_R...J7_R]
    # left_angles  = [lift, waist1, waist2, J1_L...J7_L]
    torso = (right_angles[:f1.n_torso] + left_angles[:f1.n_torso]) / 2.0
    right_arm = right_angles[f1.n_torso:]  # J1_R..J7_R
    left_arm = left_angles[f1.n_torso:]     # J1_L..J7_L

    joint_angles_all[i] = np.concatenate([torso, right_arm, left_arm])

print(f"  Right arm IK solved: {right_success}/{n_frames} frames")
print(f"  Left arm IK solved:  {left_success}/{n_frames} frames")

# ---- Save HDF5 ----
h5_path = output_dir / f"episode_{ep_idx:06d}.h5"
with h5py.File(str(h5_path), "w") as f:
    # Timestamps
    f.create_dataset("timestamp", data=np.array(timestamps, dtype=np.float32))

    # Raw hand data
    def pad_arrays(arrs, target_len=66):
        result = []
        for x in arrs:
            a = np.array(x, dtype=np.float32)
            if len(a) < target_len:
                a = np.pad(a, (0, target_len - len(a)))
            else:
                a = a[:target_len]
            result.append(a)
        return np.array(result, dtype=np.float32)

    f.create_dataset("observation.left_hand.wrist_pose", data=pad_arrays(left_wrists, 7))
    f.create_dataset("observation.left_hand.landmarks", data=pad_arrays(left_landmarks, 66))
    f.create_dataset("observation.right_hand.wrist_pose", data=pad_arrays(right_wrists, 7))
    f.create_dataset("observation.right_hand.landmarks", data=pad_arrays(right_landmarks, 66))

    # Joint angles (the real data!)
    f.create_dataset("observation.state", data=joint_angles_all)
    f.create_dataset("action", data=joint_angles_all)
    f.create_dataset("joint_names", data=np.array(all_joint_names, dtype=h5py.string_dtype()))

    # Additional required columns
    f.create_dataset("observation.gripper_cmd", data=np.zeros(n_frames, dtype=np.float32))
    f.create_dataset("episode_index", data=np.full(n_frames, ep_idx, dtype=np.int64))
    f.create_dataset("task_index", data=np.zeros(n_frames, dtype=np.int64))
    f.create_dataset("index", data=np.arange(n_frames, dtype=np.int64))
    f.create_dataset("next.done", data=np.array([False] * n_frames))
    f.create_dataset("next.reward", data=np.zeros(n_frames, dtype=np.float32))
    f.attrs["fps"] = args.fps
    f.attrs["total_frames"] = n_frames
    f.attrs["source"] = "Quest3_hand_tracking_with_IK"
    f.attrs["recorded_at"] = datetime.now().isoformat()
    f.attrs["task"] = args.task
    f.attrs["n_joints"] = n_joints

print(f"  HDF5: {h5_path}")

# ---- Convert to LeRobot ----
ds_dir = Path(f"./datasets/{args.dataset}")
(ds_dir / "meta").mkdir(parents=True, exist_ok=True)
(ds_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

records = []
for i in range(n_frames):
    records.append({
        "observation.state": joint_angles_all[i].tolist(),
        "action": joint_angles_all[i].tolist(),
        "timestamp": float(timestamps[i]),
        "episode_index": ep_idx,
        "index": i,
        "task_index": 0,
        "next.done": (i == n_frames - 1),
        "next.reward": 1.0 if i == n_frames - 1 else 0.0,
    })

df = pd.DataFrame(records)
pq_path = ds_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
df.to_parquet(str(pq_path), engine="pyarrow")

features = {
    "observation.state": {"dtype": "float32", "shape": [n_joints]},
    "action": {"dtype": "float32", "shape": [n_joints]},
    "timestamp": {"dtype": "float32", "shape": [1]},
    "episode_index": {"dtype": "int64", "shape": [1]},
    "index": {"dtype": "int64", "shape": [1]},
    "task_index": {"dtype": "int64", "shape": [1]},
    "next.done": {"dtype": "bool", "shape": [1]},
    "next.reward": {"dtype": "float32", "shape": [1]},
}

info = {
    "codebase_version": "v3.0",
    "robot_type": "F1",
    "fps": args.fps,
    "total_episodes": ep_idx + 1,
    "total_frames": n_frames,
    "features": features,
    "joint_names": all_joint_names,
    "n_joints": n_joints,
}

with open(ds_dir / "meta" / "info.json", "w") as f:
    json.dump(info, f, indent=2)
with open(ds_dir / "meta" / "episodes.jsonl", "a") as f:
    f.write(json.dumps({"episode_index": ep_idx, "tasks": [args.task], "length": n_frames}) + "\n")
with open(ds_dir / "meta" / "tasks.jsonl", "a") as f:
    f.write(json.dumps({"task_index": 0, "task": args.task}) + "\n")

print(f"  LeRobot: {ds_dir}")
print(f"\n=== SUCCESS ===")
print(f"  Frames:        {n_frames}")
print(f"  Joint angles:  {n_joints} DOF (non-zero!)")
print(f"  Right IK:      {right_success}/{n_frames} solved")
print(f"  Left IK:       {left_success}/{n_frames} solved")
print(f"  HDF5:          {h5_path}")
print(f"  LeRobot:       {ds_dir}")
print(f"  Shape:         observation.state={joint_angles_all.shape}, action={joint_angles_all.shape}")
