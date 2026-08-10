"""Quick record: capture N seconds from Quest 3 using raw UDP, save to HDF5 + LeRobot"""
import socket, struct, h5py, numpy as np, time, argparse, json
from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=float, default=5.0)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--output", type=str, default="./recordings")
args = parser.parse_args()

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

# Open UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 9000))
sock.settimeout(1.0)  # 1 second timeout

print(f"[REC] Capturing {args.duration}s of hand data on UDP 9000...")
print(f"  Wave your hands in front of the Quest 3!")

# Storage: raw packet buffers
packets = []  # (timestamp, data_bytes)

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

# Parse packets into frames
# Each packet is a CSV line: type,side,values...
# Wrist: wrist,left,x,y,z,qx,qy,qz
# Landmarks: landmarks,left,x0,y0,z0,x1,y1,z1,... (63 floats)
left_wrist = np.zeros(7, dtype=np.float32)
right_wrist = np.zeros(7, dtype=np.float32)
left_lm = np.zeros(66, dtype=np.float32)   # 22 joints * 3
right_lm = np.zeros(66, dtype=np.float32)

timestamps = []
left_wrists, right_wrists = [], []
left_landmarks, right_landmarks = [], []

frame_interval = 1.0 / args.fps
last_ts = -frame_interval

for ts, raw in packets:
    try:
        text = raw.decode("utf-8").strip()
        # Each packet has wrist + landmarks separated by newline
        for line in text.split("\n"):
            # Format: "Left wrist | f = 3702 | t = 1733651663500:, -0.151, 0.585, ..."
            if "|" not in line or ":" not in line:
                continue
            # Split metadata from values
            meta, vals_str = line.split(":", 1)
            if not vals_str.strip():
                continue
            # Parse type and side: "Left wrist" or "Right landmarks"
            type_side = meta.split("|")[0].strip()
            parts = type_side.split(" ")
            if len(parts) < 2:
                continue
            side = parts[0]  # "Left" or "Right"
            ptype = parts[1]  # "wrist" or "landmarks"
            vals = [float(x) for x in vals_str.strip().strip(",").split(",") if x.strip()]

            if ptype == "wrist" and len(vals) == 7:
                if side == "Left":
                    left_wrist = np.array(vals, dtype=np.float32)
                else:
                    right_wrist = np.array(vals, dtype=np.float32)
            elif ptype == "landmarks" and len(vals) >= 63:
                if side == "Left":
                    left_lm = np.array(vals[:66], dtype=np.float32) if len(vals) >= 66 else np.array(vals, dtype=np.float32)
                else:
                    right_lm = np.array(vals[:66], dtype=np.float32) if len(vals) >= 66 else np.array(vals, dtype=np.float32)
    except:
        continue

    # Record at fixed FPS
    if ts - last_ts >= frame_interval:
        timestamps.append(ts)
        left_wrists.append(left_wrist.copy())
        right_wrists.append(right_wrist.copy())
        left_landmarks.append(left_lm.copy())
        right_landmarks.append(right_lm.copy())
        last_ts = ts

n_frames = len(timestamps)
print(f"  Recorded {n_frames} frames at {args.fps} FPS")

# Save HDF5
# auto-increment episode number
existing = sorted(output_dir.glob("episode_*.h5"))
next_idx = len(existing)
h5_path = output_dir / f"episode_{next_idx:06d}.h5"
with h5py.File(str(h5_path), "w") as f:
    f.create_dataset("timestamp", data=np.array(timestamps, dtype=np.float32))
    f.create_dataset("observation.left_hand.wrist_pose", data=np.array([x[:7] for x in left_wrists], dtype=np.float32))
    f.create_dataset("observation.left_hand.landmarks", data=np.array([list(x[:66]) + [0.0]*(66-len(x)) if len(x) < 66 else x[:66] for x in left_landmarks], dtype=np.float32))
    f.create_dataset("observation.right_hand.wrist_pose", data=np.array([x[:7] for x in right_wrists], dtype=np.float32))
    f.create_dataset("observation.right_hand.landmarks", data=np.array([list(x[:66]) + [0.0]*(66-len(x)) if len(x) < 66 else x[:66] for x in right_landmarks], dtype=np.float32))
    f.create_dataset("observation.state", data=np.zeros((n_frames, 6), dtype=np.float32))
    f.create_dataset("action", data=np.zeros((n_frames, 6), dtype=np.float32))
    f.create_dataset("observation.gripper_cmd", data=np.zeros(n_frames, dtype=np.float32))
    f.create_dataset("episode_index", data=np.zeros(n_frames, dtype=np.int64))
    f.create_dataset("task_index", data=np.zeros(n_frames, dtype=np.int64))
    f.create_dataset("index", data=np.arange(n_frames, dtype=np.int64))
    f.create_dataset("next.done", data=np.array([False]*n_frames))
    f.create_dataset("next.reward", data=np.zeros(n_frames, dtype=np.float32))
    f.attrs["fps"] = args.fps
    f.attrs["total_frames"] = n_frames
    f.attrs["source"] = "Quest3_hand_tracking"
    f.attrs["recorded_at"] = datetime.now().isoformat()

print(f"  HDF5: {h5_path}")

# Convert to LeRobot
import pandas as pd
ds_dir = Path("./datasets/vr_test_v1")
(ds_dir / "meta").mkdir(parents=True, exist_ok=True)
(ds_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

records = []
for i in range(n_frames):
    records.append({
        "observation.state": [0.0]*6,
        "action": [0.0]*6,
        "timestamp": float(timestamps[i]),
        "episode_index": 0,
        "index": i,
        "task_index": 0,
        "next.done": (i == n_frames - 1),
        "next.reward": 1.0 if i == n_frames - 1 else 0.0,
    })

df = pd.DataFrame(records)
pq_path = ds_dir / "data" / "chunk-000" / "episode_000000.parquet"
df.to_parquet(str(pq_path), engine="pyarrow")

features = {
    "observation.state": {"dtype": "float32", "shape": [6]},
    "action": {"dtype": "float32", "shape": [6]},
    "timestamp": {"dtype": "float32", "shape": [1]},
    "episode_index": {"dtype": "int64", "shape": [1]},
    "index": {"dtype": "int64", "shape": [1]},
    "task_index": {"dtype": "int64", "shape": [1]},
    "next.done": {"dtype": "bool", "shape": [1]},
    "next.reward": {"dtype": "float32", "shape": [1]},
}
with open(ds_dir / "meta" / "info.json", "w") as f:
    json.dump({"codebase_version": "v3.0", "robot_type": "F1", "fps": 30,
               "total_episodes": 1, "total_frames": n_frames, "features": features}, f, indent=2)
with open(ds_dir / "meta" / "episodes.jsonl", "w") as f:
    f.write(json.dumps({"episode_index": 0, "tasks": ["pick up the cube"], "length": n_frames}) + "\n")
with open(ds_dir / "meta" / "tasks.jsonl", "w") as f:
    f.write(json.dumps({"task_index": 0, "task": "pick up the cube"}) + "\n")

print(f"  LeRobot: {ds_dir}")
print(f"\n=== SUCCESS ===")
print(f"  Frames recorded: {n_frames}")
print(f"  HDF5: {h5_path}")
print(f"  LeRobot: {ds_dir}")
