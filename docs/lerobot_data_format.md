# LeRobot v3.0 Data Format Reference

This is an English reference for the [LeRobot](https://github.com/huggingface/lerobot)
v3.0 dataset format used by this pipeline, distilled from the official
documentation and the `openpi` (π₀.₅) training stack. It complements
[data_collection_pipeline.md](data_collection_pipeline.md).

## 1. Directory Layout

```
<dataset_root>/
├── meta/
│   ├── info.json          # dataset metadata: features dict, fps, episode/frame counts (REQUIRED)
│   ├── episodes.jsonl     # one JSON line per episode: index, tasks, length
│   ├── tasks.jsonl        # one JSON line per task: index, description
│   ├── stats.json         # normalization statistics (generated before training)
│   └── modality.json      # state/action segment names (used by π₀.₅-style VLA stacks)
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet   # this repo's converter layout
│       └── ...                      # (or file-000.parquet, chunked layout)
└── videos/
    └── chunk-000/          # optional episode videos
```

## 2. Required Parquet Columns

| Column | dtype | shape | Meaning |
|--------|-------|-------|---------|
| `observation.state` | float32 | [n] | current robot state (joint angles) |
| `action` | float32 | [n] | target action (joint angles) |
| `timestamp` | float32 | [1] | seconds within the episode |
| `episode_index` | int64 | [1] | episode id |
| `index` | int64 | [1] | global frame id |
| `task_index` | int64 | [1] | task id |
| `next.done` | bool | [1] | true on the last frame of an episode |
| `next.reward` | float32 | [1] | 1.0 on success, else 0.0 |

Optional columns used by this project / vision datasets:

| Column | Meaning |
|--------|---------|
| `observation.images.<cam_name>` | camera images (path or bytes) |
| `observation.hand_pose` | VR hand pose (6D) |
| `observation.gripper_cmd` | gripper command (0–1) |
| `observation.left_hand.wrist_pose` / `landmarks`, `observation.right_hand.*` | raw Quest 3 hand data (HDF5 only) |

## 3. meta/info.json

The `features` dictionary is **mandatory** — LeRobot's `load_info()`
raises `KeyError` without it. Each key declares the dtype and shape of
its column:

```json
{
  "codebase_version": "v3.0",
  "robot_type": "F1",
  "fps": 30,
  "total_episodes": 7,
  "total_frames": 3913,
  "features": {
    "observation.state": {"dtype": "float32", "shape": [17]},
    "action":             {"dtype": "float32", "shape": [17]},
    "timestamp":          {"dtype": "float32", "shape": [1]},
    "episode_index":      {"dtype": "int64",   "shape": [1]},
    "index":              {"dtype": "int64",   "shape": [1]},
    "task_index":         {"dtype": "int64",   "shape": [1]},
    "next.done":          {"dtype": "bool",    "shape": [1]},
    "next.reward":        {"dtype": "float32", "shape": [1]}
  }
}
```

For this robot, `observation.state` / `action` are 17-dimensional:
`[lift, waist1, waist2, J1_R..J7_R, J1_L..J7_L]` (radians).

## 4. The Four Internal Transforms (π₀.₅-style stacks)

Downstream VLA training stacks transform the raw Parquet data in four
stages. You do not need to reimplement them, but the column names above
must match what the stack expects:

```
Raw Parquet
   │  RepackTransform    column-name mapping to internal names
   ▼
DataTransform      absolute vs delta actions
   │
   ▼
ModelTransform     image resize / tokenize / pad state & action to fixed dims
   │
   ▼
Normalization      quantile normalization using precomputed stats
```

### Absolute vs delta actions

- **Absolute actions**: `action` = absolute target joint angles (this
  project's convention — it matches VR teleoperation directly).
- **Delta actions**: `action` = joint-angle change per step. Requires
  `use_delta_action`-style flags in the training config.

### Normalization

π₀.₅ defaults to **quantile normalization** for STATE and ACTION:

```
normalized = (x - q01) / (q99 - q01)
```

where `q01`/`q99` are the 1%/99% percentiles computed over the training
set and stored in `meta/stats.json` (e.g. via `lerobot`'s
`compute_norm_stats`). Images are not normalized (identity mapping).

## 5. Training

LeRobot's training CLI consumes the dataset directory directly:

```bash
# Behavior cloning with ACT (see configs/train_bc.yaml for this repo)
python scripts/train_bc.py --config configs/train_bc.yaml

# or, raw LeRobot CLI (π₀.₅ fine-tuning example)
lerobot-train \
    --dataset.repo_id=your_org/your_dataset \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --output_dir=./outputs/pi05_f1 \
    --batch_size=32
```

Key parameters for 17-DOF state-only data:

| Parameter | Suggested value |
|-----------|-----------------|
| `policy.type` | `act` (BC baseline) or `pi05` (VLA fine-tune) |
| action convention | absolute (`use_delta_action: false`) |
| `max_state_dim` / `max_action_dim` | 32 (auto-padded) |
| LoRA (low VRAM) | disable EMA (`ema_decay=None`) |

## 6. Validating a Dataset

```bash
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; \
           print(LeRobotDataset('./datasets/f1_vr_v2'))"
```

Common failure: missing `features` dict in `meta/info.json` → `KeyError`
from `load_info()`.
