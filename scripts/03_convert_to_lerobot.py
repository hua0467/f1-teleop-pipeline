"""
F1 遥操数据采集管线 —— HDF5 → LeRobot v2.1 格式转换

把 02_record_episode.py 录制的 HDF5 文件转成 LeRobot 标准数据集，
可以直接喂给 π₀.₅ 做后训练。

使用方式：
    python 03_convert_to_lerobot.py --input ./recordings --output ./datasets/pick_cube_v1
"""

import h5py
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from datetime import datetime


def convert_hdf5_to_lerobot(
    hdf5_dir: str,
    output_dir: str,
    fps: int = 30,
    robot_type: str = "F1",
    task_description: str = "pick up the cube",
    state_keys: list = None,
    action_keys: list = None,
):
    """
    把 HDF5 文件目录转成 LeRobot v2.1 数据集

    参数:
        hdf5_dir:     HDF5 文件所在目录
        output_dir:   输出目录（LeRobot 数据集根目录）
        fps:          录制帧率
        robot_type:   机器人型号
        task_description: 任务描述文本
        state_keys:   状态数组各段的名称列表（用于 modality.json）
        action_keys:  动作数组各段的名称列表（用于 modality.json）
    """
    hdf5_dir = Path(hdf5_dir)
    output_dir = Path(output_dir)

    # 目录结构
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)

    episode_files = sorted(hdf5_dir.glob("episode_*.h5"))
    if not episode_files:
        print(f"[FAIL] No episode_*.h5 files found in {hdf5_dir}")
        return

    print(f"[DIR] Found {len(episode_files)} episode files\n")

    global_index = 0
    episodes_info = []
    all_state_shapes = []
    all_action_shapes = []

    for ep_idx, h5_path in enumerate(episode_files):
        with h5py.File(str(h5_path), "r") as f:
            n_frames = len(f["timestamp"])

            # 记录 shape 信息
            state_shape = f["observation.state"][0].shape
            action_shape = f["action"][0].shape
            all_state_shapes.append(state_shape)
            all_action_shapes.append(action_shape)

            # 构建 DataFrame
            records = []
            for i in range(n_frames):
                records.append({
                    "observation.state": f["observation.state"][i].tolist(),
                    "action": f["action"][i].tolist(),
                    "timestamp": float(f["timestamp"][i]),
                    "episode_index": ep_idx,
                    "index": global_index + i,
                    "task_index": 0,
                    "next.done": bool(f["next.done"][i]),
                    "next.reward": float(f["next.reward"][i]),
                })

            df = pd.DataFrame(records)

            # 写 Parquet
            parquet_path = output_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
            df.to_parquet(str(parquet_path), engine="pyarrow")

            duration = float(f["timestamp"][-1])
            episodes_info.append({
                "episode_index": ep_idx,
                "tasks": [task_description],
                "length": n_frames,
            })

            global_index += n_frames
            print(f"  Episode {ep_idx}: {n_frames:4d} frames | {duration:5.1f}s → {parquet_path.name}")

    # --- 写 meta/info.json ---
    state_dim = all_state_shapes[0][0]
    action_dim = all_action_shapes[0][0]

    # --- 构建 features 字典（LeRobot 库必须）---
    features = {
        "observation.state": {"dtype": "float32", "shape": [state_dim]},
        "action":             {"dtype": "float32", "shape": [action_dim]},
        "timestamp":          {"dtype": "float32", "shape": [1]},
        "episode_index":      {"dtype": "int64",   "shape": [1]},
        "index":              {"dtype": "int64",   "shape": [1]},
        "task_index":         {"dtype": "int64",   "shape": [1]},
        "next.done":          {"dtype": "bool",    "shape": [1]},
        "next.reward":        {"dtype": "float32", "shape": [1]},
    }
    # 如果有手部位姿数据，也加入 features
    h5_sample = h5py.File(str(episode_files[0]), "r")
    if "observation.hand_pose" in h5_sample:
        hp_dim = h5_sample["observation.hand_pose"][0].shape[0]
        features["observation.hand_pose"] = {"dtype": "float32", "shape": [hp_dim]}
    if "observation.gripper_cmd" in h5_sample:
        features["observation.gripper_cmd"] = {"dtype": "float32", "shape": [1]}
    h5_sample.close()

    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "fps": fps,
        "total_episodes": len(episode_files),
        "total_frames": global_index,
        "total_tasks": 1,
        "chunks_size": 100,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "splits": {},
        "data_path": "data/chunk-{file_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": features,
        "created_at": datetime.now().isoformat(),
    }
    with open(output_dir / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # --- 写 meta/episodes.jsonl ---
    with open(output_dir / "meta" / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in episodes_info:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    # --- 写 meta/tasks.jsonl ---
    with open(output_dir / "meta" / "tasks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "task_index": 0,
            "task": task_description,
        }, ensure_ascii=False) + "\n")

    # --- 写 meta/modality.json（GR00T/π₀.₅ 需要）---
    # 如果没有给具体的 key 列表，就用默认名称
    if state_keys is None:
        state_keys = [f"joint_{i}" for i in range(state_dim)]
    if action_keys is None:
        action_keys = [f"joint_{i}" for i in range(action_dim)]

    modality = {
        "state": {},
        "action": {},
    }
    for i, key in enumerate(state_keys):
        modality["state"][key] = {"start": i, "end": i + 1}
    for i, key in enumerate(action_keys):
        modality["action"][key] = {"start": i, "end": i + 1, "absolute": True}

    with open(output_dir / "meta" / "modality.json", "w", encoding="utf-8") as f:
        json.dump(modality, f, indent=2, ensure_ascii=False)

    # --- 打印摘要 ---
    print(f"\n{'='*60}")
    print(f"[OK] Conversion complete")
    print(f"{'='*60}")
    print(f"  数据集路径:  {output_dir}")
    print(f"  Episodes:    {len(episode_files)}")
    print(f"  总帧数:      {global_index}")
    print(f"  State 维度:  {state_dim}")
    print(f"  Action 维度: {action_dim}")
    print(f"  任务:        {task_description}")
    print(f"\n  下一步: 把这个目录交给做训练的同学")
    print(f"  或者用 lerobot 库验证: ")
    print(f"    python -c \"from lerobot.datasets.lerobot_dataset import LeRobotDataset; ")
    print(f"    ds = LeRobotDataset('{output_dir}'); print(ds)\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDF5 → LeRobot 格式转换")
    parser.add_argument("--input", type=str, default="./recordings",
                       help="HDF5 文件目录")
    parser.add_argument("--output", type=str, default="./datasets/f1_pick_cube_v1",
                       help="输出目录")
    parser.add_argument("--fps", type=int, default=30,
                       help="录制帧率")
    parser.add_argument("--robot", type=str, default="F1",
                       help="机器人型号")
    parser.add_argument("--task", type=str, default="pick up the cube",
                       help="任务描述文本")
    parser.add_argument("--state-keys", type=str, nargs="*",
                       help="状态维度名称列表 (如: joint_0 joint_1 gripper)")
    parser.add_argument("--action-keys", type=str, nargs="*",
                       help="动作维度名称列表")
    args = parser.parse_args()

    convert_hdf5_to_lerobot(
        hdf5_dir=args.input,
        output_dir=args.output,
        fps=args.fps,
        robot_type=args.robot,
        task_description=args.task,
        state_keys=args.state_keys,
        action_keys=args.action_keys,
    )
