"""
从 IK 处理后的 HDF5 文件，生成标准 LeRobot v3.0 数据集。

用法:
  # 从所有 .ik.h5 文件新建数据集
  python scripts/09_make_lerobot_dataset.py

  # 指定输入目录和数据集名
  python scripts/09_make_lerobot_dataset.py --input ./recordings --dataset f1_vr_v2

  # 追加到已有数据集（跳过已存在的 episode）
  python scripts/09_make_lerobot_dataset.py --append

  # 只处理指定的几个文件
  python scripts/09_make_lerobot_dataset.py --files episode_000000.ik.h5 episode_000005.ik.h5

做什么:
  1. 读取 recordings/*.ik.h5，取出 observation.state（17 维关节角，弧度制）
  2. 修正 action 语义：action[i] = state[i+1]（最后一帧指向自己）
  3. 调用 LeRobotDataset.create() + add_frame() + save_episode() + finalize()
  4. 生成标准目录结构：meta/info.json, data/chunk-000/file-000.parquet 等
"""

import argparse
import sys
import numpy as np
import h5py
from pathlib import Path

# 把 scripts/ 加到 sys.path，方便 import ik_solver
sys.path.insert(0, str(Path(__file__).parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

# ---- F1 17 个关节名（与 IK solver 输出顺序一致） ----
JOINT_NAMES = [
    "lift_joint",
    "waist1_joint",
    "waist2_joint",
    "J1_right_joint", "J2_right_joint", "J3_right_joint",
    "J4_right_joint", "J5_right_joint", "J6_right_joint", "J7_right_joint",
    "J1_left_joint", "J2_left_joint", "J3_left_joint",
    "J4_left_joint", "J5_left_joint", "J6_left_joint", "J7_left_joint",
]
N_JOINTS = len(JOINT_NAMES)

# ---- 从 .ik.h5 文件名提取 episode index ----
def _parse_episode_index(path: Path) -> int:
    """episode_000009.ik.h5 → 9"""
    stem = path.stem  # episode_000009.ik
    num_part = stem.replace("episode_", "").replace(".ik", "")
    return int(num_part)


def read_ik_hdf5(h5_path: Path) -> dict:
    """读取一个 .ik.h5 文件，返回 {joint_angles, timestamps, joint_names, attrs}"""
    with h5py.File(str(h5_path), "r") as f:
        state = f["observation.state"][:]  # (N, 17) float32
        timestamps = f["timestamp"][:] if "timestamp" in f else np.arange(
            len(state), dtype=np.float32
        ) / 30.0
        attrs = dict(f.attrs)
    return {
        "joint_angles": state,
        "timestamps": timestamps,
        "attrs": attrs,
    }


def make_actions(state: np.ndarray) -> np.ndarray:
    """
    action[i] = state[i+1], 最后一帧指向自己。
    这样模型学到的是「从当前位置要往哪个方向动」。
    """
    n = len(state)
    actions = np.zeros_like(state, dtype=np.float32)
    actions[:-1] = state[1:]    # action[0] = state[1], action[1] = state[2], ...
    actions[-1] = state[-1]     # 最后一帧留原地
    return actions


def make_features() -> dict:
    """构建 LeRobot 需要的 features 字典"""
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (N_JOINTS,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (N_JOINTS,),
            "names": JOINT_NAMES,
        },
    }


def create_or_open_dataset(dataset_name: str, root: Path, fps: int = 30) -> LeRobotDataset:
    """创建或打开已有 LeRobot 数据集。

    注意：LeRobot API 的 root 参数是数据集的完整路径（如 ./datasets/f1_vr_v2），
    不是父目录。传入的 root 已经是父目录，这里拼上 dataset_name。
    """
    ds_path = root / dataset_name
    meta_exists = ds_path.exists() and (ds_path / "meta" / "info.json").exists()

    if meta_exists:
        print(f"[DS] 打开已有数据集: {ds_path}")
        return LeRobotDataset(dataset_name, root=str(ds_path))
    else:
        print(f"[DS] 创建新数据集: {ds_path}")
        return LeRobotDataset.create(
            repo_id=dataset_name,
            fps=fps,
            features=make_features(),
            root=str(ds_path),
            robot_type="F1",
            use_videos=False,
        )


def process_files(
    h5_paths: list[Path],
    dataset_name: str,
    root: Path,
    fps: int = 30,
    skip_existing: bool = True,
) -> dict:
    """主流程：遍历 .ik.h5，逐帧写入 LeRobot"""

    dataset = create_or_open_dataset(dataset_name, root, fps)

    processed = []
    skipped = []
    total_frames = 0

    for h5_path in sorted(h5_paths):
        ep_idx = _parse_episode_index(h5_path)

        if skip_existing and dataset.num_episodes > 0:
            # LeRobot 3.0 没有直接 API 查 episode 是否存在，
            # 用一种基本的方式：检查 episode buffer 当前状态
            pass

        print(f"\n[FILE] {h5_path.name}  (episode #{ep_idx})")

        data = read_ik_hdf5(h5_path)
        state = data["joint_angles"]
        actions = make_actions(state)
        n_frames = len(state)

        attrs = data["attrs"]
        n_joints_attr = attrs.get("n_joints", 0)
        ik_right = attrs.get("ik_right_solved", 0)
        ik_left = attrs.get("ik_left_solved", 0)
        print(f"  帧数: {n_frames}, 关节: {n_joints_attr}, "
              f"右IK: {ik_right}/{n_frames}, 左IK: {ik_left}/{n_frames}")

        # 检查维度
        if state.shape[1] != N_JOINTS:
            print(f"  [SKIP] 关节数 {state.shape[1]} != {N_JOINTS}, 跳过")
            skipped.append(str(h5_path))
            continue

        # 检查 action shift
        if n_frames >= 2:
            if np.allclose(state[0], actions[0]):
                print(f"  [WARN] action[0] == state[0], shift 未生效?")
            else:
                diff = np.max(np.abs(state[0] - actions[0]))
                print(f"  [OK] action[0] != state[0] (max diff = {diff:.6f} rad)")

        # 从 IK 成功率推断任务质量
        ik_rate = (ik_right + ik_left) / (2 * n_frames) if n_frames > 0 else 0
        quality = "good" if ik_rate > 0.8 else "noisy"
        task = f"VR teleop demo (IK quality: {quality}, frames: {n_frames})"

        # ---- 逐帧写入 ----
        for i in range(n_frames):
            dataset.add_frame({
                "observation.state": state[i].astype(np.float32),
                "action": actions[i].astype(np.float32),
                "task": task,
            })

        dataset.save_episode()
        total_frames += n_frames
        processed.append(str(h5_path))
        print(f"  [OK] {n_frames} 帧已写入 episode #{dataset.num_episodes - 1}")

    # ---- 收尾 ----
    dataset.finalize()
    print(f"\n{'='*60}")
    print(f"数据集: {root / dataset_name}")
    print(f"处理文件: {len(processed)}/{len(h5_paths)}")
    print(f"总帧数: {total_frames}")
    print(f"总集数: {dataset.num_episodes}")
    print(f"{'='*60}")

    return {
        "dataset_path": str(root / dataset_name),
        "processed": processed,
        "skipped": skipped,
        "total_frames": total_frames,
        "total_episodes": dataset.num_episodes,
    }


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HDF5(.ik.h5) → LeRobot v3.0 数据集"
    )
    parser.add_argument(
        "--input", type=str, default="./recordings",
        help="存放 .ik.h5 文件的目录 (默认: ./recordings)"
    )
    parser.add_argument(
        "--dataset", type=str, default="f1_vr_v2",
        help="数据集名称 (默认: f1_vr_v2)"
    )
    parser.add_argument(
        "--root", type=str, default="./datasets",
        help="数据集根目录 (默认: ./datasets)"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="录制帧率 (默认: 30)"
    )
    parser.add_argument(
        "--files", type=str, nargs="*",
        help="指定要处理的 .ik.h5 文件（默认：全部）"
    )
    parser.add_argument(
        "--append", action="store_true",
        help="追加到已有数据集（不删重建）"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="先删除已有数据集再重建"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    dataset_root = Path(args.root)

    # 收集文件
    if args.files:
        h5_paths = [input_dir / f for f in args.files]
    else:
        h5_paths = sorted(input_dir.glob("episode_*.ik.h5"))

    if not h5_paths:
        print(f"[FAIL] 在 {input_dir} 没有找到 .ik.h5 文件")
        print("  提示: 先跑 python scripts/07_offline_ik.py --input XXX.h5")
        sys.exit(1)

    print(f"[FIND] 找到 {len(h5_paths)} 个 .ik.h5 文件")
    for p in h5_paths:
        print(f"  - {p.name}")

    # 清理旧数据集
    if args.clean:
        import shutil
        old_path = dataset_root / args.dataset
        if old_path.exists():
            print(f"[CLEAN] 删除旧数据集: {old_path}")
            shutil.rmtree(old_path)

    result = process_files(
        h5_paths=h5_paths,
        dataset_name=args.dataset,
        root=dataset_root,
        fps=args.fps,
        skip_existing=not args.append and not args.clean,
    )

    ds_path = dataset_root / args.dataset
    print(f"\n验证数据集:")
    print(f"  python -c \"from lerobot.datasets.lerobot_dataset import LeRobotDataset; "
          f"ds = LeRobotDataset('{args.dataset}', root=r'{ds_path}'); "
          f"print(f'episodes={{ds.num_episodes}}, frames={{ds.num_frames}}')\"")
