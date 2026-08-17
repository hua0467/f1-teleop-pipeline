"""
Quest 3 录制的 mp4 → 抽帧 → 与 IK 数据对齐 → 生成含视觉的 LeRobot 数据集。

用法:
  # 单个 episode
  python scripts/10_video_to_lerobot.py \
      --video ./recordings/quest3_demo.mp4 \
      --data ./recordings/episode_000009.ik.h5 \
      --dataset f1_vr_vision_v1

  # 批量（文件名按 episode 编号对应）
  python scripts/10_video_to_lerobot.py \
      --video-dir ./videos/ \
      --data-dir ./recordings/ \
      --dataset f1_vr_vision_v1

做什么:
  1. 从 mp4 以目标 fps 均匀抽帧（匹配数据帧数）
  2. 与 .ik.h5 的关节角数据逐帧对齐（按帧序号，不做时间戳对齐）
  3. 调用 LeRobot API 生成含 observation.images.cam 的数据集
"""

import argparse
import sys
import shutil
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 存帧的临时目录（LeRobot 会自动从这里读图再编码）
FRAMES_DIR = Path("./temp_frames")

# ---- F1 关节名 ----
JOINT_NAMES = [
    "lift_joint",
    "waist1_joint", "waist2_joint",
    "J1_right_joint", "J2_right_joint", "J3_right_joint",
    "J4_right_joint", "J5_right_joint", "J6_right_joint", "J7_right_joint",
    "J1_left_joint", "J2_left_joint", "J3_left_joint",
    "J4_left_joint", "J5_left_joint", "J6_left_joint", "J7_left_joint",
]
N_JOINTS = len(JOINT_NAMES)


def extract_frames(
    video_path: Path,
    target_fps: float,
    target_frames: int,
):
    """从 mp4 均匀抽帧，返回 [(frame_idx, bgr_image), ...]"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_video_frames / video_fps if video_fps > 0 else 0

    print(f"[VIDEO] {video_path.name}")
    print(f"  分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  帧率: {video_fps:.2f} fps, 总帧数: {total_video_frames}, 时长: {duration:.1f}s")
    print(f"  数据帧数: {target_frames} @ {target_fps} fps")
    print(f"  数据时长: {target_frames / target_fps:.1f}s")

    # 从视频均匀取 target_frames 帧
    # 策略：从第 0 帧开始，每 step 帧取一帧
    # step = total_video_frames / target_frames
    if target_frames >= total_video_frames:
        # 视频比数据短 → 视频有几帧取几帧，差的补最后一帧
        print(f"  [WARN] 视频帧数({total_video_frames}) < 数据帧数({target_frames})，尾部补帧")

    frames = []
    step = total_video_frames / target_frames if target_frames > 0 else 1.0

    for i in range(target_frames):
        video_frame_idx = min(int(i * step), total_video_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ret, frame = cap.read()
        if not ret:
            # 读不到就用上一帧
            if frames:
                frames.append((i, frames[-1][1].copy()))
            else:
                # 完全没帧，填黑图
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frames.append((i, np.zeros((h, w, 3), dtype=np.uint8)))
        else:
            frames.append((i, frame))

    cap.release()
    return frames


def make_actions(state: np.ndarray) -> np.ndarray:
    """action[i] = state[i+1]，最后一帧指向自己"""
    n = len(state)
    actions = np.zeros_like(state, dtype=np.float32)
    actions[:-1] = state[1:]
    actions[-1] = state[-1]
    return actions


def make_features_with_vision(image_key: str, frame_size: tuple) -> dict:
    """构建 features 字典（含图像）"""
    h, w = frame_size
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
        image_key: {
            "dtype": "image",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
    }


def process_single(
    video_path: Path,
    data_path: Path,
    dataset: LeRobotDataset,
    fps: int = 30,
) -> int:
    """处理一对 video + data，追加到 LeRobot 数据集"""
    import h5py

    # 读关节数据
    with h5py.File(str(data_path), "r") as f:
        state = f["observation.state"][:]

    n_frames = len(state)
    actions = make_actions(state)

    # 抽帧
    frames = extract_frames(video_path, fps, n_frames)

    # 取实际帧尺寸
    if len(frames) > 0:
        _, sample_frame = frames[0]
        h, w = sample_frame.shape[:2]
    else:
        h, w = 480, 640  # fallback

    # 确认 features 包含图像键
    image_key = None
    for key, ft in dataset.features.items():
        if ft.get("dtype") in ("image", "video"):
            image_key = key
            break

    # 逐帧写入
    for i in range(n_frames):
        frame_idx, bgr = frames[i] if i < len(frames) else frames[-1]
        # BGR → RGB（LeRobot 存 RGB）
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        frame_data = {
            "observation.state": state[i].astype(np.float32),
            "action": actions[i].astype(np.float32),
            "task": f"VR demo (video synced, {n_frames} frames)",
        }
        if image_key:
            frame_data[image_key] = rgb

        dataset.add_frame(frame_data)

    dataset.save_episode()
    print(f"  [OK] {n_frames} 帧已写入 episode #{dataset.num_episodes - 1}\n")
    return n_frames


def process_batch(
    video_dir: Path,
    data_dir: Path,
    dataset_name: str,
    root: Path,
    fps: int = 30,
    image_key: str = "observation.images.cam",
):
    """批量处理：视频文件 → 按编号匹配 .ik.h5 → 写入 LeRobot"""

    # 扫描文件
    videos = sorted(video_dir.glob("*.mp4"))
    data_files = sorted(data_dir.glob("episode_*.ik.h5"))

    if not videos:
        print(f"[FAIL] {video_dir} 下没有 mp4 文件")
        return
    if not data_files:
        print(f"[FAIL] {data_dir} 下没有 .ik.h5 文件")
        return

    print(f"[FIND] {len(videos)} 个视频, {len(data_files)} 个 .ik.h5 文件")

    # 尝试按文件名中的数字匹配
    pairs = _match_files(videos, data_files)
    if not pairs:
        print("[FAIL] 无法匹配视频和数据文件")
        print("  确保文件名编号一致，如: demo_009.mp4 ↔ episode_000009.ik.h5")
        return

    print(f"[PAIR] 匹配到 {len(pairs)} 对:\n")
    for v, d in pairs:
        print(f"  {v.name}  ↔  {d.name}")

    # 创建 LeRobot 数据集
    ds_path = root / dataset_name
    if ds_path.exists():
        print(f"\n[DS] 打开已有数据集: {ds_path}")
        dataset = LeRobotDataset(dataset_name, root=str(ds_path))
    else:
        print(f"\n[DS] 创建新数据集: {ds_path}")
        # 先用第一个视频抽一帧来确定图像尺寸
        cap = cv2.VideoCapture(str(pairs[0][0]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
        else:
            h, w = 480, 640
        cap.release()
        frame_size = (h, w)

        dataset = LeRobotDataset.create(
            repo_id=dataset_name,
            fps=fps,
            features=make_features_with_vision(image_key, frame_size),
            root=str(ds_path),
            robot_type="F1",
            use_videos=False,
        )

    # 逐对处理
    total = 0
    for v_path, d_path in pairs:
        n = process_single(v_path, d_path, dataset, fps)
        total += n

    dataset.finalize()
    print(f"{'='*60}")
    print(f"数据集: {ds_path}")
    print(f"总帧数: {total}")
    print(f"总集数: {dataset.num_episodes}")
    print(f"{'='*60}")


def _match_files(videos: list[Path], data_files: list[Path]) -> list[tuple[Path, Path]]:
    """用文件名中的数字匹配视频和数据文件"""
    import re

    pairs = []
    data_map = {}
    for d in data_files:
        # episode_000009.ik.h5 → 9
        nums = re.findall(r'\d+', d.stem.replace('.ik', ''))
        if nums:
            data_map[int(nums[-1])] = d

    for v in videos:
        nums = re.findall(r'\d+', v.stem)
        if nums:
            vid_num = int(nums[-1])
            if vid_num in data_map:
                pairs.append((v, data_map[vid_num]))
                continue
        # 按位置匹配（视频列表和数据列表一一对应）
        # 如果数字匹配失败，回退到位置匹配
    if not pairs and len(videos) == len(data_files):
        pairs = list(zip(videos, data_files))
    elif not pairs and len(videos) == 1 and len(data_files) >= 1:
        # 单个视频 → 所有数据
        pairs = [(videos[0], d) for d in data_files]
        print("[INFO] 单个视频匹配所有数据文件")

    return pairs


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mp4 + IK数据 → 含视觉的 LeRobot 数据集")
    parser.add_argument("--video", type=str, default=None, help="单个 mp4 文件")
    parser.add_argument("--data", type=str, default=None, help="单个 .ik.h5 文件")
    parser.add_argument("--video-dir", type=str, default="./videos", help="视频目录（批量模式）")
    parser.add_argument("--data-dir", type=str, default="./recordings", help="数据目录（批量模式）")
    parser.add_argument("--dataset", type=str, default="f1_vr_vision_v1", help="数据集名称")
    parser.add_argument("--root", type=str, default="./datasets", help="数据集根目录")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-key", type=str, default="observation.images.cam", help="图像特征名")
    parser.add_argument("--clean", action="store_true", help="先清掉旧数据集")
    args = parser.parse_args()

    root = Path(args.root)

    # 清理
    if args.clean:
        ds_path = root / args.dataset
        if ds_path.exists():
            print(f"[CLEAN] 删除: {ds_path}")
            shutil.rmtree(ds_path)

    # 单文件模式
    if args.video and args.data:
        video_path = Path(args.video)
        data_path = Path(args.data)
        if not video_path.exists():
            print(f"[FAIL] 视频不存在: {video_path}")
            sys.exit(1)
        if not data_path.exists():
            print(f"[FAIL] 数据不存在: {data_path}")
            sys.exit(1)

        # 创建临时 LeRobot 数据集
        ds_path = root / args.dataset
        if ds_path.exists():
            print(f"[DS] 打开已有数据集: {ds_path}")
            dataset = LeRobotDataset(args.dataset, root=str(ds_path))
        else:
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            h, w = (frame.shape[:2]) if ret else (480, 640)
            cap.release()
            dataset = LeRobotDataset.create(
                repo_id=args.dataset,
                fps=args.fps,
                features=make_features_with_vision(args.image_key, (h, w)),
                root=str(ds_path),
                robot_type="F1",
                use_videos=False,
            )

        n = process_single(video_path, data_path, dataset, args.fps)
        dataset.finalize()
        print(f"\n[OK] {n} 帧 → {ds_path}")

    # 批量模式
    else:
        process_batch(
            video_dir=Path(args.video_dir),
            data_dir=Path(args.data_dir),
            dataset_name=args.dataset,
            root=root,
            fps=args.fps,
            image_key=args.image_key,
        )
