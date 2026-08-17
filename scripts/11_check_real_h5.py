"""
11_check_real_h5.py —— 真机遥操 HDF5 质检（PC 端）

对象: 机器人上 f1_record_ros2.py 录制的 session_*.h5（14 DOF + 机载 JPEG）
      —— 不是 VR 线（Quest 3）的 episode_*.h5，那种文件本脚本会直接判 FAIL。

用法:
  python scripts/11_check_real_h5.py recordings/session_xxx.h5
  python scripts/11_check_real_h5.py recordings/session_*.h5    # 批量

检查项（对照真机手册 9.x 数据约定）:
  1. 结构: observation.state / action / timestamp 存在，attrs 齐全
  2. 维度: (n, 14) float32，关节名 = 右臂 7 + 左臂 7
  3. action shift: action[i] == state[i+1]（末帧 = 自身）
  4. NaN: 总帧数 / 按关节列分布 / 最长连续段 —— 单臂没收到指令的标记
  5. 数值: ±pi 越界计数；相邻帧跳变 >30 度的关节（构型失控指纹）
  6. 帧率: attrs fps vs 实测；>1s 的无帧段（SSH 断连 / APP 停发指纹）
  7. 画面: JPEG 覆盖率、抽样解码、尺寸一致性
  8. 双臂覆盖: robot_id 分布（0=双臂 1=左 2=右）

输出: 逐项 [OK]/[WARN]/[FAIL] + 总分判定（可进数据集 / 需清洗 / 废）
退出码: 0=可进数据集, 1=需清洗或废
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REQUIRED_KEYS = ["observation.state", "action", "timestamp"]
REQUIRED_ATTRS = ["total_frames", "joint_names", "joint_units", "source"]
N_JOINTS = 14
JUMP_RAD = np.deg2rad(30.0)   # 相邻帧 30 度跳变 = 异常
GAP_SEC = 1.0                 # 帧间隔超过 1 秒 = 断流

_WARN = []   # (检查项, 详情)
_FAIL = []


def w(name, detail):
    _WARN.append((name, detail))


def fail(name, detail):
    _FAIL.append((name, detail))


def nan_rows(mat):
    """每帧是否含 NaN 的 bool 数组"""
    return np.isnan(mat).any(axis=1)


def check_h5(path: Path, max_scan: int | None = None) -> bool:
    """返回 True = 可进数据集。max_scan 只影响跳变扫描的帧数上限，不影响判定。"""
    _WARN.clear()
    _FAIL.clear()
    print(f"\n===== {path.name} =====")

    try:
        import h5py
    except ImportError:
        fail("环境", "PC 没装 h5py（pip install h5py）")
        _print_report()
        return False

    if not path.exists():
        fail("文件", "不存在")
        _print_report()
        return False

    with h5py.File(str(path), "r") as f:
        # ---- 1. 结构 ----
        for k in REQUIRED_KEYS:
            if k not in f:
                fail("结构", f"缺数据集 {k} —— 这不像真机 f1_record_ros2.py 的产物")
        for a in REQUIRED_ATTRS:
            if a not in f.attrs:
                fail("结构", f"attrs 缺 {a}")
        if _FAIL:
            _print_report()
            return False

        attrs = dict(f.attrs)
        if attrs.get("source") != "f1_record_ros2":
            fail("结构", f"attrs.source={attrs.get('source')!r}，不是真机录制脚本的产物")
            _print_report()
            return False

        state = f["observation.state"][:]
        action = f["action"][:]
        ts = f["timestamp"][:]
        n = int(state.shape[0])
        print(f"[INFO] attrs.total_frames={attrs['total_frames']}  实际 {n} 帧")

        if n == 0:
            fail("维度", "0 帧")
            _print_report()
            return False
        if attrs["total_frames"] != n:
            w("结构", f"attrs.total_frames({attrs['total_frames']}) != 实际帧数({n})")

        # ---- 2. 维度 ----
        if state.ndim != 2 or state.shape[1] != N_JOINTS:
            fail("维度", f"observation.state shape={state.shape}，应为 (n, {N_JOINTS})")
            _print_report()
            return False
        if action.shape != state.shape:
            fail("维度", f"action shape={action.shape} != state shape={state.shape}")
            _print_report()
            return False
        try:
            names = json.loads(attrs["joint_names"])
            if len(names) != N_JOINTS:
                w("维度", f"attrs.joint_names 有 {len(names)} 个，应为 {N_JOINTS}")
        except (json.JSONDecodeError, TypeError):
            w("结构", "attrs.joint_names 不是合法 JSON")
        if attrs.get("joint_units") != "rad":
            w("结构", f"joint_units={attrs.get('joint_units')!r}，约定是 rad")
        print(f"[OK] 维度: state/action ({n}, {N_JOINTS}) float32")

        # ---- 3. action shift ----
        if n > 1:
            a = action[:-1]
            s = state[1:]
            eq = np.allclose(a, s, rtol=0, atol=1e-6, equal_nan=True)
            if not eq:
                bad = int((~(np.isclose(a, s, rtol=0, atol=1e-6) | (np.isnan(a) & np.isnan(s)))).any(axis=1).sum())
                fail("action", f"action[i] != state[i+1] 的有 {bad} 帧 —— 违反转换约定")
            else:
                print("[OK] action shift: action[i] == state[i+1]（末帧=自身）")

        # ---- 4. NaN ----
        nr = nan_rows(state)
        n_nan = int(nr.sum())
        if n_nan == 0:
            print("[OK] NaN: 无")
        else:
            nan_cols = np.isnan(state).sum(axis=0)
            worst = int(np.argmax(nan_cols))
            # 最长连续 NaN 段
            idx = np.flatnonzero(nr)
            if len(idx) == 0:
                run = 0
            else:
                splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                run = max(len(s) for s in splits)
            detail = (f"{n_nan}/{n} 帧 ({100.0 * n_nan / n:.1f}%)，"
                      f"最重关节 #{worst} 有 {int(nan_cols[worst])} 帧，最长连续 {run} 帧")
            if n_nan / n > 0.05:
                w("NaN", detail + " —— 建议剔除或插值后再进数据集")
            else:
                w("NaN", detail + "（占比小，训练前局部清洗即可）")

        # ---- 5. 数值范围与跳变 ----
        out_of_range = int((np.abs(np.nan_to_num(state)) > np.pi).any(axis=1).sum())
        if out_of_range:
            w("范围", f"{out_of_range} 帧关节角超出 [-pi, pi]（可能是反馈字段混入或单位错误）")
        scan_n = n if max_scan is None else min(n, max_scan)
        if n > 1 and scan_n > 1:
            d = np.abs(np.diff(state[:scan_n], axis=0))
            big = d > JUMP_RAD
            n_jump = int(big.sum())
            if n_jump:
                per_joint = big.sum(axis=0)
                wj = int(np.argmax(per_joint))
                w("跳变", f"{n_jump} 次相邻帧 >30 度跳变，最重关节 #{wj}（{int(per_joint[wj])} 次）"
                          f" —— 构型失控/跟踪丢失的指纹，定位后该段建议作废")
            else:
                print(f"[OK] 跳变: 前 {scan_n} 帧无 >30 度相邻帧跳变")
        else:
            print("[OK] 跳变: 仅 1 帧，跳过")

        # ---- 6. 帧率与断流 ----
        if n > 1:
            dt = np.diff(ts)
            dt = dt[dt >= 0]   # 负间隔 = 时间戳回绕，另行告警
            neg = int((np.diff(ts) < 0).sum())
            if neg:
                w("时间戳", f"{neg} 处时间戳回绕（非单调递增）")
            if len(dt) == 0:
                fail("时间戳", "全为负间隔，无法评估帧率")
            else:
                real_fps = 1.0 / float(np.median(dt)) if np.median(dt) > 0 else 0.0
                att_fps = float(attrs.get("fps", -1))
                print(f"[INFO] 帧率: attrs={att_fps:.1f}Hz  实测(中位间隔)={real_fps:.1f}Hz")
                gaps = int((dt > GAP_SEC).sum())
                if gaps:
                    w("断流", f"{gaps} 处帧间隔 >1s（最大 {dt.max():.1f}s）"
                              f" —— SSH 断连/APP 停发的指纹，训练前需切段")
                else:
                    print(f"[OK] 断流: 无 >1s 间隔（最大 {dt.max():.3f}s）")
        else:
            print("[OK] 帧率: 仅 1 帧，跳过")

        # ---- 7. 机载画面 ----
        cam_key = "observation.images.cam"
        if cam_key not in f:
            w("画面", "无机载画面（录制时相机不可用或没加 --cam）—— π0.5 是 VLA，无视觉数据训练大打折扣")
        else:
            try:
                cam = f[cam_key]
                m = cam.shape[0]
                cover = 100.0 * m / n
                print(f"[INFO] 画面: {m}/{n} 帧 ({cover:.1f}%)，attrs 尺寸 "
                      f"{attrs.get('cam_width', '?')}x{attrs.get('cam_height', '?')}")
                if cover < 90:
                    w("画面", f"覆盖率 {cover:.1f}% < 90%（相机启动滞后或中途掉流）")
                # 抽样解码 3 帧
                import cv2
                idxs = sorted(set([0, n // 2, min(n - 1, m - 1)]))
                shapes = set()
                for i in idxs:
                    if i < m:
                        img = cv2.imdecode(np.frombuffer(bytes(cam[i]), np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            fail("画面", f"第 {i} 帧 JPEG 解不出来")
                        else:
                            shapes.add(img.shape[:2])
                if shapes and len(shapes) == 1:
                    print(f"[OK] 画面: 抽样 {len(idxs)} 帧解码成功，尺寸一致 {next(iter(shapes))}")
                elif shapes:
                    w("画面", f"抽样帧尺寸不一致: {shapes}")
            except ImportError:
                w("画面", "PC 没装 opencv-python，跳过解码验证")

        # ---- 8. robot_id 分布 ----
        if "robot_id" in f:
            rid = f["robot_id"][:].astype(int)
            uniq, cnt = np.unique(rid, return_counts=True)
            dist = ", ".join(f"id={u}:{c}" for u, c in zip(uniq, cnt))
            print(f"[INFO] robot_id 分布: {dist}  (0=双臂 1=左 2=右)")
            if 0 not in uniq:
                w("覆盖", "全程没有 robot_id=0 的双臂指令帧")

    # ---- 汇总判定 ----
    _print_report()
    if _FAIL:
        print("[VERDICT] 废 —— 先解决 FAIL 项")
        return False
    if _WARN:
        print("[VERDICT] 需清洗 —— WARN 项处理后可用")
        return False
    print("[VERDICT] 可进数据集")
    return True


def _print_report():
    for name, detail in _FAIL:
        print(f"[FAIL] {name}: {detail}")
    for name, detail in _WARN:
        print(f"[WARN] {name}: {detail}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="真机遥操 HDF5 质检")
    parser.add_argument("inputs", nargs="+", help="session_*.h5 路径（可多个）")
    parser.add_argument("--max-scan", type=int, default=5000,
                        help="跳变扫描的帧数上限（默认 5000，大数据集防慢）")
    args = parser.parse_args()

    all_ok = True
    for p in args.inputs:
        ok = check_h5(Path(p), max_scan=args.max_scan)
        all_ok = all_ok and ok
    sys.exit(0 if all_ok else 1)
