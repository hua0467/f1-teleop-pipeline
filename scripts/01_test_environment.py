"""
F1 遥操数据采集管线 —— 环境验证脚本
跑一遍，确认所有依赖都装好了
"""

import sys

print("=" * 60)
print("  F1 Teleop Pipeline — 环境检查")
print("=" * 60)

# --- Python 版本 ---
print(f"\nPython: {sys.version}")

# --- 核心科学计算 ---
try:
    import numpy as np
    print(f"numpy:   {np.__version__} [OK]")
except ImportError as e:
    print(f"numpy:   NOT INSTALLED ({e})")

try:
    import scipy
    print(f"scipy:   {scipy.__version__} [OK]")
except ImportError as e:
    print(f"scipy:   NOT INSTALLED ({e})")

# --- 数据处理 ---
try:
    import h5py
    print(f"h5py:    {h5py.__version__} [OK]")
except ImportError as e:
    print(f"h5py:    NOT INSTALLED ({e})")

try:
    import pyarrow as pa
    print(f"pyarrow: {pa.__version__} [OK]")
except ImportError as e:
    print(f"pyarrow: NOT INSTALLED ({e})")

# --- 图像 ---
try:
    import cv2
    print(f"opencv:  {cv2.__version__} [OK]")
except ImportError as e:
    print(f"opencv:  NOT INSTALLED ({e})")

# --- 可视化 ---
try:
    import matplotlib
    print(f"matplotlib: {matplotlib.__version__} [OK]")
except ImportError as e:
    print(f"matplotlib: NOT INSTALLED ({e})")

# --- LeRobot ---
try:
    import lerobot
    print(f"lerobot: {lerobot.__version__} [OK]")
except ImportError as e:
    print(f"lerobot: NOT INSTALLED (but we can manually create datasets)")

# --- PyTorch ---
try:
    import torch
    print(f"torch:   {torch.__version__} [OK]")
    if torch.cuda.is_available():
        print(f"         CUDA: {torch.version.cuda} (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print(f"         CUDA: not available (integrated GPU, this is normal)")
except ImportError as e:
    print(f"torch:   NOT INSTALLED (not needed on this machine)")

print("\n" + "=" * 60)
print("  Environment check complete.")
print("=" * 60)
