# openpi (π₀.₅) 数据格式与训练流程 —— 学习要点

> 基于 Physical-Intelligence/openpi 官方仓库提炼
> 你的任务：Quest 3 遥操数据 → LeRobot 格式 → π₀.₅ 后训练

---

## 一、π₀.₅ 模型是什么

| 组件 | 说明 |
|------|------|
| **全称** | π₀.₅ (pi05)，来自 Physical Intelligence |
| **架构** | PaliGemma-3B（视觉编码器）+ Gemma-300M（动作专家） |
| **方法** | Flow Matching（流匹配），不是扩散模型 |
| **参数量** | ~4B |
| **输入** | 图像（可选）+ 本体状态（关节角度等）+ 任务文本 |
| **输出** | 动作向量（关节目标角度） |
| **特点** | 开箱即用的 VLA 基础模型，支持多种机器人平台做后训练 |

---

## 二、数据格式：LeRobot v3.0（核心，必须掌握）

openpi 只吃 LeRobot 格式的数据。你的 Quest 3 数据最终必须转成这个格式。

### 目录结构

```
your_dataset/
├── meta/
│   ├── info.json          ← 数据集元信息（features 定义、帧率、episode 数）
│   ├── episodes.jsonl     ← 每条 episode 的起止帧和任务标签
│   ├── tasks.jsonl        ← 任务文本描述
│   └── stats.json         ← 归一化统计量（训练前必须计算）
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── episode_000000.mp4
        └── ...
```

### Parquet 必须包含的列

| 列名 | 类型 | 形状示例 | 说明 |
|------|------|----------|------|
| `observation.state` | float32[] | [6] 或 [32] | 机器人关节当前角度（本体感知） |
| `action` | float32[] | [6] 或 [32] | 目标关节角度（你要让机器人去哪） |
| `timestamp` | float32 | [1] | 时间戳（秒） |
| `episode_index` | int64 | [1] | 第几条演示 |
| `index` | int64 | [1] | 全局帧编号 |
| `task_index` | int64 | [1] | 任务编号 |
| `next.done` | bool | [1] | 是否最后一帧 |
| `next.reward` | float32 | [1] | 成功=1，否则=0 |

### 可选列

| 列名 | 说明 |
|------|------|
| `observation.images.cam_high` | 主摄像头图像（dict，含 path 或 bytes） |
| `observation.images.cam_wrist` | 腕部摄像头 |
| `observation.hand_pose` | VR 手部位姿（你自己加的，F1 场景可能需要） |
| `observation.gripper_cmd` | 夹爪指令（你自己加的） |

### info.json 必须包含 features 字典

```json
{
  "codebase_version": "v3.0",
  "robot_type": "F1",
  "fps": 30,
  "total_episodes": 10,
  "total_frames": 3000,
  "features": {
    "observation.state": {"dtype": "float32", "shape": [6]},
    "action":             {"dtype": "float32", "shape": [6]},
    "timestamp":          {"dtype": "float32", "shape": [1]},
    "episode_index":      {"dtype": "int64",   "shape": [1]},
    "index":              {"dtype": "int64",   "shape": [1]},
    "task_index":         {"dtype": "int64",   "shape": [1]},
    "next.done":          {"dtype": "bool",    "shape": [1]},
    "next.reward":        {"dtype": "float32", "shape": [1]}
  }
}
```

> ⚠️ features 字典是 LeRobot 库 `load_info()` 函数强制要求的，缺少会直接报 KeyError。

---

## 三、数据管线：四步转换

openpi 内部对数据做四次转换。你不需要自己实现，但必须理解每一步在干什么：

```
原始 Parquet
   │
   ▼
RepackTransform    ← 列名映射（把不同机器人的字段名统一成 openpi 内部格式）
   │
   ▼
DataTransform      ← 计算 delta action / 相对动作（可选）
   │
   ▼
ModelTransform     ← 图像resize、tokenize、padding 到固定维度
   │
   ▼
Normalization      ← 用预计算的统计量做归一化（减均值除标准差 或 缩放到分位数范围）
```

### 3.1 RepackTransform（列名映射）

不同机器人数据集的列名不一样，RepackTransform 把它们统一：
- 你的 `observation.state` → openpi 内部 `state`
- 你的 `action` → openpi 内部 `actions`
- 你的 `observation.images.xxx` → openpi 内部 `image`

**要点：你的 Parquet 列名必须和 openpi 配置文件里预期的名字一致**，否则训练时报 KeyError。

### 3.2 DataTransform（动作转换）

核心概念：**绝对动作 vs 相对动作**

- **绝对动作**：`action` 是关节的绝对目标角度。例如「关节1转到 0.5 rad」。
- **相对动作（delta）**：`action` 是关节角度变化量。例如「关节1在当前基础上 +0.02 rad」。

π₀.₅ 默认用绝对动作。如果你的数据是相对动作，需要在配置里设置 `use_delta_action: true`。

**F1 场景建议：绝对动作**，简单直接，和 VR 遥操一致。

### 3.3 ModelTransform（模型预处理）

- 图像 resize 到固定分辨率
- 状态和动作向量 padding 到 `max_state_dim=32` / `max_action_dim=32`
- F1 只有 6 个关节，会被自动 padding 到 32 维（前面是有效值，后面补 0）

### 3.4 Normalization（归一化）

π₀.₅ 默认使用**分位数归一化（Quantile Normalization）**：

| 特征类型 | 归一化方式 | 使用的统计量 |
|----------|------------|-------------|
| STATE（状态） | QUANTILES | q01, q99（1%和99%分位数） |
| ACTION（动作） | QUANTILES | q01, q99 |
| VISUAL（图像） | IDENTITY | 不做归一化 |

公式：`normalized = (x - q01) / (q99 - q01)`

**训练前必须计算归一化统计量**，运行：
```bash
python scripts/compute_norm_stats.py --repo-id your_dataset_name
```
结果存入 `meta/stats.json`，训练时自动加载。

---

## 四、训练配置（关键参数）

### 最小训练命令

```bash
lerobot-train \
    --dataset.repo_id=your_org/your_dataset \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --output_dir=./outputs/pi05_f1_training \
    --batch_size=32
```

### 关键参数解释

| 参数 | 含义 | F1 建议值 |
|------|------|-----------|
| `policy.type` | 模型类型 | `pi05` |
| `policy.pretrained_path` | 预训练权重 | `lerobot/pi05_base` |
| `action_dim` | 动作维度 | F1 有几个关节就填几（6~7） |
| `max_state_dim` | 状态最大维度 | 32（自动 padding） |
| `max_action_dim` | 动作最大维度 | 32（自动 padding） |
| `use_relative_actions` | 是否用相对动作 | `false`（F1 用绝对） |
| `normalization_mapping` | 归一化方式 | 默认 QUANTILES |
| `batch_size` | 批次大小 | 32（按 GPU 内存调整） |

### LoRA 微调（低显存方案）

如果 GPU 显存不够，可以用 LoRA：
- 显存需求大幅降低
- 必须关闭 EMA：`ema_decay=None`
- 适合小数据集微调

---

## 五、从 Quest 3 到 openpi 的完整数据流

```
你的手
  │
  ▼
Quest 3 (Hand Tracking Streamer)
  │  输出：21个手部关节点 3D 坐标 + 手腕 6D 位姿
  │  传输：UDP → PC (局域网，不需要翻墙)
  │
  ▼
PC 端 Python 脚本 (04_vr_hand_record.py)
  │  功能：接收 UDP → 组装手部数据 → 暂存为 HDF5
  │  输出：episode_000000.h5
  │
  ▼
IK 解算器（待导师提供 F1 URDF）
  │  输入：手部末端位姿
  │  输出：F1 6个关节的目标角度
  │  核心问题：手部位姿 → 机械臂末端位姿的映射关系
  │
  ▼
数据录制 (02_record_episode.py 集成版)
  │  输入：机器人关节状态 + IK 动作 + VR 手部数据
  │  输出：episode_XXXXXX.h5（含完整的 observation.state + action）
  │
  ▼
格式转换 (03_convert_to_lerobot.py)
  │  HDF5 → LeRobot v3.0（Parquet + info.json + episodes.jsonl）
  │
  ▼
计算归一化统计量
  │  python scripts/compute_norm_stats.py
  │
  ▼
π₀.₅ 后训练
  │  lerobot-train --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base
  │
  ▼
部署到 F1 机器人
```

---

## 六、你现在已经跑通的部分

| 步骤 | 状态 |
|------|------|
| Quest 3 + ADB 连接 | ✅ |
| Hand Tracking Streamer APK 安装 | ✅ |
| PC 环境（Python 3.10 + 全部依赖） | ✅ |
| HDF5 录制脚本 | ✅ |
| LeRobot v3.0 转换脚本（含 features 字典） | ✅ |
| LeRobot 格式验证 | ✅ |
| Hand Tracking SDK（PC 端 UDP 接收） | ✅ |
| VR 手部数据 → HDF5 录制脚本 | ✅ |
| **Quest 3 连 WiFi → PC 接收数据** | ❌ 卡在这一步 |

---

## 七、当前阻塞 & 下一步

**唯一阻塞点**：Quest 3 没连 WiFi，Hand Tracking Streamer 的 UDP 数据发不到 PC。

**你需要做的**：把 Quest 3 连上公司 WiFi（不需要翻墙，只要和 PC 在同一个局域网）。

连上 WiFi 后：
1. 戴上 Quest 3 → 打开 Hand Tracking Streamer
2. 配置 IP: `<PC 的局域网 IP>`，Port: `9000`，Protocol: UDP
3. 我这边启动 PC 接收 → 实时看到手部坐标 → 录数据 → 转 LeRobot → 完成
