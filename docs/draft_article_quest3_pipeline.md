# [草稿-待整合] Quest 3 VR 遥操数采管线：从零搭建全记录

> ⚠️ 本文为草稿，不发表。等端到端跑通后，与后续内容整合为一篇完整文章。
> 当前进度：环境搭建 100% | 管线脚本 100% | Quest 3 APK 100% | WiFi 数据流 0%

---

## 背景

导师任务：基于 Meta Quest 3 实现 VR 遥操作数据采集，制作 LeRobot 格式数据集，对接 π₀.₅ (openpi) VLA 模型做后训练。机器人平台为 F1 轮式单臂 + 夹爪。

## 已完成工作（8月7日）

### 1. 环境搭建
- Conda Python 3.10 (f1-teleop) + LeRobot + hand-tracking-sdk + h5py + pyarrow + PyTorch
- ADB 1.0.41 + SideQuest
- Meta Quest Developer Hub

### 2. openpi 数据格式学习
- LeRobot v3.0: 8 个必须 Parquet 列 + features 字典 + info.json + episodes.jsonl
- 四步管线: RepackTransform → DataTransform → ModelTransform → Normalization
- 分位数归一化: q01/q99 替代 Z-Score
- 训练命令: lerobot-train --policy.type=pi05

### 3. 管线脚本
- 02_record_episode.py: 模拟数据 → HDF5 录制
- 03_convert_to_lerobot.py: HDF5 → LeRobot v3.0
- 04_vr_hand_record.py: Quest 3 UDP → 实时录制

### 4. Quest 3 配置
- 开发者模式 + ADB 连接
- hand-tracking-streamer APK 侧载成功

### 5. 踩坑记录
- Windows GBK vs emoji 编码
- Quest 3 原装充电线不能传数据
- SideQuest 被墙 → Clash 代理恢复
- "Install" vs "Sideload now!" 按钮文案差异

## 待完成
- Quest 3 WiFi 连接 → UDP 数据流验证
- 端到端：手部追踪 → HDF5 → LeRobot → openpi 格式合格
- 向导师获取 F1 URDF + 控制协议

## 未来整合方向
- [ ] WiFi + UDP 联调经验
- [ ] IK 解算方案
- [ ] 真实 F1 机器人遥操演示
- [ ] 完整数据集发布到 Hugging Face
