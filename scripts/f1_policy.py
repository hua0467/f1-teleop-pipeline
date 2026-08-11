"""F1 双臂人形机器人 —— OpenPI DataConfig 插件。

参照 OpenPI 官方的 aloha_policy.py 和 libero_policy.py 的模式编写。
接入 OpenPI 时，将此文件放到 src/openpi/policies/f1_policy.py 即可。

F1 机器人规格：
  - 17 DOF：3 躯干 + 7 右臂 + 7 左臂
  - 无 gripper（待硬件团队定义信号格式）
  - 数据格式：绝对关节角（弧度制），需转 delta
  - 暂无相机数据（三路图像全部填零）
"""

import dataclasses
import numpy as np

# ============================================================
# 兼容处理：在 OpenPI 内部被 import 时，下面这两个 import 可用。
# 在 F1 管线本地运行时，这两个 import 不存在，仅用于类型标注。
# ============================================================
try:
    from openpi import transforms
    from openpi.models import model as _model
except ImportError:
    transforms = None
    _model = None


# ---- F1 关节名（按 IK 求解器的顺序） ----
F1_JOINT_NAMES = [
    # 躯干 3 轴
    "lift_joint",
    "waist1_joint",
    "waist2_joint",
    # 右臂 7 轴
    "J1_right_joint",
    "J2_right_joint",
    "J3_right_joint",
    "J4_right_joint",
    "J5_right_joint",
    "J6_right_joint",
    "J7_right_joint",
    # 左臂 7 轴
    "J1_left_joint",
    "J2_left_joint",
    "J3_left_joint",
    "J4_left_joint",
    "J5_left_joint",
    "J6_left_joint",
    "J7_left_joint",
]

F1_ACTION_DIM = 17  # 当前无 gripper，全部关节一起做 delta

# ---- 三路相机：暂无 ----
F1_MISSING_CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def make_f1_example() -> dict:
    """创建一个随机输入样本，用于调试 F1 policy。"""
    return {
        "state": np.random.randn(F1_ACTION_DIM).astype(np.float32),
        "actions": np.random.randn(10, F1_ACTION_DIM).astype(np.float32),
        "prompt": "pick up the bottle",
    }


# ============================================================
# F1Inputs  ——  数据输入变换（训练 + 推理共用）
# ============================================================
@dataclasses.dataclass(frozen=True)
class F1Inputs:
    """F1 双臂机器人的输入变换。

    把管线内部的标准键值转换成模型期望的 Observation 格式。

    当前状态：
      - 无相机 → 三路图像全部填零，image_mask=False
      - state 直通（17 维绝对关节角，弧度制）
      - actions 和 prompt 直通（如果有的话）

    等相机接入后：
      - 把 EGO 图像数据填充到 base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb
      - 对应 image_mask 设为 True
    """

    def __call__(self, data: dict) -> dict:
        # ---- 图像：暂无相机，用零填充 ----
        # 拿 state 的 shape 来推 batch 维度
        state = np.asarray(data["state"])

        # 生成一张全零的"假图像"（224×224 是模型分辨率，三通道 RGB）
        dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)

        images = {
            "base_0_rgb": dummy_image,
            "left_wrist_0_rgb": dummy_image.copy(),
            "right_wrist_0_rgb": dummy_image.copy(),
        }
        image_masks = {
            "base_0_rgb": np.False_,
            "left_wrist_0_rgb": np.False_,
            "right_wrist_0_rgb": np.False_,
        }

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # ---- Actions：训练时有 ----
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        # ---- Prompt：语言指令 ----
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


# ============================================================
# F1Outputs  ——  模型输出变换（仅推理）
# ============================================================
@dataclasses.dataclass(frozen=True)
class F1Outputs:
    """F1 双臂机器人的输出变换。

    模型内部 action_dim=32（π₀.₅ 默认值），但 F1 只有 17 维。
    PadStatesAndActions 在训练时自动补零，在推理时模型输出也是 32 维。
    这里截取前 17 维，把模型输出还原成 F1 的关节角空间。

    等 gripper 接入后，截取维度可能需要调整为 18 或 19。
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        # 只取前 17 维（F1 真实 DOF），丢弃 padding 部分
        return {"actions": actions[..., :F1_ACTION_DIM]}


# ============================================================
# LeRobotF1DataConfig  ——  DataConfigFactory 子类
# ============================================================
#
# 说明：此 class 依赖 OpenPI 运行时的 import 路径：
#   from openpi import transforms as _transforms
#   from openpi.models import model as _model
#   from openpi.policies import f1_policy
#   from openpi.training.config import DataConfigFactory, AssetsConfig, DataConfig, ModelTransformFactory
#
# 使用时，把下面这个 class 粘贴到 OpenPI 的 config.py 中（放在 LeRobotLiberoDataConfig
# 旁边即可），然后在 _CONFIGS 列表里注册对应的 TrainConfig。
#
# ============================================================

# 以下是完整的 DataConfig 代码，供粘贴到 OpenPI config.py 时参考：

F1_DATACONFIG_CLASS_CODE = r'''
@dataclasses.dataclass(frozen=True)
class LeRobotF1DataConfig(DataConfigFactory):
    """F1 双臂人形机器人 —— DataConfig。

    参照 LeRobotLiberoDataConfig 的结构编写。
    F1 特点：
      - 17 DOF（3 躯干 + 7 右臂 + 7 左臂），无 gripper
      - 数据存的是绝对关节角（弧度制），需要 DeltaActions 转增量
      - 暂无相机数据（三路图像由 F1Inputs 填零）
      - 从 LeRobot 数据集的 task 字段提取语言指令
    """

    # 是否将绝对关节角转为增量（F1 数据是绝对的，需要开）
    use_delta_joint_actions: bool = True

    # 默认 prompt —— 如果数据集没有 task 字段就用这个
    default_prompt: str | None = None

    # Repack：把 LeRobot 数据集里的列名映射到管线内部标准名
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        # 数据集里的 observation.state → 管线内的 state
                        "state": "observation.state",
                        # 数据集里的 action → 管线内的 actions
                        "actions": "action",
                        # task_index 原样保留（PromptFromLeRobotTask 要用）
                        "task_index": "task_index",
                    }
                )
            ]
        )
    )

    # Action 序列的键名（LeRobot 数据集里动作列叫 "action"）
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: _model.BaseModelConfig,
    ) -> DataConfig:
        # ---- Data transforms：F1 专用的输入/输出变换 ----
        data_transforms = _transforms.Group(
            inputs=[f1_policy.F1Inputs()],
            outputs=[f1_policy.F1Outputs()],
        )

        # ---- Delta 变换：绝对关节角 → 增量 ----
        if self.use_delta_joint_actions:
            # F1 目前没有 gripper，全部 17 个关节都转 delta
            delta_action_mask = _transforms.make_bool_mask(f1_policy.F1_ACTION_DIM)
            # 等 gripper 接入后，改为类似：
            #   delta_action_mask = _transforms.make_bool_mask(16, -1)
            #   前 16 个关节转 delta，最后 1 个（gripper）保持绝对值
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # ---- Model transforms：自动生成（Resize + Tokenize + Pad） ----
        model_transforms = ModelTransformFactory(
            default_prompt=self.default_prompt
        )(model_config)

        # ---- 拼装 ----
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
'''


# ============================================================
# TrainConfig 注册条目（粘贴到 OpenPI config.py 的 _CONFIGS 列表）
# ============================================================

F1_TRAINCONFIG_ENTRY = r'''
    # ---- F1 双臂人形机器人 ----
    TrainConfig(
        name="pi05_f1_vr",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
        ),
        data=LeRobotF1DataConfig(
            repo_id="qihang/f1_vr_v1",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=30000,
        batch_size=32,
    ),
'''


# ============================================================
# 打印代码（方便查看/复制）
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("F1 Policy 示例输入:")
    print("=" * 60)
    example = make_f1_example()
    for k, v in example.items():
        print(f"  {k}: shape={v.shape if hasattr(v, 'shape') else len(v)}, dtype={v.dtype}")
    print()

    print("=" * 60)
    print("F1Inputs / F1Outputs 源码已定义在本文件顶部")
    print("=" * 60)
    print()

    print("=" * 60)
    print("LeRobotF1DataConfig  ——  粘贴到 OpenPI config.py")
    print("=" * 60)
    print(F1_DATACONFIG_CLASS_CODE)
    print()

    print("=" * 60)
    print("TrainConfig  ——  粘贴到 OpenPI config.py 的 _CONFIGS 列表")
    print("=" * 60)
    print(F1_TRAINCONFIG_ENTRY)
