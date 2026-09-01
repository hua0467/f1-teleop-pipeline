"""F1 双臂 IK 求解器

从 VR 手部追踪拿到的末端位姿（7个数 xyz+qxyzw），
反解出 17 个关节各自转多少度。

纯 numpy+scipy，没碰 pybullet（Windows 编译报错）和 pinocchio（没 conda 包）。
网上那些 IK 教程基本都是六轴机械臂的，双机械臂 + 躯干的链条得自己写。

其实核心就干了三件事：
  1. 读 URDF XML → 把 parent/child 关系抽出来组成运动学链
  2. 沿着链条用变换矩阵正算末端在哪（FK）
  3. 用 scipy L-BFGS-B 反向优化，让 FK 的结果逼近目标位姿（IK）

链的结构大概是这样（右臂）：
  base_link → lift → waist1 → waist2 → J1_R → ... → J7_R
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import xml.etree.ElementTree as ET
from pathlib import Path


class F1Kinematics:
    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.joints = {}
        self.links = {}
        self._parse_urdf()

        # Build chains for left and right arms
        self.right_arm_chain = self._get_chain("base_link", "J7_right_link")
        self.left_arm_chain = self._get_chain("base_link", "J7_left_link")
        self.torso_chain = self._get_chain("base_link", "waist2_link")

        # Get controllable joints only (revolute + prismatic, skip fixed/continuous wheels)
        self.right_arm_joints = [j for j in self.right_arm_chain
                                  if self.joints[j]['type'] in ('revolute', 'prismatic')]
        self.left_arm_joints = [j for j in self.left_arm_chain
                                 if self.joints[j]['type'] in ('revolute', 'prismatic')]
        self.torso_joints = [j for j in self.torso_chain
                              if self.joints[j]['type'] in ('revolute', 'prismatic')]

        self.n_right = len(self.right_arm_joints)
        self.n_left = len(self.left_arm_joints)
        self.n_torso = len(self.torso_joints)

        print(f"[IK] F1 Kinematics loaded:")
        print(f"  Torso:  {self.n_torso} joints -> {self.torso_joints}")
        print(f"  Right:  {self.n_right} joints -> {self.right_arm_joints}")
        print(f"  Left:   {self.n_left} joints -> {self.left_arm_joints}")
        print(f"  Total:  {self.n_torso + self.n_right + self.n_left} DOF")

    def _parse_urdf(self):
        """从 URDF XML 里把 joint/link 信息全抽出来存字典里"""
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        for joint in root.findall('joint'):
            name = joint.get('name')
            jtype = joint.get('type')
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')

            origin = joint.find('origin')
            xyz = np.array([float(x) for x in origin.get('xyz', '0 0 0').split()], dtype=np.float64)
            rpy = np.array([float(x) for x in origin.get('rpy', '0 0 0').split()], dtype=np.float64)

            axis = joint.find('axis')
            axis_xyz = np.array([float(x) for x in axis.get('xyz', '0 0 1').split()], dtype=np.float64)

            limit = joint.find('limit')
            lower = float(limit.get('lower', 0)) if limit is not None else -np.pi
            upper = float(limit.get('upper', 0)) if limit is not None else np.pi

            self.joints[name] = {
                'type': jtype,
                'parent': parent,
                'child': child,
                'origin_xyz': xyz,
                'origin_rpy': rpy,
                'axis': axis_xyz / (np.linalg.norm(axis_xyz) or 1.0),
                'lower': lower,
                'upper': upper,
            }
            self.links[child] = name

    def _get_chain(self, start_link: str, end_link: str) -> list:
        """从 end_link 一路往上找到 start_link，返回经过的关节列表"""
        chain = []
        current = end_link
        while current != start_link:
            jname = self.links.get(current)
            if jname is None:
                raise ValueError(f"Cannot find parent joint for link '{current}'")
            chain.insert(0, jname)
            current = self.joints[jname]['parent']
        return chain

    def _joint_transform(self, joint_name: str, angle: float) -> np.ndarray:
        """单个关节在给定角度下的 4x4 变换矩阵"""
        j = self.joints[joint_name]
        T = np.eye(4)

        # Origin translation + rotation
        T[:3, :3] = R.from_euler('xyz', j['origin_rpy']).as_matrix()
        T[:3, 3] = j['origin_xyz']

        # Joint motion
        if j['type'] in ('revolute', 'continuous'):
            R_j = R.from_rotvec(j['axis'] * angle).as_matrix()
            T_motion = np.eye(4)
            T_motion[:3, :3] = R_j
            T = T @ T_motion
        elif j['type'] == 'prismatic':
            T[:3, 3] += j['axis'] * angle

        return T

    def forward_kinematics(self, joint_angles: dict) -> dict:
        """给所有关节角度 → 返回左右手末端 4x4 位姿

        joint_angles: joint_name -> 角度(弧度)，不传的就默认 0
        """
        # Right arm FK
        T_right = np.eye(4)
        for jname in self.right_arm_chain:
            angle = joint_angles.get(jname, 0.0)
            T_right = T_right @ self._joint_transform(jname, angle)

        # Left arm FK
        T_left = np.eye(4)
        for jname in self.left_arm_chain:
            angle = joint_angles.get(jname, 0.0)
            T_left = T_left @ self._joint_transform(jname, angle)

        return {
            'right_ee': T_right,
            'left_ee': T_left,
        }

    def _chain_fk(self, chain: list, angles: np.ndarray) -> np.ndarray:
        """给一条链的所有关节角 → 返回末端的 4x4 位姿"""
        T = np.eye(4)
        for jname, angle in zip(chain, angles):
            T = T @ self._joint_transform(jname, angle)
        return T

    def _pose_error(self, T_current: np.ndarray, T_target: np.ndarray) -> float:
        """位置 + 朝向的综合误差，IK 优化就是最小化这个值"""
        pos_err = np.linalg.norm(T_current[:3, 3] - T_target[:3, 3])

        # Orientation error: angle of relative rotation
        R_current = T_current[:3, :3]
        R_target = T_target[:3, :3]
        R_diff = R_target @ R_current.T
        rot_err = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))

        # Weight: 1m ~= 1 rad
        return pos_err + 0.5 * rot_err

    def solve_ik(self, target_pos, target_quat, side='right', initial_guess=None):
        """IK 主函数：给定末端目标位姿 → 返回该侧手臂的关节角

        target_pos: [x, y, z] 机器人基座坐标系下的目标位置
        target_quat: [x, y, z, w] 目标姿态四元数 (scipy 约定 xyzw)
        side: 'right' 或 'left'
        initial_guess: 上一帧的关节角（做初始值用，这样收敛快很多）

        返回该侧手臂的所有关节角度（弧度）
        """
        chain = self.right_arm_joints if side == 'right' else self.left_arm_joints
        n = len(chain)

        # Build target 4x4 matrix
        T_target = np.eye(4)
        T_target[:3, :3] = R.from_quat(target_quat).as_matrix()
        T_target[:3, 3] = target_pos

        # Joint bounds (in radians)
        bounds = []
        for jname in chain:
            j = self.joints[jname]
            lo = max(j['lower'], -np.pi)
            hi = min(j['upper'], np.pi)
            bounds.append((lo, hi))

        # Initial guess: use mid-range if not provided
        if initial_guess is None:
            x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])
        else:
            x0 = np.array(initial_guess, dtype=np.float64)

        # Clamp initial guess to bounds
        for i in range(n):
            x0[i] = np.clip(x0[i], bounds[i][0], bounds[i][1])

        def cost(angles):
            T = self._chain_fk(chain, angles)
            return self._pose_error(T, T_target)

        # 两个初始猜测：上一帧角度 + 零位。上一帧通常直接收敛
        guesses = [x0]
        home = np.zeros(n)
        for i in range(n):
            home[i] = np.clip(0.0, bounds[i][0], bounds[i][1])
        if not np.allclose(x0, home):
            guesses.append(home)

        best_result = None
        best_cost = float('inf')

        for guess in guesses:
            result = minimize(
                cost, guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30, 'disp': False, 'ftol': 1e-4}
            )
            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result

        return best_result.x

    def get_joint_limits(self, chain_type: str = 'all') -> dict:
        """查关节限位，返回 {关节名: (下限, 上限)}"""
        if chain_type == 'right':
            chains = [self.right_arm_joints]
        elif chain_type == 'left':
            chains = [self.left_arm_joints]
        elif chain_type == 'torso':
            chains = [self.torso_joints]
        else:
            chains = [self.torso_joints, self.right_arm_joints, self.left_arm_joints]

        limits = {}
        for chain in chains:
            for jname in chain:
                j = self.joints[jname]
                limits[jname] = (j['lower'], j['upper'])
        return limits

    def get_all_joint_names(self) -> list:
        """17 个关节的排好序的名字列表（躯干+右臂+左臂，去重）"""
        # 左右臂的 chain 里也包含躯干关节，所以取后半截才是纯手臂
        right_arm_only = self.right_arm_joints[self.n_torso:]  # J1_R..J7_R
        left_arm_only = self.left_arm_joints[self.n_torso:]     # J1_L..J7_L
        return self.torso_joints + right_arm_only + left_arm_only


def hand_pose_to_robot_target(hand_wrist, hand_quat=None):
    """VR 手部坐标 → F1 机器人末端目标坐标

    坐标系映射（Quest 3 为例）：
      Quest:  X=右, Y=上, Z=后
      Robot:  X=前, Y=左, Z=上
      映射:   QuestY→RobotZ, Quest-Z→RobotX, Quest-X→RobotY

    下面这几个 scale/offset 是目测的，没经过物理标定。
    需要后续在 Isaac Sim 里跑一下看关节角合不合理。  ← TODO
    """

    # 坐标轴重映射
    target_pos = np.array([
        -hand_wrist[2],  # VR -Z → Robot X (前)
        -hand_wrist[0],  # VR -X → Robot Y (左)
        hand_wrist[1],   # VR Y  → Robot Z (上)
    ], dtype=np.float64)

    scale = 0.8
    offset = np.array([0.35, 0.0, 0.5])
    target_pos = target_pos * scale + offset

    # 没给姿态就用默认值（手心朝前）
    if hand_quat is None:
        hand_quat = np.array([0.0, 0.0, 0.0, 1.0])

    # 四元数暂时直接拷过去，后续需要旋转到机器人坐标系
    target_quat = hand_quat.copy()

    return target_pos, target_quat


if __name__ == "__main__":
    # 直接跑这个脚本可以快速测一下 FK 和 IK
    urdf_path = Path(__file__).parent.parent / "urdf" / "urdf" / "F1_URDF_V04.urdf"
    if not urdf_path.exists():
        urdf_path = Path.home() / "F1_URDF_V04" / "urdf" / "F1_URDF_V04.urdf"

    print(f"[Test] URDF: {urdf_path}")
    f1 = F1Kinematics(str(urdf_path))

    # 零位 FK
    zero = {j: 0.0 for j in f1.get_all_joint_names()}
    poses = f1.forward_kinematics(zero)
    print(f"\n[Test] FK 零位:")
    print(f"  右手末端: {poses['right_ee'][:3, 3]}")
    print(f"  左手末端: {poses['left_ee'][:3, 3]}")

    # IK 测试
    print(f"\n[Test] 右臂 IK：目标 [0.4, -0.2, 0.6]")
    target_pos = np.array([0.4, -0.2, 0.6])
    target_quat = np.array([0, 0, 0, 1])
    angles = f1.solve_ik(target_pos, target_quat, side='right')
    for jname, a in zip(f1.right_arm_joints, angles):
        print(f"  {jname}: {np.degrees(a):.1f}°")

    # 验证
    T = f1._chain_fk(f1.right_arm_joints, angles)
    err = np.linalg.norm(T[:3, 3] - target_pos)
    print(f"  位置误差: {err:.4f}m (目标: <0.001m)")
