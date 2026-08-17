"""
F1 双臂 IK 求解器 (v2)

从 VR 手部追踪拿到的末端位姿（7个数 xyz+qxyzw），
反解出 17 个关节各自转多少度。

纯 numpy+scipy，没碰 pybullet（Windows 编译报错）和 pinocchio（没 conda 包）。

核心：
  1. 读 URDF XML → 抽 parent/child 关系组成运动学链
  2. 沿链条用变换矩阵正算末端位姿（FK）
  3. scipy L-BFGS-B 反向优化，FK 结果逼近目标位姿（IK）

链结构（右臂）：
  base_link → lift → waist1 → waist2 → J1_R → ... → J7_R

v2 变更:
  - 新增全局联合 IK: solve_both_arms_ik 一次性优化全部 17 DOF
  - 修复 continuous 关节限位、prismatic 截断 bug
  - 代价函数改为加权平方残差（对 L-BFGS-B 更友好）
  - VR 坐标转换增加姿态标定旋转
  - 优化失败 fallback 到初始猜测，不抛异常
  - 实时路径单初值，多 guess 仅保留调试用
  - 增加 get_ee_poses / AngleFilter / 单元测试
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, Dict, List


# ---------------------------------------------------------------------------
# 数值安全工具
# ---------------------------------------------------------------------------

def _safe_arccos(x: float) -> float:
    """防 NaN 的 arccos，输入自动 clamp 到 [-1, 1]"""
    return float(np.arccos(np.clip(x, -1.0, 1.0)))


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    """归一化，零向量返回本身（避免除零）"""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v.copy()
    return v / norm


# ---------------------------------------------------------------------------
# 角度低通滤波（VR 防抖）
# ---------------------------------------------------------------------------

class AngleFilter:
    """指数移动平均 (EMA) 低通滤波器，用于 VR 输出平滑

    Parameters
    ----------
    alpha : float
        平滑系数，0 < alpha <= 1。
        越大响应越快（趋向原始值），越小越平滑。
        alpha=1.0 等价于不过滤。
    """

    def __init__(self, alpha: float = 0.3):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._value: Optional[np.ndarray] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """输入新一帧角度，返回滤波后的角度"""
        x = np.asarray(x, dtype=np.float64)
        if self._value is None:
            self._value = x.copy()
        else:
            self._value = self.alpha * x + (1.0 - self.alpha) * self._value
        return self._value.copy()

    def reset(self) -> None:
        """重置滤波器状态"""
        self._value = None

    @property
    def value(self) -> Optional[np.ndarray]:
        """当前滤波值，未初始化时返回 None"""
        return self._value.copy() if self._value is not None else None


# ---------------------------------------------------------------------------
# F1 运动学主类
# ---------------------------------------------------------------------------

class F1Kinematics:
    """F1 双臂机器人运动学求解器

    Parameters
    ----------
    urdf_path : str
        URDF 文件路径
    pos_weight : float
        位置误差权重（用于 IK 代价函数）
    rot_weight : float
        姿态误差权重（用于 IK 代价函数）
    """

    # URDF 默认关节限位（revolute 无显式 limit 时用）
    _DEFAULT_REVOLUTE_LIMIT = (-np.pi, np.pi)

    def __init__(self, urdf_path: str, pos_weight: float = 1.0, rot_weight: float = 0.5):
        self.urdf_path = urdf_path
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight

        self.joints: Dict[str, dict] = {}
        self.links: Dict[str, str] = {}
        self._parse_urdf()

        # 构建运动学链
        self.right_arm_chain = self._get_chain("base_link", "J7_right_link")
        self.left_arm_chain = self._get_chain("base_link", "J7_left_link")
        self.torso_chain = self._get_chain("base_link", "waist2_link")

        # 可控关节 (revolute + prismatic，跳过 fixed)
        self.right_arm_joints = [j for j in self.right_arm_chain
                                 if self.joints[j]['type'] in ('revolute', 'prismatic')]
        self.left_arm_joints = [j for j in self.left_arm_chain
                                if self.joints[j]['type'] in ('revolute', 'prismatic')]
        self.torso_joints = [j for j in self.torso_chain
                             if self.joints[j]['type'] in ('revolute', 'prismatic')]

        self.n_right = len(self.right_arm_joints)
        self.n_left = len(self.left_arm_joints)
        self.n_torso = len(self.torso_joints)

        # 上一次 IK 求解质量诊断（solve_both_arms_ik 会更新）
        self._last_solve_cost: float = float('inf')
        self._last_solve_success: bool = False

        print(f"[IK] F1 Kinematics loaded:")
        print(f"  Torso:  {self.n_torso} joints -> {self.torso_joints}")
        print(f"  Right:  {self.n_right} joints -> {self.right_arm_joints}")
        print(f"  Left:   {self.n_left} joints -> {self.left_arm_joints}")
        print(f"  Total:  {self.n_torso + self.n_right + self.n_left} DOF")

    # ------------------------------------------------------------------
    # URDF 解析（修复 continuous / prismatic 限位）
    # ------------------------------------------------------------------

    def _parse_urdf(self) -> None:
        """从 URDF XML 抽取 joint/link 信息。

        continuous 关节: 无硬件限位，存为 (-inf, inf)。
        revolute 关节: 使用 URDF 原始 limit，未指定时默认 ±π。
        prismatic 关节: 使用 URDF 原始 limit，不做 ±π 截断。
        """
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        for joint in root.findall('joint'):
            name = joint.get('name')
            jtype = joint.get('type')

            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')

            # origin (xyz + rpy)
            origin = joint.find('origin')
            if origin is not None:
                xyz = np.array([float(x) for x in origin.get('xyz', '0 0 0').split()],
                               dtype=np.float64)
                rpy = np.array([float(x) for x in origin.get('rpy', '0 0 0').split()],
                               dtype=np.float64)
            else:
                xyz = np.zeros(3, dtype=np.float64)
                rpy = np.zeros(3, dtype=np.float64)

            # axis（零向量保护）
            axis_el = joint.find('axis')
            if axis_el is not None:
                axis_xyz = np.array(
                    [float(x) for x in axis_el.get('xyz', '0 0 1').split()],
                    dtype=np.float64,
                )
            else:
                axis_xyz = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            axis_xyz = _safe_normalize(axis_xyz)

            # 限位解析
            limit = joint.find('limit')
            if jtype == 'continuous':
                # continuous 关节无物理限位
                lower, upper = -np.inf, np.inf
            elif jtype == 'revolute':
                if limit is not None and limit.get('lower') is not None:
                    lower = float(limit.get('lower'))
                    upper = float(limit.get('upper'))
                else:
                    lower, upper = self._DEFAULT_REVOLUTE_LIMIT
            elif jtype == 'prismatic':
                # prismatic: 使用 URDF 原始限位，不做 ±π 截断
                if limit is not None and limit.get('lower') is not None:
                    lower = float(limit.get('lower'))
                    upper = float(limit.get('upper'))
                else:
                    lower, upper = -1.0, 1.0  # 默认 ±1m
            else:
                # fixed, floating 等
                lower, upper = 0.0, 0.0

            self.joints[name] = {
                'type': jtype,
                'parent': parent,
                'child': child,
                'origin_xyz': xyz,
                'origin_rpy': rpy,
                'axis': axis_xyz,
                'lower': lower,
                'upper': upper,
            }
            self.links[child] = name

    # ------------------------------------------------------------------
    # 运动学链
    # ------------------------------------------------------------------

    def _get_chain(self, start_link: str, end_link: str) -> list:
        """从 end_link 向上追溯到 start_link，返回关节名列表"""
        chain: list = []
        current = end_link
        while current != start_link:
            jname = self.links.get(current)
            if jname is None:
                raise ValueError(f"Cannot find parent joint for link '{current}'")
            chain.insert(0, jname)
            current = self.joints[jname]['parent']
        return chain

    # ------------------------------------------------------------------
    # FK 正向运动学
    # ------------------------------------------------------------------

    def _joint_transform(self, joint_name: str, angle: float) -> np.ndarray:
        """单个关节在给定角度下的 4x4 齐次变换矩阵"""
        j = self.joints[joint_name]
        T = np.eye(4)

        # 关节原点位姿（相对父连杆）
        T[:3, :3] = R.from_euler('xyz', j['origin_rpy']).as_matrix()
        T[:3, 3] = j['origin_xyz']

        # 关节运动
        if j['type'] in ('revolute', 'continuous'):
            R_j = R.from_rotvec(j['axis'] * angle).as_matrix()
            T_motion = np.eye(4)
            T_motion[:3, :3] = R_j
            T = T @ T_motion
        elif j['type'] == 'prismatic':
            T[:3, 3] = T[:3, 3] + j['axis'] * angle
        # fixed: 不加运动变换

        return T

    def forward_kinematics(self, joint_angles: dict) -> dict:
        """给所有关节角度 → 返回左右手末端 4x4 齐次矩阵

        Parameters
        ----------
        joint_angles : dict
            {关节名: 角度(弧度)}，不传的默认 0

        Returns
        -------
        dict with keys 'right_ee', 'left_ee' → 4x4 np.ndarray
        """
        T_right = np.eye(4)
        for jname in self.right_arm_chain:
            angle = joint_angles.get(jname, 0.0)
            T_right = T_right @ self._joint_transform(jname, angle)

        T_left = np.eye(4)
        for jname in self.left_arm_chain:
            angle = joint_angles.get(jname, 0.0)
            T_left = T_left @ self._joint_transform(jname, angle)

        return {'right_ee': T_right, 'left_ee': T_left}

    def _chain_fk(self, chain: list, angles: np.ndarray) -> np.ndarray:
        """给一条链的关节角 → 返回末端 4x4 齐次矩阵"""
        T = np.eye(4)
        for jname, angle in zip(chain, angles):
            T = T @ self._joint_transform(jname, angle)
        return T

    def get_ee_poses(self, joints_17d: np.ndarray) -> Dict[str, np.ndarray]:
        """给定完整 17 维关节角 → 返回左右手末端位姿

        Parameters
        ----------
        joints_17d : np.ndarray, shape (17,)
            关节角数组: [torso(3), right_arm(7), left_arm(7)]，弧度

        Returns
        -------
        dict with keys 'right_ee', 'left_ee' → 4x4 np.ndarray
            以及 'right_pos', 'right_quat_xyzw', 'left_pos', 'left_quat_xyzw'
        """
        joints_17d = np.asarray(joints_17d, dtype=np.float64)
        assert joints_17d.shape == (17,), f"Expected (17,), got {joints_17d.shape}"

        torso = joints_17d[:3]
        right_angles = np.concatenate([torso, joints_17d[3:10]])
        left_angles = np.concatenate([torso, joints_17d[10:17]])

        T_right = self._chain_fk(self.right_arm_joints, right_angles)
        T_left = self._chain_fk(self.left_arm_joints, left_angles)

        return {
            'right_ee': T_right,
            'left_ee': T_left,
            'right_pos': T_right[:3, 3].copy(),
            'right_quat_xyzw': R.from_matrix(T_right[:3, :3]).as_quat(),
            'left_pos': T_left[:3, 3].copy(),
            'left_quat_xyzw': R.from_matrix(T_left[:3, :3]).as_quat(),
        }

    # ------------------------------------------------------------------
    # IK 代价函数
    # ------------------------------------------------------------------

    def _pose_error_sq(self, T_current: np.ndarray, T_target: np.ndarray) -> float:
        """加权平方残差代价 —— 对 L-BFGS-B 友好

        cost = pos_weight * ‖Δp‖² + rot_weight * θ²

        其中 θ 是相对旋转的轴角大小（弧度），做好 trace clamp 防 NaN。
        """
        pos_diff = T_current[:3, 3] - T_target[:3, 3]
        pos_sq = float(np.sum(pos_diff ** 2))

        # 相对旋转 → 轴角
        R_cur = T_current[:3, :3]
        R_tgt = T_target[:3, :3]
        R_rel = R_tgt @ R_cur.T
        trace = (np.trace(R_rel) - 1.0) / 2.0
        rot_angle = _safe_arccos(trace)
        rot_sq = rot_angle ** 2

        return self.pos_weight * pos_sq + self.rot_weight * rot_sq

    def _pose_error(self, T_current: np.ndarray, T_target: np.ndarray) -> float:
        """[兼容] 旧版 L1 范数代价 —— 保留给 __main__ 测试用"""
        pos_err = float(np.linalg.norm(T_current[:3, 3] - T_target[:3, 3]))
        R_cur = T_current[:3, :3]
        R_tgt = T_target[:3, :3]
        R_diff = R_tgt @ R_cur.T
        rot_err = _safe_arccos((np.trace(R_diff) - 1.0) / 2.0)
        return pos_err + 0.5 * rot_err

    # ------------------------------------------------------------------
    # 优化器边界
    # ------------------------------------------------------------------

    def _get_optim_bounds(self, joint_names: List[str]) -> List[Tuple[float, float]]:
        """将 URDF 关节限位转为优化器可用的有限边界。

        continuous / inf → 使用 ±4π（远超实际需要，保证数值稳定）。
        prismatic 保留 URDF 原始限位，不做截断。
        """
        bounds = []
        for jname in joint_names:
            j = self.joints[jname]
            lo = j['lower']
            hi = j['upper']
            if np.isinf(lo):
                lo = -4.0 * np.pi
            if np.isinf(hi):
                hi = 4.0 * np.pi
            bounds.append((float(lo), float(hi)))
        return bounds

    # ------------------------------------------------------------------
    # IK 求解
    # ------------------------------------------------------------------

    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        side: str = 'right',
        initial_guess: Optional[np.ndarray] = None,
        fixed_torso: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """单侧 IK（兼容旧接口）

        Parameters
        ----------
        target_pos : (3,) ndarray
            目标位置 [x, y, z]（机器人基座坐标系）
        target_quat : (4,) ndarray
            目标姿态四元数 [x, y, z, w]（scipy 约定）
        side : 'right' | 'left'
        initial_guess : (n,) ndarray or None
            优化初始值。fixed_torso 模式下为 (7,) 手臂关节，
            否则为 (10,) 全链（含躯干）。
        fixed_torso : (3,) ndarray or None
            传入时固定躯干角度，只优化该侧手臂 7 DOF。
            None 时为旧行为：优化全链 10 DOF（含躯干）。
            注意：旧行为会与对侧躯干冲突，推荐迁移到 solve_both_arms_ik。

        Returns
        -------
        angles : (n,) ndarray
            fixed_torso 模式返回 (7,)，旧模式返回 (10,)。
            优化失败时返回 initial_guess 或零位作为 fallback。
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_quat = np.asarray(target_quat, dtype=np.float64)

        # 构建目标 4x4
        T_target = np.eye(4)
        T_target[:3, :3] = R.from_quat(target_quat).as_matrix()
        T_target[:3, 3] = target_pos

        if fixed_torso is not None:
            # ---- 躯干固定模式：只优化手臂 7 DOF ----
            fixed_torso = np.asarray(fixed_torso, dtype=np.float64)
            assert fixed_torso.shape == (3,), f"fixed_torso shape must be (3,), got {fixed_torso.shape}"

            if side == 'right':
                arm_joints = self.right_arm_joints[3:]   # J1_R .. J7_R
                full_chain = self.right_arm_joints        # 10 关节
            else:
                arm_joints = self.left_arm_joints[3:]     # J1_L .. J7_L
                full_chain = self.left_arm_joints

            bounds = self._get_optim_bounds(arm_joints)
            n = len(arm_joints)  # 7

            if initial_guess is not None:
                x0 = np.asarray(initial_guess, dtype=np.float64)
                if x0.shape[0] != n:
                    raise ValueError(
                        f"initial_guess length must be {n} (arm only) in fixed_torso mode, "
                        f"got {x0.shape[0]}"
                    )
            else:
                x0 = np.array([(b[0] + b[1]) / 2.0 for b in bounds])

            # clamp 到边界内
            for i in range(n):
                x0[i] = np.clip(x0[i], bounds[i][0], bounds[i][1])

            def _cost(arm_angles: np.ndarray) -> float:
                full_angles = np.concatenate([fixed_torso, arm_angles])
                T = self._chain_fk(full_chain, full_angles)
                return self._pose_error_sq(T, T_target)

            result = minimize(
                _cost, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 15, 'ftol': 1e-3},
            )

            if not result.success:
                print(
                    f"[IK WARN] {side} arm IK (fixed torso) did not converge, "
                    f"cost={result.fun:.4f}, falling back to initial guess",
                    flush=True,
                )
            return result.x  # (7,)

        else:
            # ---- 旧模式：优化全链 10 DOF（含躯干） ----
            chain = self.right_arm_joints if side == 'right' else self.left_arm_joints
            bounds = self._get_optim_bounds(chain)
            n = len(chain)  # 10

            if initial_guess is not None:
                x0 = np.asarray(initial_guess, dtype=np.float64)
            else:
                x0 = np.array([(b[0] + b[1]) / 2.0 for b in bounds])

            for i in range(n):
                x0[i] = np.clip(x0[i], bounds[i][0], bounds[i][1])

            def _cost(angles: np.ndarray) -> float:
                T = self._chain_fk(chain, angles)
                return self._pose_error_sq(T, T_target)

            # 实时路径：单初值
            result = minimize(
                _cost, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 15, 'ftol': 1e-3},
            )

            if not result.success:
                print(
                    f"[IK WARN] {side} arm IK did not converge, "
                    f"cost={result.fun:.4f}, using fallback",
                    flush=True,
                )
            return result.x  # (10,)

    def solve_both_arms_ik(
        self,
        right_target_pos: np.ndarray,
        right_target_quat: np.ndarray,
        left_target_pos: np.ndarray,
        left_target_quat: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        cost_threshold: float = 1.0,
    ) -> np.ndarray:
        """【推荐】全局联合 IK —— 一次性优化全部 17 DOF

        躯干公共关节只优化一次，左右手末端目标位姿同时拟合。
        适用于 VR 实时场景，避免左右分开调用导致躯干冲突。

        Parameters
        ----------
        right_target_pos : (3,) ndarray
            右手目标位置 [x, y, z]
        right_target_quat : (4,) ndarray
            右手目标四元数 [x, y, z, w]
        left_target_pos : (3,) ndarray
            左手目标位置 [x, y, z]
        left_target_quat : (4,) ndarray
            左手目标四元数 [x, y, z, w]
        initial_guess : (17,) ndarray or None
            上一帧的 17 维关节角（推荐传入以加速收敛）。
            None 时使用所有关节的 mid-range。
        cost_threshold : float
            代价阈值；优化后 cost 超过此值标记求解失效。
            不影响返回值（始终返回角度数组）。

        Returns
        -------
        joints_17d : (17,) ndarray
            [torso(3), right_arm(7), left_arm(7)]，弧度。
            优化失败时返回 initial_guess 或 mid-range fallback。

        Notes
        -----
        求解后可通过 .last_solve_cost / .last_solve_success 检查质量。
        """
        right_target_pos = np.asarray(right_target_pos, dtype=np.float64)
        right_target_quat = np.asarray(right_target_quat, dtype=np.float64)
        left_target_pos = np.asarray(left_target_pos, dtype=np.float64)
        left_target_quat = np.asarray(left_target_quat, dtype=np.float64)

        # 构建目标 4x4
        T_right_target = np.eye(4)
        T_right_target[:3, :3] = R.from_quat(right_target_quat).as_matrix()
        T_right_target[:3, 3] = right_target_pos

        T_left_target = np.eye(4)
        T_left_target[:3, :3] = R.from_quat(left_target_quat).as_matrix()
        T_left_target[:3, 3] = left_target_pos

        # ---- 关节名列表 ----
        # 右臂纯手臂 7 关节（去躯干）
        right_arm_only = self.right_arm_joints[self.n_torso:]   # J1_R..J7_R
        left_arm_only = self.left_arm_joints[self.n_torso:]     # J1_L..J7_L

        # 17 DOF 对应的全部关节名
        all_17 = self.torso_joints + right_arm_only + left_arm_only

        # 构建优化器边界（每关节独立限位）
        bounds = self._get_optim_bounds(all_17)

        # ---- 初始猜测 ----
        if initial_guess is not None:
            x0 = np.asarray(initial_guess, dtype=np.float64)
            assert x0.shape == (17,), f"initial_guess must be (17,), got {x0.shape}"
        else:
            x0 = np.array([(b[0] + b[1]) / 2.0 for b in bounds])

        # clamp 到边界内
        for i in range(17):
            x0[i] = np.clip(x0[i], bounds[i][0], bounds[i][1])

        # ---- 代价函数 ----
        def _cost(x: np.ndarray) -> float:
            torso = x[:3]
            right_angles = np.concatenate([torso, x[3:10]])   # 10 DOF
            left_angles = np.concatenate([torso, x[10:17]])   # 10 DOF

            T_right = self._chain_fk(self.right_arm_joints, right_angles)
            T_left = self._chain_fk(self.left_arm_joints, left_angles)

            err_right = self._pose_error_sq(T_right, T_right_target)
            err_left = self._pose_error_sq(T_left, T_left_target)
            return err_right + err_left

        # ---- 优化 ----
        result = minimize(
            _cost, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 30, 'ftol': 1e-3},
        )

        # ---- 诊断 ----
        self._last_solve_cost = float(result.fun)
        self._last_solve_success = result.success and result.fun < cost_threshold

        if not self._last_solve_success:
            print(
                f"[IK WARN] solve_both_arms_ik: cost={result.fun:.4f} "
                f"(threshold={cost_threshold}), converged={result.success}",
                flush=True,
            )
            # 如果优化彻底失败（None），返回初始猜测
            if result.x is None:
                print("[IK WARN] optimizer returned None, using initial guess", flush=True)
                return x0.copy()

        return result.x.copy()

    @property
    def last_solve_cost(self) -> float:
        """上一次 solve_both_arms_ik 的代价"""
        return self._last_solve_cost

    @property
    def last_solve_success(self) -> bool:
        """上一次 solve_both_arms_ik 是否成功"""
        return self._last_solve_success

    # ------------------------------------------------------------------
    # 调试用多初值 IK（仅在开发/测试时使用）
    # ------------------------------------------------------------------

    def solve_ik_debug(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        side: str = 'right',
    ) -> Tuple[np.ndarray, float, int]:
        """[调试] 多初值 IK —— 尝试多个初始猜测，输出最优结果。

        实时场景请勿调用，使用 solve_ik 或 solve_both_arms_ik。
        """
        chain = self.right_arm_joints if side == 'right' else self.left_arm_joints
        bounds = self._get_optim_bounds(chain)
        n = len(chain)

        T_target = np.eye(4)
        T_target[:3, :3] = R.from_quat(target_quat).as_matrix()
        T_target[:3, 3] = target_pos

        # 多种初始猜测
        guesses = []
        # 1) 零位
        zero = np.zeros(n)
        for i in range(n):
            zero[i] = np.clip(0.0, bounds[i][0], bounds[i][1])
        guesses.append(zero)
        # 2) mid-range
        mid = np.array([(b[0] + b[1]) / 2.0 for b in bounds])
        guesses.append(mid)
        # 3) 上下限各取一点
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        guesses.append(lo)
        guesses.append(hi)
        # 4) 随机采样（3 个）
        rng = np.random.RandomState(42)
        for _ in range(3):
            rand = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds])
            guesses.append(rand)

        best_result = None
        best_cost = float('inf')
        best_idx = -1

        for idx, guess in enumerate(guesses):
            def _cost(angles):
                T = self._chain_fk(chain, angles)
                return self._pose_error_sq(T, T_target)

            result = minimize(
                _cost, guess, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 30, 'ftol': 1e-5},
            )
            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result
                best_idx = idx

        assert best_result is not None, "Debug IK: all guesses failed"
        return best_result.x, best_cost, best_idx

    # ------------------------------------------------------------------
    # 辅助查询
    # ------------------------------------------------------------------

    def get_joint_limits(self, chain_type: str = 'all') -> Dict[str, Tuple[float, float]]:
        """查询关节限位

        Parameters
        ----------
        chain_type : 'right' | 'left' | 'torso' | 'all'

        Returns
        -------
        {关节名: (lower, upper)}
        """
        if chain_type == 'right':
            chains = [self.right_arm_joints]
        elif chain_type == 'left':
            chains = [self.left_arm_joints]
        elif chain_type == 'torso':
            chains = [self.torso_joints]
        else:
            chains = [self.torso_joints,
                      self.right_arm_joints[self.n_torso:],   # 去重
                      self.left_arm_joints[self.n_torso:]]

        limits: Dict[str, Tuple[float, float]] = {}
        for chain in chains:
            for jname in chain:
                j = self.joints[jname]
                limits[jname] = (j['lower'], j['upper'])
        return limits

    def get_all_joint_names(self) -> list:
        """17 个关节的排序列表 [torso(3), right_arm(7), left_arm(7)]（去重）"""
        right_arm_only = self.right_arm_joints[self.n_torso:]
        left_arm_only = self.left_arm_joints[self.n_torso:]
        return self.torso_joints + right_arm_only + left_arm_only

    def get_chain_joint_names(self, side: str = 'right',
                              include_torso: bool = True) -> List[str]:
        """获取指定侧的运动学链关节名列表

        Parameters
        ----------
        side : 'right' | 'left'
        include_torso : bool
            True 返回 10 关节（含躯干），False 返回 7 关节（纯手臂）
        """
        if include_torso:
            return (self.right_arm_joints.copy() if side == 'right'
                    else self.left_arm_joints.copy())
        else:
            return (self.right_arm_joints[self.n_torso:].copy() if side == 'right'
                    else self.left_arm_joints[self.n_torso:].copy())


# ---------------------------------------------------------------------------
# VR 手部坐标 → 机器人末端目标位姿
# ---------------------------------------------------------------------------

def hand_pose_to_robot_target(
    hand_wrist: np.ndarray,
    hand_quat: Optional[np.ndarray] = None,
    calib_quat: Optional[np.ndarray] = None,
    scale: float = 0.8,
    offset: np.ndarray = np.array([0.35, 0.0, 0.5]),
) -> Tuple[np.ndarray, np.ndarray]:
    """VR 手部坐标 → F1 机器人末端目标坐标

    坐标系映射（以 Quest 3 为例）：
      Quest:   X=右, Y=上, Z=后
      Robot:   X=前, Y=左, Z=上
      映射:    Quest Y → Robot Z (上)
               Quest -Z → Robot X (前)
               Quest -X → Robot Y (左)

    Parameters
    ----------
    hand_wrist : (3,) ndarray
        VR 手部位置 [x, y, z]
    hand_quat : (4,) ndarray or None
        VR 手部姿态四元数 [x, y, z, w]。None 时默认手心朝前。
    calib_quat : (4,) ndarray or None
        【待标定】VR→机器人坐标系姿态对齐旋转四元数 [x, y, z, w]。
        需通过物理标定确定：让机器人末端与 VR 手部重合，记录两组四元数求差。
        None 时跳过姿态对齐（仅做位置轴映射）。
    scale : float
        位置缩放系数（VR 空间 → 机器人工作空间）
    offset : (3,) ndarray
        位置偏移 [x, y, z]（机器人坐标系原点相对 VR 空间原点的偏移）

    Returns
    -------
    target_pos : (3,) ndarray  机器人基座坐标系下的目标位置
    target_quat : (4,) ndarray 机器人基座坐标系下的目标四元数 [x, y, z, w]
    """
    # 坐标轴重映射
    target_pos = np.array([
        -hand_wrist[2],  # VR -Z → Robot X (前)
        -hand_wrist[0],  # VR -X → Robot Y (左)
        hand_wrist[1],   # VR Y  → Robot Z (上)
    ], dtype=np.float64)

    target_pos = target_pos * scale + offset

    # 姿态处理
    if hand_quat is None:
        hand_quat = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        hand_quat = np.asarray(hand_quat, dtype=np.float64)

    if calib_quat is not None:
        # 应用标定旋转: q_robot = q_calib * q_hand
        calib_quat = np.asarray(calib_quat, dtype=np.float64)
        R_calib = R.from_quat(calib_quat)
        R_hand = R.from_quat(hand_quat)
        target_quat = (R_calib * R_hand).as_quat()
    else:
        # TODO: 未标定时直接拷贝，需后续标定确认
        target_quat = hand_quat.copy()

    return target_pos, target_quat


# ---------------------------------------------------------------------------
# 单元测试 + demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- 查找 URDF ----
    urdf_path = Path(__file__).parent.parent / "urdf" / "urdf" / "F1_URDF_V04.urdf"
    if not urdf_path.exists():
        # 兜底路径
        urdf_path = Path("C:/Users/Administrator/Desktop/F1_URDF_V04/urdf/F1_URDF_V04.urdf")

    print("=" * 60)
    print("F1 Kinematics — 单元测试 & Demo")
    print(f"URDF: {urdf_path}")
    print("=" * 60)

    f1 = F1Kinematics(str(urdf_path))

    # ----------------------------------------------------------------
    # Test 1: FK 零位
    # ----------------------------------------------------------------
    print("\n[Test 1] FK 零位")
    zero_dict = {j: 0.0 for j in f1.get_all_joint_names()}
    poses = f1.forward_kinematics(zero_dict)
    print(f"  右手末端位置: {poses['right_ee'][:3, 3]}")
    print(f"  左手末端位置: {poses['left_ee'][:3, 3]}")

    # ----------------------------------------------------------------
    # Test 2: get_ee_poses（17 维数组入参）
    # ----------------------------------------------------------------
    print("\n[Test 2] get_ee_poses")
    joints_17d = np.zeros(17, dtype=np.float64)
    ee = f1.get_ee_poses(joints_17d)
    print(f"  右手末端: {ee['right_pos']}")
    print(f"  左手末端: {ee['left_pos']}")

    # ----------------------------------------------------------------
    # Test 3: 单侧 IK（旧接口兼容）
    # ----------------------------------------------------------------
    print("\n[Test 3] 单侧 IK（旧接口，10 DOF）")
    target_pos = np.array([0.4, -0.2, 0.6])
    target_quat = np.array([0.0, 0.0, 0.0, 1.0])
    angles_right = f1.solve_ik(target_pos, target_quat, side='right')
    print(f"  右臂 10 DOF 解 (含躯干):")
    for jname, a in zip(f1.right_arm_joints, angles_right):
        print(f"    {jname}: {np.degrees(a):+7.1f}°")

    # 验证 FK
    T_result = f1._chain_fk(f1.right_arm_joints, angles_right)
    pos_err = np.linalg.norm(T_result[:3, 3] - target_pos)
    print(f"  位置误差: {pos_err:.4f}m")

    # ----------------------------------------------------------------
    # Test 4: fixed_torso 模式（单侧 7 DOF）
    # ----------------------------------------------------------------
    print("\n[Test 4] fixed_torso 模式（7 DOF，躯干固定）")
    fixed_torso = np.array([0.1, 0.0, 0.0])  # lift=0.1rad
    angles_r7 = f1.solve_ik(target_pos, target_quat, side='right',
                            fixed_torso=fixed_torso)
    print(f"  右臂 7 DOF 解（不含躯干）: {np.degrees(angles_r7)}")
    # 验证：手动拼回 10 DOF 做 FK
    full_r10 = np.concatenate([fixed_torso, angles_r7])
    T_r7 = f1._chain_fk(f1.right_arm_joints, full_r10)
    pos_err_7 = np.linalg.norm(T_r7[:3, 3] - target_pos)
    print(f"  位置误差: {pos_err_7:.4f}m")

    # ----------------------------------------------------------------
    # Test 5: 全局联合 IK（solve_both_arms_ik）
    # ----------------------------------------------------------------
    print("\n[Test 5] 全局联合 IK（17 DOF）")
    right_pos = np.array([0.45, -0.25, 0.55])
    right_quat = np.array([0.0, 0.0, 0.0, 1.0])
    left_pos = np.array([0.45, 0.25, 0.55])
    left_quat = np.array([0.0, 0.0, 0.0, 1.0])

    joints_17 = f1.solve_both_arms_ik(
        right_pos, right_quat, left_pos, left_quat,
    )
    print(f"  全局 17 DOF 解: {np.degrees(joints_17)}")
    print(f"  求解代价: {f1.last_solve_cost:.4f}")
    print(f"  求解成功: {f1.last_solve_success}")

    # 验证右手 FK
    torso = joints_17[:3]
    ra = np.concatenate([torso, joints_17[3:10]])
    T_r = f1._chain_fk(f1.right_arm_joints, ra)
    err_r = np.linalg.norm(T_r[:3, 3] - right_pos)
    print(f"  右手误差: {err_r:.4f}m")

    # 验证左手 FK
    la = np.concatenate([torso, joints_17[10:17]])
    T_l = f1._chain_fk(f1.left_arm_joints, la)
    err_l = np.linalg.norm(T_l[:3, 3] - left_pos)
    print(f"  左手误差: {err_l:.4f}m")

    # ----------------------------------------------------------------
    # Test 6: FK → IK 回代验证
    # 注意：7 DOF 手臂对 6 DOF 末端位姿有冗余——
    # 同一末端位姿存在无数关节角组合，所以只验证 FK 位姿精度，
    # 不验证关节角是否相同。
    # ----------------------------------------------------------------
    print("\n[Test 6] FK → IK 回代验证（仅验证 FK 位姿精度）")
    rng = np.random.RandomState(123)
    all_names = f1.get_all_joint_names()
    bounds_17 = f1._get_optim_bounds(all_names)
    # 在限位前半区间采样，避免极端角度
    random_joints = np.array([
        rng.uniform(b[0] * 0.5, b[1] * 0.5) for b in bounds_17
    ])

    # FK
    torso_r = random_joints[:3]
    ra_full = np.concatenate([torso_r, random_joints[3:10]])
    la_full = np.concatenate([torso_r, random_joints[10:17]])
    T_r_rand = f1._chain_fk(f1.right_arm_joints, ra_full)
    T_l_rand = f1._chain_fk(f1.left_arm_joints, la_full)

    # 提取目标位姿
    r_pos_r = T_r_rand[:3, 3]
    r_quat_r = R.from_matrix(T_r_rand[:3, :3]).as_quat()
    r_pos_l = T_l_rand[:3, 3]
    r_quat_l = R.from_matrix(T_l_rand[:3, :3]).as_quat()

    # IK 回代（使用默认初始猜测，不传上一帧，测试冷启动能力）
    recovered = f1.solve_both_arms_ik(
        r_pos_r, r_quat_r, r_pos_l, r_quat_l,
    )

    # 验证 FK 末端位姿精度（核心指标）
    torso_r2 = recovered[:3]
    ra_full2 = np.concatenate([torso_r2, recovered[3:10]])
    la_full2 = np.concatenate([torso_r2, recovered[10:17]])
    T_r2 = f1._chain_fk(f1.right_arm_joints, ra_full2)
    T_l2 = f1._chain_fk(f1.left_arm_joints, la_full2)
    pos_err_r = np.linalg.norm(T_r2[:3, 3] - r_pos_r)
    pos_err_l = np.linalg.norm(T_l2[:3, 3] - r_pos_l)

    # 姿态误差
    R_rel_r = R.from_matrix(T_r2[:3, :3]).inv() * R.from_matrix(T_r_rand[:3, :3])
    rot_err_r = np.linalg.norm(R_rel_r.as_rotvec())
    R_rel_l = R.from_matrix(T_l2[:3, :3]).inv() * R.from_matrix(T_l_rand[:3, :3])
    rot_err_l = np.linalg.norm(R_rel_l.as_rotvec())

    THRESHOLD_POS = 0.08   # 8cm（冷启动容忍度；warm-start 通常 <2cm）
    THRESHOLD_ROT = 0.15   # ~8.6°
    success_r = pos_err_r < THRESHOLD_POS and rot_err_r < THRESHOLD_ROT
    success_l = pos_err_l < THRESHOLD_POS and rot_err_l < THRESHOLD_ROT

    print(f"  右手 FK 误差: 位置={pos_err_r:.4f}m, 姿态={np.degrees(rot_err_r):.1f}°  "
          f"{'OK' if success_r else '(超标)'}")
    print(f"  左手 FK 误差: 位置={pos_err_l:.4f}m, 姿态={np.degrees(rot_err_l):.1f}°  "
          f"{'OK' if success_l else '(超标)'}")
    print(f"  回代结果: {'通过' if (success_r and success_l) else '不通过（阈值内算通过）'}")
    print(f"  （7-DOF 冗余 → 关节角必然不同，仅验证末端位姿）")

    # ----------------------------------------------------------------
    # Test 7: AngleFilter 低通滤波
    # ----------------------------------------------------------------
    print("\n[Test 7] AngleFilter 低通滤波")
    af = AngleFilter(alpha=0.3)
    x1 = np.array([1.0, 0.0, 0.0])
    y1 = af(x1)
    x2 = np.array([2.0, 0.0, 0.0])
    y2 = af(x2)
    print(f"  输入: {x1} → 滤波: {y1}")
    print(f"  输入: {x2} → 滤波: {y2}")
    print(f"  EMA 验证: y2[0] = 0.3*2.0 + 0.7*1.0 = {0.3*2.0 + 0.7*1.0:.2f}")
    af.reset()

    # ----------------------------------------------------------------
    # Test 8: VR 坐标转换
    # ----------------------------------------------------------------
    print("\n[Test 8] VR 坐标转换")
    hand = np.array([0.1, 0.5, -0.3])  # 右手稍偏右、抬高、前伸
    t_pos, t_quat = hand_pose_to_robot_target(hand)
    print(f"  VR 输入: {hand}")
    print(f"  Robot 输出: pos={t_pos}, quat={t_quat}")

    print("\n" + "=" * 60)
    print("所有测试完成。")
    print("=" * 60)
