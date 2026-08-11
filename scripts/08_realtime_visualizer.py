"""
F1 VR 遥操数据实时可视化平台
========================================================================
戴上 Quest 3 做动作，浏览器实时显示 17 轴关节角曲线。
能一边录一边看，有异常抖动/跳变自动报警。

用法:
  python scripts/08_realtime_visualizer.py                # 实时监听 UDP :9000
  python scripts/08_realtime_visualizer.py --port 9999     # 换端口（Pico 用 9999）
  python scripts/08_realtime_visualizer.py --no-ik          # 只看手部原始数据
  python scripts/08_realtime_visualizer.py --replay XXX.h5  # 回放历史文件

然后浏览器打开 http://localhost:8765

依赖: fastapi, uvicorn, numpy, scipy, h5py（前两个是新装的，后三个管线已有）
------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import struct
import threading
import time
import queue
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

# ============================================================================
# 命令行参数
# ============================================================================
parser = argparse.ArgumentParser(description="F1 VR 遥操数据实时可视化")
parser.add_argument("--port", type=int, default=9000, help="UDP 监听端口 (Quest3=9000, Pico=9999)")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--no-ik", action="store_true", help="跳过 IK，只看手部原始数据")
parser.add_argument("--replay", type=str, default=None, help="回放 HDF5 文件而非实时监听")
parser.add_argument("--web-port", type=int, default=8765, help="Web 服务端口")
parser.add_argument("--fps", type=int, default=30, help="IK 处理帧率")
parser.add_argument("--max-history", type=int, default=3600, help="全量历史最大帧数（默认 3600 = 2分钟）")
parser.add_argument("--alert-threshold", type=float, default=10.0, help="帧间跳变报警阈值（度）")

# ============================================================================
# 状态管理 —— 所有共享数据都在这里，一把锁管住
# ============================================================================
class DashboardState:
    def __init__(self, max_ring: int = 900, max_history: int = 3600, alert_deg: float = 10.0):
        self.max_ring = max_ring          # 30s @ 30fps
        self.max_history = max_history
        self.alert_deg = alert_deg

        # 环形缓冲区（最近 30 秒，发给新连上的客户端追赶用）
        self.ring: deque = deque(maxlen=max_ring)

        # 全量历史 [{ts, joints:[17], right_pos:[3], left_pos:[3], right_ik:bool, left_ik:bool}, ...]
        self.history: deque = deque(maxlen=max_history)

        # 当前帧
        self.current: dict | None = None
        self.current_raw_wrist: dict = {"left": None, "right": None}

        # 告警
        self.alerts: deque = deque(maxlen=200)

        # 录制
        self.recording = False
        self.record_start_time: float | None = None
        self.record_buffer: list = []

        # IK
        self.ik_enabled = True
        self.right_guess: np.ndarray | None = None
        self.left_guess: np.ndarray | None = None
        self.right_ok_count = 0
        self.left_ok_count = 0
        self.ik_total = 0

        # WebSocket 客户端列表
        self.clients: set = set()

        # 帧计数器
        self.frame_count = 0

        # 线程安全的锁（用于 UDP 线程和 asyncio 之间的桥接）
        self._lock = threading.Lock()
        self._pending_raw: deque = deque()  # UDP 线程写入，asyncio 读出

    def push_raw(self, side: str, ptype: str, values: list):
        """UDP 线程调用：把解析好的手部数据扔进来"""
        with self._lock:
            self._pending_raw.append({
                "side": side,
                "type": ptype,
                "values": values,
                "ts": time.time(),
            })

    def drain_raw(self) -> list:
        """asyncio 线程调用：把积压的原始数据全部取走"""
        with self._lock:
            batch = list(self._pending_raw)
            self._pending_raw.clear()
        return batch

    def add_frame(self, frame: dict):
        """存入一帧 IK 结果"""
        self.ring.append(frame)
        self.history.append(frame)
        self.current = frame
        self.frame_count += 1

        # 检测帧间跳变（跟上一帧比）
        if len(self.history) >= 2:
            prev = self.history[-2]["joints"]
            curr = frame["joints"]
            if prev is not None and curr is not None:
                diffs = np.abs(np.array(curr) - np.array(prev))
                max_idx = int(np.argmax(diffs))
                max_diff = float(diffs[max_idx])
                if max_diff > self.alert_deg:
                    self.add_alert("warning",
                        f"帧间跳变 {max_diff:.1f}° → 关节#{max_idx} ({self._joint_name(max_idx)})")

    def add_alert(self, level: str, msg: str):
        self.alerts.append({
            "ts": time.time(),
            "time": datetime.now().strftime("%H:%M:%S"),
            "level": level,  # warning / error / info
            "msg": msg,
        })

    def start_recording(self):
        self.recording = True
        self.record_start_time = time.time()
        self.record_buffer.clear()
        self.add_alert("info", "▶ 开始录制数据...")

    def stop_recording(self) -> list:
        self.recording = False
        frames = list(self.record_buffer)
        self.record_buffer.clear()
        duration = time.time() - (self.record_start_time or time.time())
        self.add_alert("info", f"⏹ 录制结束：{len(frames)} 帧，{duration:.1f} 秒")
        return frames

    def record_frame(self, frame: dict):
        if self.recording:
            self.record_buffer.append(frame)

    def get_status(self) -> dict:
        j = self.current
        joints = j["joints"] if j else None
        return {
            "frame_count": self.frame_count,
            "recording": self.recording,
            "record_elapsed": time.time() - self.record_start_time if self.recording and self.record_start_time else 0,
            "record_frames": len(self.record_buffer) if self.recording else 0,
            "right_ik_ok": self.right_ok_count,
            "left_ik_ok": self.left_ok_count,
            "ik_total": self.ik_total,
            "current_joints": joints,
            "right_wrist": j["right_wrist"] if j else None,
            "left_wrist": j["left_wrist"] if j else None,
            "alerts": list(self.alerts),
        }

    # ---- 关节名映射 ----
    JOINT_NAMES = [
        "lift", "waist1", "waist2",
        "J1_R", "J2_R", "J3_R", "J4_R", "J5_R", "J6_R", "J7_R",
        "J1_L", "J2_L", "J3_L", "J4_L", "J5_L", "J6_L", "J7_L",
    ]

    def _joint_name(self, idx: int) -> str:
        if 0 <= idx < len(self.JOINT_NAMES):
            return self.JOINT_NAMES[idx]
        return f"joint_{idx}"


# ============================================================================
# UDP 监听线程
# ============================================================================
def udp_listener(port: int, host: str, state: DashboardState, stop_event: threading.Event):
    """在后台线程跑，持续收 UDP 包解析后 push 到 state"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as e:
        print(f"[UDP] 绑定 {host}:{port} 失败: {e}")
        print("[UDP] 提示：端口可能被占用，换个 --port 试试")
        stop_event.set()
        return

    sock.settimeout(0.5)
    print(f"[UDP] 监听 {host}:{port}，等 Quest 3 发数据...")

    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[UDP] 收包异常: {e}")
            continue

        try:
            text = data.decode("utf-8", errors="ignore").strip()
            for line in text.split("\n"):
                _parse_udp_line(line, state)
        except Exception:
            pass  # 解析失败就跳过，不影响主循环

    sock.close()
    print("[UDP] 监听线程退出")


def _parse_udp_line(line: str, state: DashboardState):
    """解析一行 UDP 数据"""
    if "|" not in line or ":" not in line:
        return

    meta, vals_str = line.split(":", 1)
    if not vals_str.strip():
        return

    type_side = meta.split("|")[0].strip()
    parts = type_side.split(" ")
    if len(parts) < 2:
        return

    side = parts[0]   # "Left" / "Right"
    ptype = parts[1]  # "wrist" / "landmarks"

    if side not in ("Left", "Right"):
        return

    try:
        vals = [float(x) for x in vals_str.strip().strip(",").split(",") if x.strip()]
    except ValueError:
        return

    if ptype == "wrist" and len(vals) >= 7:
        state.push_raw(side, "wrist", vals[:7])
    elif ptype == "landmarks" and len(vals) >= 63:
        # 手指关键点（目前 IK 没用到，但存着以后用）
        pass


# ============================================================================
# IK 处理器 —— 包装 ik_solver.py 的能力
# ============================================================================
class IKProcessor:
    def __init__(self, urdf_path: str | None = None):
        if urdf_path is None:
            # 自动找 URDF
            candidates = [
                Path(__file__).parent.parent / "urdf" / "F1_URDF_V04.urdf",
                Path("C:/Users/Administrator/Desktop/F1_URDF_V04/urdf/F1_URDF_V04.urdf"),
            ]
            urdf_path = None
            for c in candidates:
                if c.exists():
                    urdf_path = str(c)
                    break
            if urdf_path is None:
                raise FileNotFoundError(f"找不到 URDF 文件，试过: {candidates}")

        # 动态 import，避免没装 scipy 时 import 就炸
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from ik_solver import F1Kinematics, hand_pose_to_robot_target

        print(f"[IK] 加载 URDF: {urdf_path}")
        self.f1 = F1Kinematics(str(urdf_path))
        self.all_joint_names = self.f1.get_all_joint_names()
        self.n_joints = len(self.all_joint_names)

        # 关节限位（角度制）
        limits = self.f1.get_joint_limits("all")
        self.joint_limits_deg = []
        for jn in self.all_joint_names:
            lo, hi = limits.get(jn, (-np.pi, np.pi))
            self.joint_limits_deg.append((np.degrees(lo), np.degrees(hi)))

    def solve_frame(self, right_wrist, left_wrist, right_guess, left_guess) -> dict:
        """对一帧手部数据做 IK"""
        from ik_solver import hand_pose_to_robot_target

        # --- 右手 ---
        ra = np.zeros(self.f1.n_right)
        right_ok = False
        r_pos = np.array(right_wrist[:3]) if right_wrist is not None else np.zeros(3)
        r_quat = np.array(right_wrist[3:7]) if (right_wrist is not None and len(right_wrist) >= 7) else np.array([0, 0, 0, 1])

        if right_wrist is not None and np.linalg.norm(r_pos) > 0.001:
            try:
                t_pos, t_quat = hand_pose_to_robot_target(r_pos, r_quat)
                ra = self.f1.solve_ik(t_pos, t_quat, side="right", initial_guess=right_guess)
                right_guess = ra.copy()
                right_ok = True
            except Exception:
                ra = np.zeros(self.f1.n_right)

        # --- 左手 ---
        la = np.zeros(self.f1.n_left)
        left_ok = False
        l_pos = np.array(left_wrist[:3]) if left_wrist is not None else np.zeros(3)
        l_quat = np.array(left_wrist[3:7]) if (left_wrist is not None and len(left_wrist) >= 7) else np.array([0, 0, 0, 1])

        if left_wrist is not None and np.linalg.norm(l_pos) > 0.001:
            try:
                t_pos, t_quat = hand_pose_to_robot_target(l_pos, l_quat)
                la = self.f1.solve_ik(t_pos, t_quat, side="left", initial_guess=left_guess)
                left_guess = la.copy()
                left_ok = True
            except Exception:
                la = np.zeros(self.f1.n_left)

        # --- 合并 17 维 ---
        torso = (ra[:self.f1.n_torso] + la[:self.f1.n_torso]) / 2.0
        right_arm = ra[self.f1.n_torso:]
        left_arm = la[self.f1.n_torso:]
        joints = np.concatenate([torso, right_arm, left_arm])

        # 角度制（前端画图用）
        joints_deg = np.degrees(joints).tolist()

        return {
            "joints": joints_deg,
            "right_wrist": r_pos.tolist(),
            "left_wrist": l_pos.tolist(),
            "right_ik_ok": right_ok,
            "left_ik_ok": left_ok,
            "right_guess": right_guess,
            "left_guess": left_guess,
        }


# ============================================================================
# HTML 仪表盘 —— 吃我 500 行前端代码
# ============================================================================
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>F1 VR 遥操数据实时监控</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a1a;color:#ddd;font-family:'Segoe UI','Microsoft YaHei',sans-serif;overflow:hidden;height:100vh}
#app{display:flex;flex-direction:column;height:100vh}
/* ---- 顶栏 ---- */
#topbar{display:flex;align-items:center;padding:8px 16px;background:#222;border-bottom:1px solid #333;gap:16px;flex-shrink:0}
#topbar h1{font-size:18px;color:#fff;white-space:nowrap}
.status-dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.status-dot.green{background:#00e676;box-shadow:0 0 6px #00e676}
.status-dot.red{background:#ff5252;box-shadow:0 0 6px #ff5252}
.status-dot.yellow{background:#ffd740;box-shadow:0 0 6px #ffd740}
.stat-item{font-size:13px;color:#aaa;white-space:nowrap}
.stat-item span{color:#fff;font-weight:bold}
.stat-item .label{color:#888}
/* ---- 按钮 ---- */
.btn{padding:6px 16px;border:none;border-radius:4px;cursor:pointer;font-size:13px;font-weight:bold;transition:all 0.2s}
.btn:disabled{opacity:0.4;cursor:not-allowed}
.btn-record{background:#ff5252;color:#fff}
.btn-record:hover:not(:disabled){background:#ff1744}
.btn-record.recording{animation:pulse 1s infinite}
.btn-stop{background:#555;color:#fff}
.btn-stop:hover:not(:disabled){background:#777}
.btn-clear{background:transparent;color:#888;border:1px solid #555}
.btn-clear:hover:not(:disabled){background:#333;color:#fff}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
/* ---- 主体 ---- */
#main{display:flex;flex:1;overflow:hidden}
/* ---- 左侧面板 ---- */
#left-panel{width:260px;background:#1e1e1e;border-right:1px solid #333;padding:12px;overflow-y:auto;flex-shrink:0}
.panel-section{margin-bottom:16px}
.panel-section h3{font-size:13px;color:#888;border-bottom:1px solid #333;padding-bottom:4px;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px}
.joint-row{display:flex;justify-content:space-between;padding:3px 0;font-size:13px}
.joint-row .name{color:#bbb}
.joint-row .val{color:#fff;font-weight:bold;font-family:'Cascadia Code','Consolas',monospace}
.joint-row .val.warn{color:#ffd740}
.joint-row .val.danger{color:#ff5252}
/* ---- 右侧图表 ---- */
#charts{flex:1;display:flex;flex-direction:column;overflow:hidden}
.chart-row{flex:1;min-height:0}
/* ---- 底部告警 ---- */
#alerts-panel{height:120px;background:#1e1e1e;border-top:1px solid #333;overflow-y:auto;padding:6px 12px;flex-shrink:0}
.alert-row{font-size:12px;padding:2px 0;font-family:'Cascadia Code','Consolas',monospace}
.alert-row.warn{color:#ffd740}
.alert-row.error{color:#ff5252}
.alert-row.info{color:#4fc3f7}
/* ---- 无数据遮罩 ---- */
#no-data-overlay{display:flex;align-items:center;justify-content:center;color:#666;font-size:20px;position:absolute;top:40%;left:50%;transform:translate(-50%,-50%);pointer-events:none}
</style>
</head>
<body>
<div id="app">
  <!-- 顶栏 -->
  <div id="topbar">
    <span id="status-dot" class="status-dot red"></span>
    <h1>F1 VR 遥操</h1>
    <div class="stat-item">帧率: <span id="fps-val">--</span> fps</div>
    <div class="stat-item">已运行: <span id="elapsed-val">00:00</span></div>
    <div class="stat-item">总帧: <span id="frame-count-val">0</span></div>
    <div class="stat-item">右IK: <span id="right-ik-val">--</span></div>
    <div class="stat-item">左IK: <span id="left-ik-val">--</span></div>
    <div style="flex:1"></div>
    <button id="btn-record" class="btn btn-record" onclick="toggleRecord()">● 开始录制</button>
    <button id="btn-clear" class="btn btn-clear" onclick="clearAlerts()">清告警</button>
  </div>
  <!-- 主体 -->
  <div id="main">
    <div id="left-panel">
      <div class="panel-section">
        <h3>▼ 躯干 (3轴)</h3>
        <div id="joint-torso"></div>
      </div>
      <div class="panel-section">
        <h3>▼ 右臂 (7轴)</h3>
        <div id="joint-right"></div>
      </div>
      <div class="panel-section">
        <h3>▼ 左臂 (7轴)</h3>
        <div id="joint-left"></div>
      </div>
      <div class="panel-section">
        <h3>▼ 手腕位姿</h3>
        <div id="wrist-info" style="font-size:12px;color:#888;">等待数据...</div>
      </div>
    </div>
    <div id="charts">
      <div id="chart-torso" class="chart-row"></div>
      <div id="chart-right" class="chart-row"></div>
      <div id="chart-left" class="chart-row"></div>
    </div>
  </div>
  <!-- 告警 -->
  <div id="alerts-panel"></div>
</div>

<script>
// ========================================================================
// 全局状态
// ========================================================================
const JOINT_NAMES = [
  "lift","waist1","waist2",
  "J1_R","J2_R","J3_R","J4_R","J5_R","J6_R","J7_R",
  "J1_L","J2_L","J3_L","J4_L","J5_L","J6_L","J7_L"
];
const TORSO_IDX = [0,1,2];
const RIGHT_IDX = [3,4,5,6,7,8,9];
const LEFT_IDX  = [10,11,12,13,14,15,16];

let allFrames = [];          // 全量 [{ts, joints:[17], right_ik_ok, left_ik_ok}, ...]
let recording = false;
let startTime = null;
let alertBuffer = [];

// ========================================================================
// ECharts 初始化
// ========================================================================
function makeChart(domId, jointIndices, colorList, title) {
  const dom = document.getElementById(domId);
  const chart = echarts.init(dom, 'dark');
  const series = jointIndices.map((ji, i) => ({
    name: JOINT_NAMES[ji],
    type: 'line',
    showSymbol: false,
    smooth: true,
    lineStyle: { width: 1.5, color: colorList[i] },
    data: [],
    emphasis: { focus: 'series' },
  }));
  chart.setOption({
    title: { text: title, left: 8, top: 4, textStyle: { fontSize: 12, color: '#888' } },
    tooltip: { trigger: 'axis' },
    legend: {
      type: 'scroll', bottom: 0, textStyle: { fontSize: 10, color: '#aaa' },
      itemWidth: 14, itemHeight: 8,
    },
    grid: { left: 50, right: 16, top: 28, bottom: 28 },
    xAxis: { type: 'time', axisLabel: { fontSize: 10 }, splitLine: { show: false } },
    yAxis: {
      type: 'value', name: '°', axisLabel: { fontSize: 10 },
      splitLine: { lineStyle: { color: '#333', type: 'dashed' } },
      min: -180, max: 180,
    },
    series: series,
    animation: false,
  });
  return chart;
}

const COLORS_TORSO = ['#4fc3f7','#81d4fa','#b3e5fc'];
const COLORS_RIGHT = ['#ff5252','#ff8a50','#ffab40','#ffd740','#ffe57f','#ff6e40','#ff3d00'];
const COLORS_LEFT  = ['#448aff','#536dfe','#7c4dff','#b388ff','#40c4ff','#69f0ae','#00e5ff'];

const chartTorso = makeChart('chart-torso', TORSO_IDX, COLORS_TORSO, '躯干 (3轴)');
const chartRight = makeChart('chart-right', RIGHT_IDX, COLORS_RIGHT, '右臂 (7轴)');
const chartLeft  = makeChart('chart-left',  LEFT_IDX,  COLORS_LEFT,  '左臂 (7轴)');

window.addEventListener('resize', () => { chartTorso.resize(); chartRight.resize(); chartLeft.resize(); });

// ========================================================================
// 更新左侧数字面板
// ========================================================================
function updateJointPanel(containerId, indices) {
  const container = document.getElementById(containerId);
  if (!allFrames.length) { container.innerHTML = '<div style="color:#666;font-size:12px">等待数据...</div>'; return; }
  const latest = allFrames[allFrames.length - 1];
  const joints = latest.joints;
  let html = '';
  indices.forEach((ji, i) => {
    const val = joints[ji];
    const cls = Math.abs(val) > 150 ? 'danger' : (Math.abs(val) > 120 ? 'warn' : '');
    html += `<div class="joint-row"><span class="name">${JOINT_NAMES[ji]}</span><span class="val ${cls}">${val.toFixed(1)}°</span></div>`;
  });
  container.innerHTML = html;
}

function updateWristPanel() {
  const el = document.getElementById('wrist-info');
  if (!allFrames.length) { el.innerHTML = '等待数据...'; return; }
  const latest = allFrames[allFrames.length - 1];
  const rw = latest.right_wrist, lw = latest.left_wrist;
  el.innerHTML = `
    右手: ${rw ? rw.map(v=>v.toFixed(3)).join(', ') : '--'}<br>
    左手: ${lw ? lw.map(v=>v.toFixed(3)).join(', ') : '--'}<br>
    <span style="color:${latest.right_ik_ok?'#4caf50':'#f44336'}">右IK: ${latest.right_ik_ok?'OK':'FAIL'}</span> ·
    <span style="color:${latest.left_ik_ok?'#4caf50':'#f44336'}">左IK: ${latest.left_ik_ok?'OK':'FAIL'}</span>
  `;
}

// ========================================================================
// 更新曲线图（30秒滑动窗口）
// ========================================================================
let lastChartUpdate = 0;
function updateCharts() {
  const now = Date.now();
  if (now - lastChartUpdate < 200) return; // 每秒最多刷 5 次，省性能
  lastChartUpdate = now;

  const cutoff = now - 30_000; // 最近 30 秒
  const windowFrames = allFrames.filter(f => f._ts_ms >= cutoff);

  const extractSeries = (indices) => indices.map((ji, i) => ({
    name: JOINT_NAMES[ji],
    data: windowFrames.map(f => [f._ts_ms, f.joints[ji]]),
  }));

  chartTorso.setOption({ series: extractSeries(TORSO_IDX) });
  chartRight.setOption({ series: extractSeries(RIGHT_IDX) });
  chartLeft.setOption({  series: extractSeries(LEFT_IDX)  });
}

// ========================================================================
// 告警面板
// ========================================================================
function pushAlert(time, level, msg) {
  alertBuffer.push({ time, level, msg });
  if (alertBuffer.length > 50) alertBuffer.shift();
  renderAlerts();
}
function renderAlerts() {
  const el = document.getElementById('alerts-panel');
  el.innerHTML = alertBuffer.slice(-30).reverse().map(a =>
    `<div class="alert-row ${a.level==='warning'?'warn':a.level}">[${a.time}] ${a.msg}</div>`
  ).join('');
}
function clearAlerts() { alertBuffer = []; renderAlerts(); }

// ========================================================================
// 录制
// ========================================================================
function toggleRecord() {
  // 即时反馈，不等服务器
  const newState = !recording;
  setRecordingUI(newState);
  if (newState) {
    ws.send(JSON.stringify({ type: 'start_record' }));
  } else {
    ws.send(JSON.stringify({ type: 'stop_record' }));
  }
}
function setRecordingUI(isRecording) {
  recording = isRecording;
  const btn = document.getElementById('btn-record');
  if (isRecording) {
    btn.textContent = '■ 停止录制';
    btn.classList.add('recording');
  } else {
    btn.textContent = '● 开始录制';
    btn.classList.remove('recording');
  }
}

// ========================================================================
// 顶栏状态
// ========================================================================
let lastStatusUpdate = 0;
function updateStatusBar(status) {
  const now = Date.now();
  if (now - lastStatusUpdate < 500) return;
  lastStatusUpdate = now;

  document.getElementById('frame-count-val').textContent = status.frame_count || 0;
  document.getElementById('right-ik-val').textContent = `${status.right_ik_ok || 0}/${status.ik_total || 0}`;
  document.getElementById('left-ik-val').textContent = `${status.left_ik_ok || 0}/${status.ik_total || 0}`;

  // 状态灯
  const dot = document.getElementById('status-dot');
  const hasData = (status.frame_count || 0) > 0;
  dot.className = 'status-dot ' + (hasData ? 'green' : 'yellow');
  dot.title = hasData ? '数据正常' : '等待VR数据...';

  // 录制状态
  if (status.recording !== recording) setRecordingUI(status.recording);

  // 告警
  (status.alerts || []).forEach(a => pushAlert(a.time, a.level, a.msg));
}

// ========================================================================
// WebSocket
// ========================================================================
let ws = null;
let reconnectTimer = null;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onopen = () => {
    document.getElementById('status-dot').className = 'status-dot yellow';
    document.getElementById('status-dot').title = 'WebSocket已连接，等待VR数据...';
    if (reconnectTimer) { clearInterval(reconnectTimer); reconnectTimer = null; }
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === 'frame') {
      msg.data._ts_ms = msg.data._ts * 1000; // 时间戳转毫秒
      allFrames.push(msg.data);
      // 限制前端缓存（保留最近 120 秒，避免内存炸）
      const cutoff = Date.now() - 120_000;
      allFrames = allFrames.filter(f => f._ts_ms > cutoff);

      updateJointPanel('joint-torso', TORSO_IDX);
      updateJointPanel('joint-right', RIGHT_IDX);
      updateJointPanel('joint-left',  LEFT_IDX);
      updateWristPanel();
      updateCharts();

      // 帧率
      if (allFrames.length > 1) {
        const recent = allFrames.slice(-30);
        const dt = (recent[recent.length-1]._ts_ms - recent[0]._ts_ms) / 1000;
        if (dt > 0) document.getElementById('fps-val').textContent = (recent.length / dt).toFixed(0);
      }

      // 运行时间
      if (allFrames.length > 0 && startTime) {
        const elapsed = Math.floor((Date.now() / 1000) - startTime);
        const m = Math.floor(elapsed / 60), s = elapsed % 60;
        document.getElementById('elapsed-val').textContent =
          String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
      }
    }

    if (msg.type === 'status') {
      updateStatusBar(msg.data);
    }

    if (msg.type === 'recording_changed') {
      setRecordingUI(msg.recording);
    }

    if (msg.type === 'alert') {
      pushAlert(msg.time, msg.level, msg.msg);
    }
  };

  ws.onclose = () => {
    document.getElementById('status-dot').className = 'status-dot red';
    document.getElementById('status-dot').title = '已断开，3秒后自动重连...';
    if (!reconnectTimer) reconnectTimer = setInterval(connect, 3000);
  };

  ws.onerror = () => ws.close();
}

connect();
startTime = Date.now() / 1000;
</script>
</body>
</html>"""


# ============================================================================
# FastAPI 应用
# ============================================================================
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

app = FastAPI(title="F1 VR Monitor", docs_url=None, redoc_url=None)


def _get_state(request: Request):
    return request.app.state.dashboard


@app.get("/")
async def index():
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/api/status")
async def api_status(request: Request):
    return JSONResponse(_get_state(request).get_status())


@app.get("/api/history")
async def api_history(request: Request, window: int = 0):
    """获取历史数据。window=0 返回全量，window=N 返回最近 N 秒"""
    state = _get_state(request)
    if window > 0:
        cutoff = time.time() - window
        frames = [f for f in state.history if f.get("ts", 0) >= cutoff]
    else:
        frames = list(state.history)
    # 转成 JSON-safe 格式
    return JSONResponse([{
        "ts": f["ts"],
        "joints": f["joints"],
        "right_wrist": f.get("right_wrist"),
        "left_wrist": f.get("left_wrist"),
        "right_ik_ok": f.get("right_ik_ok"),
        "left_ik_ok": f.get("left_ik_ok"),
    } for f in frames])


@app.get("/api/joint_limits")
async def api_joint_limits(request: Request):
    state = _get_state(request)
    processor = getattr(state, "ik_processor", None)
    if processor:
        return JSONResponse({
            "names": processor.all_joint_names,
            "limits_deg": processor.joint_limits_deg,
        })
    return JSONResponse({"error": "IK 未启用"})


@app.post("/api/record/start")
async def api_record_start(request: Request):
    state = _get_state(request)
    if not state.recording:
        state.start_recording()
        return JSONResponse({"ok": True, "msg": "录制已开始"})
    return JSONResponse({"ok": False, "msg": "已经在录了"})


@app.post("/api/record/stop")
async def api_record_stop(request: Request):
    state = _get_state(request)
    if state.recording:
        frames = state.stop_recording()
        # 存 HDF5
        h5_path = _save_hdf5(frames, state)
        return JSONResponse({"ok": True, "frames": len(frames), "path": str(h5_path)})
    return JSONResponse({"ok": False, "msg": "没在录"})


def _save_hdf5(frames: list, state: DashboardState) -> Path:
    """把录制缓冲区的帧存成 HDF5"""
    import h5py

    output_dir = Path("./recordings")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("episode_*.h5"))
    next_idx = len(existing)
    h5_path = output_dir / f"episode_{next_idx:06d}.h5"

    n = len(frames)
    if n == 0:
        return h5_path

    timestamps = np.array([f["ts"] for f in frames], dtype=np.float32)
    joints = np.array([f["joints"] for f in frames], dtype=np.float32)
    right_wrists = np.array([f.get("right_wrist", [0]*3) for f in frames], dtype=np.float32)
    left_wrists = np.array([f.get("left_wrist", [0]*3) for f in frames], dtype=np.float32)

    with h5py.File(str(h5_path), "w") as f:
        f.create_dataset("timestamp", data=timestamps)
        f.create_dataset("observation.state", data=joints)
        f.create_dataset("action", data=joints)
        f.create_dataset("observation.right_hand.wrist_pose", data=right_wrists)
        f.create_dataset("observation.left_hand.wrist_pose", data=left_wrists)
        f.create_dataset("episode_index", data=np.zeros(n, dtype=np.int64))
        f.create_dataset("task_index", data=np.zeros(n, dtype=np.int64))
        f.create_dataset("index", data=np.arange(n, dtype=np.int64))
        f.attrs["total_frames"] = n
        f.attrs["fps"] = 30
        f.attrs["source"] = "VR_realtime_recording"
        f.attrs["recorded_at"] = datetime.now().isoformat()

    print(f"[REC] 已保存: {h5_path} ({n} 帧)")

    # 自动跑离线 IK 生成 LeRobot
    try:
        import subprocess
        import sys
        script = Path(__file__).parent / "07_offline_ik.py"
        if script.exists():
            print(f"[REC] 自动触发离线 IK 处理...")
            subprocess.Popen([
                sys.executable, str(script),
                "--input", str(h5_path),
                "--dataset", "f1_vr_v1",
                "--task", "recorded from dashboard",
            ])
    except Exception as e:
        print(f"[REC] 离线 IK 触发失败: {e}")

    return h5_path


# ============================================================================
# WebSocket —— 核心数据通道
# ============================================================================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state = app.state.dashboard
    state.clients.add(ws)

    # 把已有历史发过去（最近 30 秒），让新打开的页面能追上
    cutoff = time.time() - 30
    for frame in state.ring:
        if frame.get("ts", 0) >= cutoff:
            await _safe_send(ws, {"type": "frame", "data": frame})

    try:
        while True:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=30)
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "start_record":
                state.start_recording()
                await _safe_send(ws, {"type": "recording_changed", "recording": True})
            elif msg.get("type") == "stop_record":
                frames = state.stop_recording()
                await _safe_send(ws, {"type": "recording_changed", "recording": False})
                if frames:
                    h5_path = _save_hdf5(frames, state)
                    await _safe_send(ws, {
                        "type": "alert",
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "level": "info",
                        "msg": f"录制完成: {len(frames)} 帧 → {h5_path.name}",
                    })
            elif msg.get("type") == "ping":
                await _safe_send(ws, {"type": "pong"})
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        state.clients.discard(ws)


async def _safe_send(ws: WebSocket, msg: dict):
    try:
        await ws.send_json(msg)
    except Exception:
        pass


# ============================================================================
# 后台循环 —— UDP → IK → 广播
# ============================================================================
async def processing_loop(state: DashboardState, ik: IKProcessor | None, fps: int):
    """主循环：每 1/fps 秒跑一帧"""
    interval = 1.0 / fps
    last_frame_ts = time.time()
    last_status_broadcast = 0
    last_wrist = {"Left": None, "Right": None}  # 缓存上一次有效的手腕数据

    print(f"[LOOP] 开始处理，{fps} FPS")

    while True:
        await asyncio.sleep(interval * 0.5)  # 半间隔检查一次，减少延迟

        # 取 UDP 原始数据
        raw_batch = state.drain_raw()
        for raw in raw_batch:
            side = raw["side"]
            ptype = raw["type"]
            if ptype == "wrist":
                last_wrist[side] = raw["values"]

        # 按 FPS 节流
        now = time.time()
        if now - last_frame_ts < interval:
            continue
        last_frame_ts = now

        # 没有手部数据就跳过
        if last_wrist["Left"] is None and last_wrist["Right"] is None:
            # 每 2 秒广播一次状态（让前端知道还在等数据）
            if now - last_status_broadcast > 2:
                await _broadcast_status(state)
                last_status_broadcast = now
            continue

        ts = time.time()
        frame = {
            "ts": ts,
            "joints": None,
            "right_wrist": last_wrist.get("Right"),
            "left_wrist": last_wrist.get("Left"),
            "right_ik_ok": False,
            "left_ik_ok": False,
        }

        # IK
        if ik is not None and state.ik_enabled:
            result = ik.solve_frame(
                last_wrist.get("Right"),
                last_wrist.get("Left"),
                state.right_guess,
                state.left_guess,
            )
            frame["joints"] = result["joints"]
            frame["right_wrist"] = result["right_wrist"]
            frame["left_wrist"] = result["left_wrist"]
            frame["right_ik_ok"] = result["right_ik_ok"]
            frame["left_ik_ok"] = result["left_ik_ok"]
            state.right_guess = result["right_guess"]
            state.left_guess = result["left_guess"]
            state.ik_total += 1
            if result["right_ik_ok"]:
                state.right_ok_count += 1
            if result["left_ik_ok"]:
                state.left_ok_count += 1
            if not result["right_ik_ok"]:
                state.add_alert("error", f"右手 IK 失败 #{state.ik_total}")
            if not result["left_ik_ok"]:
                state.add_alert("error", f"左手 IK 失败 #{state.ik_total}")
        else:
            # 无 IK 模式：直接传手部位移当伪关节角（调试用）
            r = last_wrist.get("Right", [0]*3)
            l = last_wrist.get("Left", [0]*3)
            frame["joints"] = [r[0], r[1], r[2]] + [0]*7 + [l[0], l[1], l[2]] + [0]*7

        state.add_frame(frame)
        state.record_frame(frame)

        # 广播给所有客户端
        await _broadcast_frame(state, frame)

        # 每 1 秒广播一次状态
        if now - last_status_broadcast > 1:
            await _broadcast_status(state)
            last_status_broadcast = now


async def _broadcast_frame(state: DashboardState, frame: dict):
    if not state.clients:
        return
    payload = {"type": "frame", "data": frame}
    dead = set()
    for ws in state.clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    state.clients -= dead


async def _broadcast_status(state: DashboardState):
    if not state.clients:
        return
    payload = {"type": "status", "data": state.get_status()}
    dead = set()
    for ws in state.clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    state.clients -= dead


# ============================================================================
# 启动
# ============================================================================
def main():
    args = parser.parse_args()
    state = DashboardState(max_ring=args.fps * 30, max_history=args.max_history, alert_deg=args.alert_threshold)

    # IK 处理器
    ik = None
    if not args.no_ik:
        try:
            ik = IKProcessor()
            state.ik_enabled = True
        except Exception as e:
            print(f"[IK] 加载失败: {e}")
            print("[IK] 将用 --no-ik 模式运行（只显示手部原始数据）")
            state.ik_enabled = False

    # 挂到 app 上
    app.state.dashboard = state
    app.state.ik_processor = ik

    # 回放模式 vs 实时模式
    if args.replay:
        print(f"[REPLAY] 回放模式: {args.replay}")
        # TODO: 回放模式（后面再加）
        print("[REPLAY] 回放模式暂未实现，请使用实时模式")
        return

    # 启动 UDP 监听线程
    stop_event = threading.Event()
    udp_thread = threading.Thread(
        target=udp_listener,
        args=(args.port, args.host, state, stop_event),
        daemon=True,
    )
    udp_thread.start()

    # 把 IK processor 挂到 state 上（api 需要）
    state.ik_processor = ik

    # 启动后台处理循环（通过 router event，比 on_event/lifespan 兼容性好）
    processing_task = None

    async def on_startup():
        nonlocal processing_task
        processing_task = asyncio.create_task(processing_loop(state, ik, args.fps))

    async def on_shutdown():
        if processing_task:
            processing_task.cancel()
        stop_event.set()
        print("[MAIN] 服务关闭")

    app.router.add_event_handler("startup", on_startup)
    app.router.add_event_handler("shutdown", on_shutdown)

    print(f"""
+==========================================================+
|  F1 VR 遥操数据实时可视化平台                             |
|                                                          |
|  >> 浏览器打开: http://localhost:{args.web_port}              |
|                                                          |
|  UDP 端口: {args.port}                                        |
|  IK 状态:   {'[OK] 已加载' if ik else '[!] 未启用 (--no-ik)'}                |
|                                                          |
|  戴上 Quest 3，打开 Hand Tracking Streamer，              |
|  然后刷新浏览器页面就能看到实时曲线了。                     |
+==========================================================+
""")

    uvicorn.run(app, host="0.0.0.0", port=args.web_port, log_level="warning")


if __name__ == "__main__":
    main()
