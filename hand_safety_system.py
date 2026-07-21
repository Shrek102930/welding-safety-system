"""
HRC Welding — Hand Safety System v5.0
=====================================
基于 MediaPipe HandLandmarker（21 关键点）的手部安全感知系统。

架构：1 相机线程 + 1 推理线程 + 1 主线程 (GUI only)

  CameraThread ──Queue(maxsize=2)──▶ InferenceThread ──render_lock──▶ Main Thread
      │                                      │                            │
      │ 阻塞读取深度帧 (OK)                  ├── MediaPipe HandLandmarker  │
      │ 阻塞读取彩色帧 (OK)                   ├── SafetyEngine (手部距离)   └── cv2.imshow
      │                                      ├── SignalOutput              └── cv2.waitKey(1)
      │                                      └── Visualizer.draw

技术选型：
  - 手部检测：MediaPipe HandLandmarker（Google，21 关键点，含指尖）
  - 深度相机：OpenNI 2 + Gemini Pro
  - 通讯协议：TCP JSON Lines（tcp://0.0.0.0:9000，单行 JSON，\n 分隔）

启动方式：
  python hand_safety_system.py --visualize      # 真实相机 + 可视化
  python hand_safety_system.py --simulate        # 模拟模式（无相机）
  python hand_safety_system.py                   # 生产模式（仅信号）
"""
import sys
import os
import time
import math
import json
import ctypes
import socket
import threading
import queue
import argparse
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Tuple, List, Any

import numpy as np
import cv2
import openni2

# ═══════════════════════════════════════════════════════════════
# 路径 & 默认配置
# ═══════════════════════════════════════════════════════════════

BASE = Path(__file__).resolve().parent

# 焊枪尖端在相机坐标系中的默认偏移（mm）
WELD_TIP_OFFSET = (300.0, -200.0, 1000.0)  # X=右, Y=下, Z=前

# 安全区域阈值（米）
GREEN_THRESHOLD_M = 0.5   # > 0.5m = 绿色安全
YELLOW_THRESHOLD_M = 0.25 # ≤ 0.25m = 红色危险, 0.25~0.5m = 黄色减速

# 信号输出间隔（秒）
SIGNAL_INTERVAL_S = 1.0

# 推理跳帧间隔：每 N 帧才跑一次 MediaPipe（1 = 每帧跑，3 = 约 10fps）
INFERENCE_SKIP = 1

# 相机读帧超时（秒）：超过此时间无有效帧则触发自动重连
CAMERA_TIMEOUT_S = 15.0

# 状态滤波参数
ZONE_DEBOUNCE_FRAMES = 5     # 降权去抖动帧数（升权立即生效）
PERSON_DEBOUNCE_FRAMES = 1   # 人体出现/消失去抖动帧数
DIST_EMA_ALPHA = 0.3         # 距离指数平滑系数（0=无平滑, 1=只用新值）
ZONE_MEDIAN_WINDOW = 3       # 区域中值窗口大小（奇数，去单帧尖刺）

# 深度值采样核大小（中值滤波窗口）
DEPTH_KERNEL_SIZE = 11

# 焊头 ArUco 标记追踪参数（dynamic 模式 — 多面 3D 打印支架 + 4 个二维码）
ARUCO_DICT = cv2.aruco.DICT_6X6_250    # 6×6 编码，250 个唯一 ID
ARUCO_TARGET_IDS = {0, 1, 2, 3}         # 4 个面各一个码，任意一个被识别即可
ARUCO_CACHE_FRAMES = 30                 # 全部丢失后缓存最近位置帧数




# ═══════════════════════════════════════════════════════════════
# MediaPipe Hand 21 关键点（Google HandLandmarker 标准定义）
# ═══════════════════════════════════════════════════════════════

HAND_NAMES: List[str] = [
    "wrist",               # 0
    "thumb_cmc",           # 1
    "thumb_mcp",           # 2
    "thumb_ip",            # 3
    "thumb_tip",           # 4
    "index_finger_mcp",    # 5
    "index_finger_pip",    # 6
    "index_finger_dip",    # 7
    "index_finger_tip",    # 8
    "middle_finger_mcp",   # 9
    "middle_finger_pip",   # 10
    "middle_finger_dip",   # 11
    "middle_finger_tip",   # 12
    "ring_finger_mcp",     # 13
    "ring_finger_pip",     # 14
    "ring_finger_dip",     # 15
    "ring_finger_tip",     # 16
    "pinky_mcp",           # 17
    "pinky_pip",           # 18
    "pinky_dip",           # 19
    "pinky_tip",           # 20
]

# 关注的手部关键点（只取指尖 + 腕部用于距离计算）
HAND_LANDMARKS = {
    "wrist",
    "index_finger_tip", "middle_finger_tip",
    "ring_finger_tip", "pinky_tip", "thumb_tip",
}

# 21 点骨架连线（MediaPipe Hand Connections）
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # 拇指
    (0, 1), (1, 2), (2, 3), (3, 4),
    # 食指
    (0, 5), (5, 6), (6, 7), (7, 8),
    # 中指
    (0, 9), (9, 10), (10, 11), (11, 12),
    # 无名指
    (0, 13), (13, 14), (14, 15), (15, 16),
    # 小指
    (0, 17), (17, 18), (18, 19), (19, 20),
    # 掌间连线
    (5, 9), (9, 13), (13, 17),
]


# ═══════════════════════════════════════════════════════════════
# Camera — openni2 封装（仅在 CameraThread 中调用）
# ═══════════════════════════════════════════════════════════════


class Camera:
    """OpenNI 2 深度/彩色相机（openni2 包装）。

    所有方法都是同步阻塞的，因此只能在独立线程中调用。
    """

    def __init__(self) -> None:
        self._dev: Any = None
        self._depth_stream: Any = None
        self._color_stream: Any = None
        self._color_cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """初始化 OpenNI、打开设备和流。成功返回 True。"""
        try:
            openni2.initialize()
            self._dev = openni2.Device.open_any()
        except Exception:
            return False

        # deeply型计算（必须）
        self._depth_stream = self._dev.create_stream(openni2.SENSOR_DEPTH)
        self._depth_stream.start()

        # openni2 无法解码 MJPEG 彩色流，改用 UVC（已确认设备号为 1）
        self._color_stream = None
        self._color_cap = self._try_open_color_uvc()
        return True

    def _try_open_color_uvc(self) -> Optional[cv2.VideoCapture]:
        """直接打开 Orbbec 彩色流（DShow 设备 1，已通过诊断确认）。"""
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[WARN] Color UVC device 1 not opened", file=sys.stderr)
            return None
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if not ret or frame is None or frame.mean() < 1.0:
            print("[WARN] Color UVC device 1 black frame", file=sys.stderr)
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"[OK] Color UVC opened: device 1 shape={frame.shape} mean={frame.mean():.1f}",
              file=sys.stderr)
        return cap

    def read_depth(self) -> Optional[np.ndarray]:
        """阻塞读取一帧深度图。返回 uint16 (H, W) 或 None。"""
        frame = self._depth_stream.read_frame()
        if frame is None:
            return None
        h, w = frame.height, frame.width
        data = frame.get_buffer_as_uint8()
        copy_data = bytes(data)  # 纯 Python bytes，完全脱钩
        arr = np.frombuffer(copy_data, dtype=np.uint16).reshape((h, w))
        return arr

    def read_color(self) -> Optional[np.ndarray]:
        if self._color_cap is None:
            return None
        ret, frame = self._color_cap.read()
        return frame if ret and frame is not None else None

    def has_color(self) -> bool:
        return self._color_cap is not None

    def close(self) -> None:
        if self._color_cap is not None:
            self._color_cap.release()
            self._color_cap = None
        try:
            openni2.unload()
        except Exception:
            pass

    def reconnect(self) -> bool:
        """关闭并重新打开相机（USB 断开后自动恢复）。"""
        self.close()
        time.sleep(1.0)
        return self.open()


# ═══════════════════════════════════════════════════════════════
# CameraThread — 唯一的子线程，阻塞读帧


class CameraThread(threading.Thread):
    """独占线程，持续从深度相机取帧。

    使用 15 秒超时检测 USB 断开。通过 .timed_out 标志通知推理线程。
    """

    def __init__(self, camera: Camera) -> None:
        super().__init__(daemon=True, name="Camera")
        self._cam = camera
        self._running = False
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.ready = threading.Event()
        self.timed_out = False        # USB 断开标志（推理线程读完重置）
        self._last_frame_t = 0.0

    def run(self) -> None:
        self._running = True
        has_color = self._cam.has_color()
        self.ready.set()
        self._last_frame_t = time.time()

        while self._running:
            depth = self._cam.read_depth()
            if depth is None:
                # 超时检测
                if time.time() - self._last_frame_t > CAMERA_TIMEOUT_S:
                    self.timed_out = True
                time.sleep(0.5)
                continue

            self._last_frame_t = time.time()
            self.timed_out = False

            color = self._cam.read_color() if has_color else None

            # 只保留最新帧（丢弃旧帧，防止主线程积压）
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put((depth, color))

    def stop(self) -> None:
        self._running = False
        self.join(timeout=2.0)


# ═══════════════════════════════════════════════════════════════
# 3D 坐标计算工具
# ═══════════════════════════════════════════════════════════════


class CameraIntrinsics:
    """相机内参（默认值匹配 Gemini Pro 640x480 深度图）。"""
    fx: float = 580.0
    fy: float = 580.0
    cx: float = 320.0
    cy: float = 240.0


def pixel_to_3d(u: float, v: float, z_mm: float,
                K: CameraIntrinsics) -> Tuple[float, float, float]:
    """像素坐标 + 深度值 → 3D 相机坐标（mm）。"""
    if z_mm <= 0:
        return (float("nan"), float("nan"), float("nan"))
    z = float(z_mm)
    x = (u - K.cx) * z / K.fx
    y = (v - K.cy) * z / K.fy
    return (x, y, z)


def median_depth(depth: np.ndarray, u: int, v: int, kernel: int) -> float:
    """取像素点周围 kernel×kernel 区域的深度中值（去噪）。"""
    h, w = depth.shape
    half = kernel // 2
    r1 = max(0, v - half)
    r2 = min(h, v + half + 1)
    c1 = max(0, u - half)
    c2 = min(w, u + half + 1)
    patch = depth[r1:r2, c1:c2]
    valid = patch[(patch > 0) & (patch < 10000)]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def dist_m(a: Tuple[float, float, float],
           b: Tuple[float, float, float]) -> float:
    """两个 3D 点之间的欧氏距离（m）。坐标单位：mm。"""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz) / 1000.0


def depth_to_hand_input(depth: np.ndarray) -> np.ndarray:
    """深度图 → 灰度图（近处亮、远处暗），适合 HandLandmarker 输入。

    手部靠近相机时呈现高亮，背景变暗，边缘清晰。
    """
    mask = (depth > 0) & (depth < 8000)
    if not np.any(mask):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    d = depth.astype(np.float32)
    lo = float(np.percentile(d[mask], 3))
    hi = float(np.percentile(d[mask], 97))
    if hi - lo < 300:
        hi = lo + 300

    norm = np.clip(d, lo, hi)
    norm = ((norm - lo) / (hi - lo) * 255).astype(np.uint8)
    norm[~mask] = 0
    norm = 255 - norm  # 反转：近处亮（手→亮），远处暗（背景→黑）

    # 强 CLAHE 增强局部对比度
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(norm)

    # 轻微高斯平滑去噪
    smoothed = cv2.GaussianBlur(enhanced, (3, 3), 0.5)

    # 转 3 通道 RGB（模型要求）
    return cv2.cvtColor(smoothed, cv2.COLOR_GRAY2RGB)


def _depth_heatmap_for_vis(depth: np.ndarray) -> np.ndarray:
    """深度图 → BGR 热力图（JET 彩色），仅用于可视化。"""
    mask = (depth > 0) & (depth < 8000)
    if not np.any(mask):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    d = depth.astype(np.float32)
    lo = float(np.percentile(d[mask], 3))
    hi = float(np.percentile(d[mask], 97))
    if hi - lo < 500:
        hi = lo + 500
    norm = np.clip(d, lo, hi)
    norm = ((norm - lo) / (hi - lo) * 255).astype(np.uint8)
    norm[~mask] = 0
    norm = 255 - norm
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)
    return cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)


# ═══════════════════════════════════════════════════════════════
# TipTracker — 焊头定位（固定偏移 / ArUco 二维码动态追踪）
# ═══════════════════════════════════════════════════════════════


class TipTracker:
    """焊头位置追踪器。

    两种模式：
      - fixed:  固定偏移（--offset-x/y/z），不需要任何标记
      - dynamic: 追踪焊头上贴的 **ArUco 二维码标记**（打印、裁剪、贴上即可）
                 OpenCV aruco 模块检测，亚像素角点精度，唯一 ID 校验（绝对不会认错）

    在推理线程中使用，每帧检测一次。
    """

    def __init__(self, default_tip: Tuple[float, float, float],
                 mode: str = "fixed") -> None:
        self.default_tip = default_tip
        self.mode = mode           # "fixed" | "dynamic"
        self._last_valid: Optional[Tuple[float, float, float]] = None
        self._last_frame = 0
        self._frame_count = 0

        # ArUco 字典 + 检测器
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(
            self._aruco_dict, self._aruco_params)

        # 可视化用状态
        self.latest_led_pos: Optional[Tuple[int, int]] = None  # (u, v) 像素
        self.marker_pixel_radius: float = 12.0  # 二维码在屏幕上的视觉半径（px）
        self.latest_corners: Optional[np.ndarray] = None  # 最近检测到的角点，供可视化绘制边框
        self.latest_ids: Optional[np.ndarray] = None      # 最近检测到的 ID

    def detect(self, color: Optional[np.ndarray],
               depth: np.ndarray) -> Tuple[float, float, float]:
        """返回焊头 3D 坐标 (mm)。fixed 模式直接返回 default_tip。"""
        if self.mode == "fixed" or color is None:
            return self.default_tip
        return self._detect_marker(color, depth)

    def detect_with_vis(
        self, color: Optional[np.ndarray], depth: np.ndarray
    ) -> Tuple[Tuple[float, float, float], bool, Optional[Tuple[int, int]], float]:
        """返回 (tip_xyz, detected, led_pixel_pos, marker_pixel_radius)。供 Visualizer 渲染。"""
        if self.mode == "fixed" or color is None:
            self.latest_led_pos = None
            self.latest_corners = None
            self.latest_ids = None
            return self.default_tip, False, None, 12.0

        tip = self._detect_marker(color, depth)
        detected = (self.latest_led_pos is not None)
        return tip, detected, self.latest_led_pos, self.marker_pixel_radius

    def _detect_marker(
        self, bgr: np.ndarray, depth: np.ndarray
    ) -> Tuple[float, float, float]:
        """检测焊头上的多面 ArUco 二维码 → 取深度 → 3D 坐标。

        支架 4 个面各有 ID=0~3，任意一个被识别即可定位。
        同时检测到多个时取面积最大的（即最正对相机的那个面）。
        """
        self._frame_count += 1
        self.latest_led_pos = None
        dh, dw = depth.shape
        K = CameraIntrinsics()

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 检测所有 ArUco 标记
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        if ids is None:
            if self._frame_count % 30 == 0:
                print(f"[ArUco] No markers detected (frame {self._frame_count})",
                      file=sys.stderr)
            self.latest_corners = None
            self.latest_ids = None
            return self._fallback()

        # 只保留目标 ID
        mask = np.isin(ids.flatten(), list(ARUCO_TARGET_IDS))
        if not mask.any():
            if self._frame_count % 30 == 0:
                print(f"[ArUco] Detected {len(ids)} markers but none in target IDs "
                      f"{ARUCO_TARGET_IDS}", file=sys.stderr)
            self.latest_corners = None
            self.latest_ids = None
            return self._fallback()

        # 过滤后保留的角点和 ID
        filtered_corners = [c for i, c in enumerate(corners) if mask[i]]
        filtered_ids = ids.flatten()[mask]
        self.latest_corners = filtered_corners
        self.latest_ids = filtered_ids

        # 从目标 ID 中挑面积最大的（最正对相机的面）
        best_idx = -1
        best_area = 0
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in ARUCO_TARGET_IDS:
                # 用角点围成的四边形面积衡量可见大小
                pts = corners[i][0]
                a = cv2.contourArea(pts)
                if a > best_area:
                    best_area = a
                    best_idx = i

        if best_idx < 0:
            return self._fallback()

        # 取 4 个角点中心作为像素坐标
        pts = corners[best_idx][0]
        cu = int(np.clip(np.mean(pts[:, 0]), 0, dw - 1))
        cv = int(np.clip(np.mean(pts[:, 1]), 0, dh - 1))

        # 计算二维码视觉半径（角点到中心的平均距离），用于绘制焊头图标大小
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        radii = [np.hypot(p[0] - cx, p[1] - cy) for p in pts]
        self.marker_pixel_radius = max(8.0, min(40.0, np.mean(radii)))

        self.latest_led_pos = (cu, cv)

        if self._frame_count % 30 == 0:
            print(f"[ArUco] DETECTED ID={ids.flatten()[best_idx]} "
                  f"pos=({cu},{cv}) area={best_area:.0f} "
                  f"radius={self.marker_pixel_radius:.1f}px",
                  file=sys.stderr)

        # 读深度（中值滤波去噪）
        z_mm = median_depth(depth, cu, cv, DEPTH_KERNEL_SIZE)
        if 0 < z_mm < 10000:
            xyz = pixel_to_3d(cu, cv, z_mm, K)
            self._last_valid = xyz
            self._last_frame = self._frame_count
            return xyz

        return self._fallback()

    def _fallback(self) -> Tuple[float, float, float]:
        """缓存 → 默认偏移。"""
        if self._last_valid is not None:
            age = self._frame_count - self._last_frame
            if age <= ARUCO_CACHE_FRAMES:
                return self._last_valid
        return self.default_tip

    def switch_mode(self, mode: str) -> None:
        """运行时切换 fixed / dynamic。"""
        self.mode = mode


# ═══════════════════════════════════════════════════════════════
# HandDetector — MediaPipe HandLandmarker（21 关键点）
# ═══════════════════════════════════════════════════════════════

HAND_MODEL_PATH = str(BASE / "hand_landmarker.task")


class HandDetector:
    """Google MediaPipe HandLandmarker 封装。

    21 个手部关键点（含指尖）。有彩色流用彩色，没有则用深度热力图。
    只检测手，不检测全身，速度快且不会把墙壁椅子误识别为人体。
    """

    def __init__(self) -> None:
        self._landmarker: Any = None
        self._mp: Any = None
        # 时域平滑缓存
        self._smooth_cache: Dict[str, Tuple[float, float, float]] = {}
        self._smooth_alpha = 0.4
        # 手部存在去抖动
        self._hand_present_ctr = 0
        self._hand_absent_ctr = 0
        self._hand_confirmed = False

    def _smooth_coord(self, key: str, xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if key not in self._smooth_cache:
            self._smooth_cache[key] = xyz
            return xyz
        prev = self._smooth_cache[key]
        smoothed = (
            self._smooth_alpha * xyz[0] + (1 - self._smooth_alpha) * prev[0],
            self._smooth_alpha * xyz[1] + (1 - self._smooth_alpha) * prev[1],
            self._smooth_alpha * xyz[2] + (1 - self._smooth_alpha) * prev[2],
        )
        self._smooth_cache[key] = smoothed
        return smoothed

    def _init(self) -> None:
        if self._landmarker is not None:
            return
        import importlib
        vision = importlib.import_module("mediapipe.tasks.python.vision")
        base_mod = importlib.import_module(
            "mediapipe.tasks.python.core.base_options"
        )
        self._mp = importlib.import_module("mediapipe")

        opts = vision.HandLandmarkerOptions(
            base_options=base_mod.BaseOptions(
                model_asset_path=HAND_MODEL_PATH
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opts)

    def detect(self, color: Optional[np.ndarray], depth: np.ndarray
               ) -> Tuple[Optional[Any], Dict[str, Dict[str, Tuple[float, float, float]]], Optional[np.ndarray]]:
        """检测手部关键点，返回 (mediapipe_result, hands_3d, bg_heatmap)。

        hands_3d 格式: {"hand_0": {"index_finger_tip": (x,y,z), ...}}
        """
        self._init()
        dh, dw = depth.shape
        K = CameraIntrinsics()
        mp = self._mp

        bg_heatmap: Optional[np.ndarray] = None

        if color is not None:
            # CLAHE 增强工业光照下的对比度
            lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
        else:
            # 无彩色相机：用深度灰度图替代（手→亮，背景→暗）
            hand_rgb = depth_to_hand_input(depth)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(hand_rgb),
            )
            # 保留深度热力图给可视化用
            bg_heatmap = _depth_heatmap_for_vis(depth)

        result = self._landmarker.detect(mp_image)

        # 手部存在去抖动
        has_hand = bool(result and result.hand_landmarks)
        if has_hand:
            self._hand_present_ctr += 1
            self._hand_absent_ctr = 0
        else:
            self._hand_absent_ctr += 1
            self._hand_present_ctr = 0
        if self._hand_present_ctr >= PERSON_DEBOUNCE_FRAMES:
            self._hand_confirmed = True
        elif self._hand_absent_ctr >= PERSON_DEBOUNCE_FRAMES:
            self._hand_confirmed = False

        hands_3d: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
        if result and result.hand_landmarks and self._hand_confirmed:
            for hi, hand_lms in enumerate(result.hand_landmarks):
                pts: Dict[str, Tuple[float, float, float]] = {}
                for idx, lm in enumerate(hand_lms):
                    if idx >= len(HAND_NAMES):
                        break
                    u = int(np.clip(lm.x * (dw - 1), 0, dw - 1))
                    color_h = color.shape[0] if color is not None else dh
                    v = int(np.clip(lm.y * color_h / max(dh, 1) * (dh - 1), 0, dh - 1))
                    z_mm = median_depth(depth, u, v, DEPTH_KERNEL_SIZE)
                    if z_mm > 0:
                        xyz = pixel_to_3d(u, v, z_mm, K)
                        if not math.isnan(xyz[0]):
                            key_name = HAND_NAMES[idx]
                            smoothed = self._smooth_coord(f"hand_{hi}_{key_name}", xyz)
                            pts[key_name] = smoothed
                if pts:
                    hands_3d[f"hand_{hi}"] = pts

        return result, hands_3d, bg_heatmap

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


# ═══════════════════════════════════════════════════════════════
# SafetyEngine — 风险评估与决策
# ═══════════════════════════════════════════════════════════════


class SafetyEngine:
    """安全评估引擎。

    计算人手到焊枪的最近距离，结合历史轨迹推断意图，
    输出风险区域和技能指令。

    内置三段状态滤波：
      - 距离 EMA 平滑（减少单帧噪声尖刺）
      - 区域去抖动（降权需 N 帧确认，升权立即触发）
      - 人体去抖动（出现/消失各需 N 帧确认，防止 MediaPipe 误检丢失）
    """

    def __init__(self, tip_xyz: Tuple[float, float, float],
                 green_m: float = GREEN_THRESHOLD_M,
                 yellow_m: float = YELLOW_THRESHOLD_M) -> None:
        self.tip = tip_xyz          # 焊枪尖端 3D 坐标 (mm)，默认偏移（fixed 模式）
        self.green = green_m        # 绿色阈值 (m)
        self.yellow = yellow_m      # 红色阈值 (m)
        self._history: deque = deque(maxlen=30)  # 距离历史
        self._prev_dist: Optional[float] = None
        self._prev_time: float = time.time()
        self._last_zone: Optional[str] = None

        # 区域去抖动状态
        self._zone_debounce_ctr: int = 0
        self._candidate_zone: Optional[str] = None

        # 人体去抖动状态
        self._person_present_ctr: int = 0   # 有人帧计数
        self._person_absent_ctr: int = 0    # 无人帧计数
        self._was_person_present: bool = False

        # 距离 EMA 状态
        self._dist_ema: Optional[float] = None

        # 区域中值滤波窗（消除单帧尖刺）
        self._zone_mid_hist: List[str] = []

    def evaluate(self, hands_3d: dict,
                 tip_xyz: Optional[Tuple[float, float, float]] = None) -> dict:
        """评估当前帧的安全状态。

        只计算 HAND_LANDMARKS 中的关键点到焊枪的距离（指尖 + 腕部）。

        Args:
            hands_3d: {"hand_0": {"index_finger_tip": (x,y,z), ...}}
            tip_xyz: 动态焊头坐标 (mm)；None 时使用 self.tip（默认偏移）

        Returns:
            {
              "zone": str,              # green / yellow / red
              "risk_distance_m": float,  # 最近距离 (m)，无人时为 None
              "closest_hand": str,       # 最近的人 ("person_0"/"none")
              "closest_landmark": str,   # 最近的关键点名
              "approach_speed_ms": float,# 接近速度 (m/s)
              "intent": str,
              "skill_command": str|None
            }
        """
        now = time.time()
        dt = max(now - self._prev_time, 0.01)
        self._prev_time = now

        # 确定焊头位置：优先动态 LED，回退默认偏移
        actual_tip = tip_xyz if tip_xyz is not None else self.tip

        # ── 手部去抖动 ──
        has_raw = bool(hands_3d)
        if has_raw:
            self._person_present_ctr += 1
            self._person_absent_ctr = 0
        else:
            self._person_absent_ctr += 1
            self._person_present_ctr = 0

        # 只有连续 N 帧满足条件才切换状态
        if self._person_present_ctr >= PERSON_DEBOUNCE_FRAMES:
            person_confirmed = True
        elif self._person_absent_ctr >= PERSON_DEBOUNCE_FRAMES:
            person_confirmed = False
        else:
            # 去抖动期间保持上一次状态
            person_confirmed = self._was_person_present
        self._was_person_present = person_confirmed

        # 无人 → 直接安全，清空去抖动锁
        if not person_confirmed:
            self._dist_ema = None
            self._prev_dist = None
            self._history.clear()
            self._zone_debounce_ctr = 0
            self._candidate_zone = None
            self._last_zone = "green"
            return {
                "zone": "green",
                "risk_distance_m": None,
                "closest_hand": "none",
                "closest_landmark": "none",
                "approach_speed_ms": 0.0,
                "intent": "no_hands",
                "skill_command": None,
            }

        # 找最近的手部关键点（只关注指尖 + 腕部）
        min_dist = float("inf")
        closest_hand = "none"
        closest_landmark = "none"
        for hand_id, pts in hands_3d.items():
            for lm_name, xyz in pts.items():
                if lm_name not in HAND_LANDMARKS:
                    continue   # 忽略手指关节，只取指尖 + 腕部
                d = dist_m(xyz, actual_tip)
                if d < min_dist:
                    min_dist = d
                    closest_hand = hand_id
                    closest_landmark = lm_name

        # ── 距离 EMA 平滑 ──
        if self._dist_ema is None:
            self._dist_ema = min_dist
        else:
            self._dist_ema = (DIST_EMA_ALPHA * min_dist +
                              (1 - DIST_EMA_ALPHA) * self._dist_ema)
        smoothed_dist = self._dist_ema

        # 接近速度（基于平滑后距离）
        speed = 0.0
        if self._prev_dist is not None:
            speed = (self._prev_dist - smoothed_dist) / dt
        self._prev_dist = smoothed_dist
        self._history.append(smoothed_dist)

        # ── 区域判定 + 中值滤波去尖刺 + 去抖动 ──
        if smoothed_dist <= self.yellow:
            raw_zone = "red"
        elif smoothed_dist <= self.green:
            raw_zone = "yellow"
        else:
            raw_zone = "green"

        # 中值滤波：把 raw_zone 推入窗口，取出现次数最多的作为输出
        self._zone_mid_hist.append(raw_zone)
        if len(self._zone_mid_hist) > ZONE_MEDIAN_WINDOW:
            self._zone_mid_hist.pop(0)
        # majority vote（窗口满或不满都取多数，不满时取最新）
        if len(self._zone_mid_hist) == ZONE_MEDIAN_WINDOW:
            votes = sorted(self._zone_mid_hist, key=lambda z: _zone_level(z))
            filtered_zone = votes[len(votes) // 2]  # 中值
        else:
            filtered_zone = raw_zone

        zone = self._apply_zone_debounce(filtered_zone)
        self._last_zone = zone

        # 意图推断
        intent = self._infer_intent()

        # 技能指令
        skill = None
        if zone == "red":
            skill = "stop_welding"
        elif zone == "yellow":
            skill = "slow_down"

        return {
            "zone": zone,
            "risk_distance_m": round(smoothed_dist, 3),
            "closest_hand": closest_hand,
            "closest_landmark": closest_landmark,
            "approach_speed_ms": round(speed, 3),
            "intent": intent,
            "skill_command": skill,
        }

    def _apply_zone_debounce(self, raw_zone: str) -> str:
        """区域去抖动：升权立即，降权需 ZONE_DEBOUNCE_FRAMES 帧确认。"""
        target = raw_zone

        if target == self._candidate_zone:
            # 同一个候选 → 计数，但不超过上限
            if self._zone_debounce_ctr <= ZONE_DEBOUNCE_FRAMES:
                self._zone_debounce_ctr += 1
        else:
            # 候选变了
            prev = self._candidate_zone or "green"
            # 升权（green→yellow, green→red, yellow→red）：立即
            if _zone_level(target) > _zone_level(prev):
                self._candidate_zone = target
                self._zone_debounce_ctr = ZONE_DEBOUNCE_FRAMES
            else:
                # 降权 → 重置计数
                self._candidate_zone = target
                self._zone_debounce_ctr = 1

        if self._zone_debounce_ctr >= ZONE_DEBOUNCE_FRAMES:
            return self._candidate_zone
        # 去抖动未完成 → 保持上一次确认的 zone
        return self._last_zone or "green"

    def _infer_intent(self) -> str:
        """基于 5 帧历史距离推断意图。"""
        h = [x for x in self._history if x is not None]
        if len(h) < 5:
            return "unknown"

        last5 = h[-5:]
        if all(x <= self.yellow for x in last5):
            return "in_red_zone"
        elif last5[-1] < last5[0] - 0.1:
            return "approaching"
        elif last5[-1] > last5[0] + 0.1:
            return "retreating"
        elif all(x > self.green for x in last5):
            return "passing_safely"
        return "unknown"

    @property
    def last_zone(self) -> Optional[str]:
        return self._last_zone


def _zone_level(zone: str) -> int:
    """区域危险等级映射：green=0, yellow=1, red=2。"""
    return {"green": 0, "yellow": 1, "red": 2}.get(zone, -1)


# ═══════════════════════════════════════════════════════════════
# TCPServer — JSON 信号广播
# ═══════════════════════════════════════════════════════════════


class TCPServer:
    """TCP 服务端，向硬件组广播 JSON 信号。

    非阻塞 accept / sendall，支持多客户端连接。
    """

    def __init__(self, port: int = 9000) -> None:
        self._sock: Optional[socket.socket] = None
        self._clients: List[socket.socket] = []
        self._port = port

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", self._port))
        self._sock.listen(5)
        self._sock.setblocking(False)

    def accept(self) -> None:
        """非阻塞接受新客户端连接。"""
        if self._sock is None:
            return
        try:
            client, addr = self._sock.accept()
            client.setblocking(False)
            self._clients.append(client)
            print(f"[TCP] Client connected: {addr}", file=sys.stderr)
        except BlockingIOError:
            pass

    def broadcast(self, line: str) -> None:
        """向所有已连接客户端广播一行 JSON。"""
        data = (line + "\n").encode("utf-8")
        dead: List[socket.socket] = []
        for c in self._clients:
            try:
                c.sendall(data)
            except Exception:
                dead.append(c)
        for c in dead:
            self._clients.remove(c)
            try:
                c.close()
            except Exception:
                pass

    def stop(self) -> None:
        for c in self._clients:
            try:
                c.close()
            except Exception:
                pass
        self._clients.clear()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


# ═══════════════════════════════════════════════════════════════
# SignalOutput — 信号格式化与输出
# ═══════════════════════════════════════════════════════════════


class SignalOutput:
    """输出 JSON 信号到 stdout / 文件 / TCP。"""

    def __init__(self, out_file: Optional[str] = None,
                 tcp: Optional[TCPServer] = None) -> None:
        self._out = open(out_file, "w", encoding="utf-8") if out_file else sys.stdout
        self._tcp = tcp

    def emit(self, metrics: dict) -> None:
        """格式化并发送一条安全信号。"""
        d = metrics["risk_distance_m"]
        hp = "none"
        if metrics["closest_hand"] != "none" and metrics["closest_landmark"] != "none":
            hp = f"{metrics['closest_hand']}_{metrics['closest_landmark']}"

        record = {
            "timestamp_ms": int(time.time() * 1000),
            "zone": metrics["zone"],
            "risk_distance_m": round(d, 3) if d is not None else None,
            "approach_speed_ms": metrics["approach_speed_ms"],
            "hand_part": hp,
            "intent": metrics["intent"],
            "skill_command": metrics["skill_command"],
            "sensor_uncertainty": 0.15,
        }
        line = json.dumps(record, ensure_ascii=False)
        self._out.write(line + "\n")
        self._out.flush()

        if self._tcp is not None:
            self._tcp.accept()
            self._tcp.broadcast(line)

    def close(self) -> None:
        if self._out is not sys.stdout:
            self._out.close()
        if self._tcp is not None:
            self._tcp.stop()


# ═══════════════════════════════════════════════════════════════
# Visualizer — OpenCV 可视化渲染
# ═══════════════════════════════════════════════════════════════

ZONE_COLORS = {
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
}


class Visualizer:
    """OpenCV 可视化窗口渲染。

    显示：深度热力图背景 + 人体骨架 + 焊枪位置 + 距离连线 + 信息面板。
    """

    def __init__(self) -> None:
        self._fps_start = time.time()
        self._fps_count = 0
        self._fps = 0.0

    def draw(self, depth: np.ndarray, color: Optional[np.ndarray],
             result: Any, metrics: dict,
             tip_xyz: Tuple[float, float, float],
             bg_heatmap: Optional[np.ndarray] = None,
             tip_mode: str = "fixed",
             led_pos: Optional[Tuple[int, int]] = None,
             led_detected: bool = False,
             marker_radius: float = 12.0,
             aruco_corners: Optional[List[np.ndarray]] = None,
             aruco_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """渲染一帧可视化图像。

        Args:
            depth: uint16 深度图
            color: BGR 彩色图（可为 None）
            result: MediaPipe 检测结果
            metrics: SafetyEngine.evaluate() 返回的安全评估
            tip_xyz: 焊枪尖端 3D 坐标
            bg_heatmap: 预计算好的 BGR 热力图背景（避免重复算）
        """
        dh, dw = depth.shape
        K = CameraIntrinsics()

        # ── FPS 计算 ──
        self._fps_count += 1
        now = time.time()
        if now - self._fps_start >= 1.0:
            self._fps = self._fps_count / (now - self._fps_start)
            self._fps_count = 0
            self._fps_start = now

        zone = metrics["zone"]
        zone_color = ZONE_COLORS.get(zone, (0, 255, 0))
        no_person = (metrics["closest_hand"] == "none")

        # ── 背景：彩色图优先，resize 到深度图尺寸 ──
        if color is not None:
            bg = cv2.resize(color, (dw, dh), interpolation=cv2.INTER_LINEAR)
        elif bg_heatmap is not None:
            bg = bg_heatmap
        else:
            bg = _depth_heatmap_for_vis(depth)

        # ── 顶部区域状态条 ──
        bar_h = 24
        bar_colors = {
            "green": (0, 180, 0),
            "yellow": (0, 200, 200),
            "red": (0, 0, 200),
        }
        bar_c = bar_colors.get(zone, (60, 60, 60))
        cv2.rectangle(bg, (0, 0), (dw, bar_h), bar_c, -1)
        if no_person:
            status_txt = "NO PERSON — SAFE"
            txt_c = (200, 200, 200)
        else:
            status_txt = f"ZONE: {zone.upper()} — {metrics.get('skill_command') or 'NONE'}"
            txt_c = (255, 255, 255)
        cv2.putText(bg, status_txt, (10, bar_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt_c, 1, cv2.LINE_AA)
        # FPS
        fps_txt = f"{self._fps:.1f} fps"
        tw = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        cv2.putText(bg, fps_txt, (dw - tw - 10, bar_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # ── ArUco 标记边框（彩色帧 640×480 → 深度帧 640×400 需缩放 Y） ──
        if aruco_corners is not None and aruco_ids is not None:
            for i, corners_4 in enumerate(aruco_corners):
                pts = corners_4[0].copy()  # 4×2
                # 缩放 Y 坐标：480→400
                pts[:, 1] = pts[:, 1] * dh / 480.0
                pts = pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(bg, [pts], True, (255, 255, 0), 2, cv2.LINE_AA)
                # 显示 ID
                cx = int(np.mean(pts[:, 0, 0]))
                cy = int(np.mean(pts[:, 0, 1]))
                cv2.putText(bg, f"ArUco ID={aruco_ids[i]}", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # ── LED 亮区标记（坐标从 640×480 缩放到 640×400） ──
        if led_pos is not None:
            lx, ly = led_pos
            ly = int(ly * dh / 480.0)  # 彩色 Y → 深度 Y
            led_color = (0, 255, 255) if led_detected else (100, 100, 100)
            cv2.circle(bg, (lx, ly), 10, led_color, 2, cv2.LINE_AA)
            cv2.circle(bg, (lx, ly), 3, led_color, -1, cv2.LINE_AA)
            label = "ARUCO" if led_detected else "ARUCO (cached)"
            cv2.putText(bg, label, (lx + 14, ly - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, led_color, 1)

        # ── 焊枪尖端投影 ──
        tx, ty, tz = tip_xyz
        if tz > 0:
            tu = int(tx * K.fx / tz + K.cx)
            tv = int(ty * K.fy / tz + K.cy)
            if 0 <= tu < dw and 0 <= tv < dh:
                # 根据模式决定焊头图标大小：
                #   fixed:  小尺寸（cross=12, circle=10）
                #   aruco:  按二维码视觉半径缩放（cross=radius*2.5, circle=radius*2）
                #   cached: 中等尺寸（cross=18, circle=15）
                if tip_mode == "fixed":
                    cross_size = 12
                    circle_radius = 10
                elif led_detected:
                    cross_size = int(marker_radius * 2.5)
                    circle_radius = int(marker_radius * 2)
                else:
                    cross_size = 18
                    circle_radius = 15
                cv2.drawMarker(bg, (tu, tv), zone_color,
                               cv2.MARKER_CROSS, cross_size, 3)
                cv2.circle(bg, (tu, tv), circle_radius, zone_color, 2)
                # 标签：显示焊头来源
                if tip_mode == "fixed":
                    tip_label = "WELD TIP (fixed)"
                elif led_detected:
                    tip_label = "WELD TIP (aruco)"
                else:
                    tip_label = "WELD TIP (cached)"
                cv2.putText(bg, tip_label, (tu + 30, tv - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, zone_color, 2)
        else:
            tu, tv = -1, -1

        # ── 手部骨架 ──
        if result and result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                # 骨架连线
                for a, b in HAND_CONNECTIONS:
                    if a < len(hand_lms) and b < len(hand_lms):
                        ax = int(hand_lms[a].x * dw)
                        ay = int(hand_lms[a].y * dh)
                        bx = int(hand_lms[b].x * dw)
                        by = int(hand_lms[b].y * dh)
                        cv2.line(bg, (ax, ay), (bx, by), (255, 255, 255), 1, cv2.LINE_AA)

                # 关键点（指尖用黄色大圆，其余灰色小圆）
                for idx, lm in enumerate(hand_lms):
                    px = int(lm.x * dw)
                    py = int(lm.y * dh)
                    is_tip = idx in (4, 8, 12, 16, 20)  # 五个指尖
                    is_wrist = (idx == 0)
                    cv2.circle(bg, (px, py),
                               6 if (is_tip or is_wrist) else 3,
                               (0, 255, 255) if is_tip else
                               (0, 200, 255) if is_wrist else (200, 200, 200),
                               -1, cv2.LINE_AA)
                    if is_tip:
                        name = HAND_NAMES[idx]
                        cv2.putText(bg, name.split("_")[0], (px + 6, py - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                    (0, 255, 255), 1, cv2.LINE_AA)

        # ── 距离线 ──
        rd = metrics["risk_distance_m"]
        if rd is not None and rd < float("inf") and tu >= 0:
            ch = metrics["closest_hand"]
            cl = metrics["closest_landmark"]
            hu = hv = -1
            if result and result.hand_landmarks:
                for hi, hand_lms in enumerate(result.hand_landmarks):
                    hid = f"hand_{hi}"
                    if hid == ch and cl in HAND_NAMES:
                        idx = HAND_NAMES.index(cl)
                        if idx < len(hand_lms):
                            lm = hand_lms[idx]
                            hu = int(lm.x * (dw - 1))
                            hv = int(lm.y * (dh - 1))
                        break
            if hu >= 0:
                cv2.line(bg, (hu, hv), (tu, tv), zone_color, 2, cv2.LINE_AA)
                mx = (hu + tu) // 2
                my = (hv + tv) // 2
                cv2.putText(bg, f"{rd:.2f}m", (mx - 30, my - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, zone_color, 2)

        # ── 信息面板 ──
        self._draw_panel(bg, metrics, zone, rd, tip_mode, led_detected)

        # ── 底部按键提示 ──
        h, _ = bg.shape[:2]
        cv2.rectangle(bg, (0, h - 18), (dw, h), (30, 30, 30), -1)
        hints = "[Q] Quit  [M] Switch tip mode (fixed/dynamic)"
        cv2.putText(bg, hints, (8, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        bg = cv2.resize(bg, (1280, 960), interpolation=cv2.INTER_LINEAR)
        return bg

    def _draw_panel(self, img: np.ndarray, metrics: dict,
                    zone: str, rd: Optional[float],
                    tip_mode: str = "fixed",
                    led_detected: bool = False) -> None:
        """在图像左上角绘制半透明信息面板。"""
        overlay = img.copy()
        cv2.rectangle(overlay, (8, 26), (290, 228), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        cv2.rectangle(img, (8, 26), (290, 228), (100, 100, 100), 1)

        lines = [
            f"Zone: {zone.upper()}",
            f"Dist: {rd if rd else '-'}m",
            f"Part: {metrics['closest_hand']}/{metrics['closest_landmark']}",
            f"Speed: {metrics['approach_speed_ms']:.2f} m/s",
            f"Intent: {metrics['intent']}",
            f"Cmd: {metrics.get('skill_command') or '-'}",
        ]
        y = 28
        for txt in lines:
            cv2.putText(img, txt, (18, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22

        # 第7行：模式
        mode_txt = f"Mode: {tip_mode.upper()}"
        mode_color = (0, 255, 0) if tip_mode == "dynamic" else (200, 200, 200)
        cv2.putText(img, mode_txt, (18, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, mode_color, 1, cv2.LINE_AA)
        y += 22

        # 第8行：焊头来源
        if tip_mode == "fixed":
            tip_txt = "Tip: fixed offset"
            tip_color = (150, 150, 150)
        elif led_detected:
            tip_txt = "Tip: aruco tracked"
            tip_color = (0, 255, 255)
        else:
            tip_txt = "Tip: aruco cached/offset"
            tip_color = (100, 100, 100)
        cv2.putText(img, tip_txt, (18, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, tip_color, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
# 启动信息
# ═══════════════════════════════════════════════════════════════


def print_banner(port: int) -> None:
    """打印启动横幅。"""
    lines = [
        "=" * 60,
        "  Hand Safety System v5.2",
        "  Detection: MediaPipe HandLandmarker (21 keypoints)",
    ]
    if port:
        lines += [
            f"  TCP Server: tcp://0.0.0.0:{port}",
            f"  Hardware connects to: tcp://localhost:{port}",
        ]
    lines += [
        "  Signal fields: timestamp_ms, zone, risk_distance_m,",
        "                 approach_speed_ms, hand_part, intent,",
        "                 skill_command, sensor_uncertainty",
        "  Skill cmds:  null / slow_down / stop_welding",
        f"  Zones: green(>{GREEN_THRESHOLD_M}m) / yellow(>{YELLOW_THRESHOLD_M}m) / red",
        "=" * 60,
    ]
    print("\n" + "\n".join(lines) + "\n", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hand Safety System v4.1 — MediaPipe HandLandmarker"
    )
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="开启 OpenCV 可视化窗口")
    parser.add_argument("--simulate", action="store_true",
                        help="模拟模式（无相机，用于接口测试）")
    parser.add_argument("--tcp-port", type=int, default=9000,
                        help="TCP 服务端口（0=禁用）")
    parser.add_argument("--no-signal", action="store_true",
                        help="禁用所有信号输出（仅测试检测用）")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="信号输出文件路径（默认 stdout）")
    parser.add_argument("--offset-x", type=float, default=WELD_TIP_OFFSET[0],
                        help="焊枪尖端 X 偏移 (mm)")
    parser.add_argument("--offset-y", type=float, default=WELD_TIP_OFFSET[1],
                        help="焊枪尖端 Y 偏移 (mm)")
    parser.add_argument("--offset-z", type=float, default=WELD_TIP_OFFSET[2],
                        help="焊枪尖端 Z 偏移 (mm)")
    parser.add_argument("--green", type=float, default=GREEN_THRESHOLD_M,
                        help="绿色区域阈值 (m)")
    parser.add_argument("--yellow", type=float, default=YELLOW_THRESHOLD_M,
                        help="红色区域阈值 (m)")
    parser.add_argument("--force-zone", type=str, default=None,
                        choices=["green", "yellow", "red"],
                        help="强制输出指定区域信号（调试/硬件测试用）")
    parser.add_argument("--tip-mode", type=str, default="fixed",
                        choices=["fixed", "dynamic"],
                        help="焊头定位模式: fixed=固定偏移, dynamic=ArUco二维码追踪")
    args = parser.parse_args()

    tip = (args.offset_x, args.offset_y, args.offset_z)

    # ── TCP 服务 ──
    tcp: Optional[TCPServer] = None
    if args.tcp_port > 0 and not args.no_signal:
        tcp = TCPServer(args.tcp_port)
        try:
            tcp.start()
        except OSError as e:
            print(f"[WARN] TCP failed: {e}", file=sys.stderr)
            tcp = None

    print_banner(args.tcp_port if tcp else 0)

    # ── 相机 / 模拟 ──
    cam_thread: Optional[CameraThread] = None
    camera: Optional[Camera] = None
    sim_phase: float = 0.0
    sim_mode: bool = args.simulate

    if not sim_mode:
        camera = Camera()
        for retry in range(5):
            if camera.open():
                break
            print(f"[WARN] Camera open attempt {retry + 1}/5 failed, retrying...",
                  file=sys.stderr)
            time.sleep(3)
        else:
            print("[ERROR] Camera not found after 5 attempts. Try --simulate.",
                  file=sys.stderr)
            if tcp:
                tcp.stop()
            return 1
        cam_thread = CameraThread(camera)
        cam_thread.start()
        cam_thread.ready.wait(5)
        print(
            f"[OK] Camera connected (color={'yes' if camera.has_color() else 'no'})",
            file=sys.stderr,
        )
    else:
        print("[OK] Simulation mode (no camera needed)", file=sys.stderr)

    # ── 初始化模块 ──
    detector = HandDetector()
    safety = SafetyEngine(tip, args.green, args.yellow)
    signal = None if args.no_signal else SignalOutput(args.output, tcp)
    viz = Visualizer() if args.visualize else None
    tracker = TipTracker(tip, mode=args.tip_mode)

    print(
        f"[OK] Weld tip offset: ({tip[0]:.0f}, {tip[1]:.0f}, {tip[2]:.0f}) mm",
        file=sys.stderr,
    )
    if args.tip_mode == "dynamic":
        print("[OK] Tip mode: dynamic (ArUco tracking)", file=sys.stderr)

    print(
        f"[OK] Thresholds: green>{args.green}m  yellow>{args.yellow}m  red",
        file=sys.stderr,
    )
    if args.force_zone:
        print(f"[OK] Force ZONE enabled: always {args.force_zone.upper()}",
              file=sys.stderr)
    if args.simulate:
        print("[NOTE] Simulating depth input (no camera)", file=sys.stderr)
    print(
        "-" * 55 + "\n  Running...  Press q to quit\n" + "-" * 55,
        file=sys.stderr,
    )

    # ── 推理线程 → 主线程共享数据 ──
    _render_frame: list = [None]   # 渲染好的 BGR 图
    _render_lock = threading.Lock()
    _running = threading.Event()
    _running.set()
    _tip_mode: list = [args.tip_mode]  # 运行时切换 fixed/dynamic

    def inference_loop() -> None:
        nonlocal cam_thread
        last_signal_t = 0.0
        last_zone: Optional[str] = None
        frame_count = 0
        last_result: Any = None      # MediaPipe 检测结果缓存
        last_hands_3d: dict = {}     # 3D 姿态缓存
        last_bg_h: Optional[np.ndarray] = None  # 热力图缓存

        while _running.is_set():
            # ① 获取帧
            if sim_mode:
                nonlocal sim_phase
                sim_phase += 0.04
                d_m = 1.3 + 1.2 * math.sin(sim_phase)
                depth = np.full((480, 640), int(d_m * 1000), dtype=np.uint16)
                color = None
            else:
                assert cam_thread is not None
                # ── USB 断开自动重连 ──
                if cam_thread.timed_out and camera is not None:
                    cam_thread.timed_out = False
                    print("[WARN] Camera timed out, reconnecting...", file=sys.stderr)
                    cam_thread.stop()
                    if camera.reconnect():
                        print("[OK] Camera reconnected", file=sys.stderr)
                        cam_thread = CameraThread(camera)
                        cam_thread.start()
                        continue
                    else:
                        print("[ERROR] Camera reconnect failed, retrying...",
                              file=sys.stderr)
                        time.sleep(2)
                        continue

                try:
                    depth, color = cam_thread.frame_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if depth is None:
                    continue

            # ② 跳帧：每 INFERENCE_SKIP 帧跑一次 MediaPipe，其余帧用缓存
            frame_count += 1
            if frame_count % INFERENCE_SKIP == 0:
                result, hands_3d, bg_h = detector.detect(color, depth)
                last_result = result
                last_hands_3d = hands_3d
                last_bg_h = bg_h
            else:
                result = last_result
                hands_3d = last_hands_3d
                bg_h = last_bg_h

            # ③ 焊头检测（ArUco 二维码追踪 / 固定偏移）
            led_detected = False
            led_pos: Optional[Tuple[int, int]] = None
            marker_radius = 12.0
            if sim_mode:
                dynamic_tip = tip
            else:
                dynamic_tip, led_detected, led_pos, marker_radius = \
                    tracker.detect_with_vis(color, depth)

            # ④ 安全评估
            metrics = safety.evaluate(hands_3d, tip_xyz=dynamic_tip)

            # ── 强制区域模式 ──
            if args.force_zone:
                force_zone = args.force_zone
                metrics["zone"] = force_zone
                if force_zone == "green":
                    metrics["skill_command"] = None
                elif force_zone == "yellow":
                    metrics["skill_command"] = "slow_down"
                elif force_zone == "red":
                    metrics["skill_command"] = "stop_welding"

            # ④ 信号输出
            if signal is not None:
                now = time.time()
                if (now - last_signal_t >= SIGNAL_INTERVAL_S
                        or metrics["zone"] != last_zone):
                    signal.emit(metrics)
                    last_signal_t = now
                    last_zone = metrics["zone"]

            # ⑤ 渲染图写入共享变量（主线程来读）
            if viz is not None:
                frame = viz.draw(depth, color, result, metrics, dynamic_tip,
                                 bg_heatmap=bg_h,
                                 tip_mode=_tip_mode[0],
                                 led_pos=led_pos,
                                 led_detected=led_detected,
                                 marker_radius=marker_radius,
                                 aruco_corners=tracker.latest_corners,
                                 aruco_ids=tracker.latest_ids)
                with _render_lock:
                    _render_frame[0] = frame

    # 推理线程启动
    t = threading.Thread(target=inference_loop, daemon=True, name="Inference")
    t.start()

    # ── 主线程只做 GUI ──
    def cleanup() -> None:
        """释放所有资源。"""
        _running.clear()
        t.join(timeout=3)
        detector.close()
        if camera is not None:
            camera.close()
        if signal is not None:
            signal.close()
        if cam_thread is not None:
            cam_thread.stop()
        cv2.destroyAllWindows()

    try:
        # 可视化模式下立即创建窗口
        if viz is not None:
            cv2.imshow("Hand Safety", np.zeros((400, 640, 3), dtype=np.uint8))
        while _running.is_set():
            if viz is not None:
                # 检测窗口是否被手动关闭
                if cv2.getWindowProperty("Hand Safety", cv2.WND_PROP_VISIBLE) < 1:
                    print("[GUI] Window closed", file=sys.stderr)
                    break
                with _render_lock:
                    frame = _render_frame[0]
                if frame is not None:
                    cv2.imshow("Hand Safety", frame)
            # waitKey 无论有没有新帧都要保持窗口响应
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[GUI] Quit by user", file=sys.stderr)
                break
            if key == ord("m"):
                new_mode = "dynamic" if _tip_mode[0] == "fixed" else "fixed"
                _tip_mode[0] = new_mode
                tracker.switch_mode(new_mode)
                print(f"[GUI] Tip mode switched to: {new_mode}", file=sys.stderr)
    except KeyboardInterrupt:
        print("[GUI] Interrupted", file=sys.stderr)
    finally:
        cleanup()
        # 确保所有子线程终止后进程完全退出
        os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
