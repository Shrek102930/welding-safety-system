<p align="center">
  <h1 align="center">🤖 人机协作机械臂焊接安全可验证具身智能系统</h1>
  <p align="center">
    <i>Embodied Intelligence Safety System for Human-Robot Collaborative Welding</i>
  </p>
  
  <p align="center">
    <a href="https://github.com/Shrek102930/welding-safety-system/stargazers"><img src="https://img.shields.io/github/stars/Shrek102930/welding-safety-system?style=flat&logo=github&color=yellow" alt="Stars"></a>
    <a href="https://github.com/Shrek102930/welding-safety-system/forks"><img src="https://img.shields.io/github/forks/Shrek102930/welding-safety-system?style=flat&logo=github" alt="Forks"></a>
    <a href="https://github.com/Shrek102930/welding-safety-system/issues"><img src="https://img.shields.io/github/issues/Shrek102930/welding-safety-system" alt="Issues"></a>
  </p>

  <p align="center">
    <b>Python</b> ·
    <b>PyTorch</b> ·
    <b>MediaPipe</b> ·
    <b>YOLOv8</b> ·
    <b>ROS 2</b> ·
    <b>Gazebo</b> ·
    <b>ESP32-S3</b>
  </p>
</p>

---

## 📖 简介 / Overview

针对焊接产线 **人机共域作业** 的安全痛点，研发了这套 **"感知 → 决策 → 执行 → 验证" 全链路具身智能安全系统**。通过多模态感知融合穿透弧光烟尘、基于 GNN 时序意图预测实现前瞻性风险判断、ISO/TS 15066 标准验证器兜底——在**无物理围栏**条件下实现人机协作安全闭环。

> 传统方案依赖物理隔离，无法适应柔性产线频繁换线需求。本系统用 AI 替代围栏，让机器人"看见"并"理解"人的行为。

## ✨ 核心亮点 / Highlights

| 能力 | 方案 | 效果 |
|------|------|------|
| 🔬 **多模态感知融合** | 毫米波雷达 + 深度相机 + 视觉姿态估计 | 穿透弧光烟尘，3D 风险距离实时估计 |
| 🧠 **意图预测网络** | 时序模型 + 图神经网络 (GNN) | 自建 **5,500 片段**数据集，准确率 **90%+** |
| ⚡ **软实时执行** | ROS 2 + Gazebo 仿真 + ESP32-S3 实物部署 | 手部推理 **8–12ms**，多模态融合 **10ms** |
| 🛡️ **安全验证兜底** | ISO/TS 15066 标准安全验证器 | 形式化保证人机协作安全边界 |
| 🌐 **Web 决策面板** | Flask + WebSocket 全链路可视化 | 日志追溯 + 参数配置 + 状态监控 |

### 落地成果

```
急停频率 ↓ 78%     碰撞率 < 5%      产线 OEE ↑ 可量化
推理延迟 8–12ms    数据集 5,500 片段   通过 ISO/TS 15066 验证
```

---

## 🏗️ 系统架构 / Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Web 决策面板 (Flask)                        │
│              全链路可视化 │ 日志 │ 参数配置                    │
└──────────────────────┬──────────────────────────────────────┘
                       │ WebSocket / REST
┌──────────────────────▼──────────────────────────────────────┐
│               安全决策引擎 (ROS 2 Node)                       │
│                                                               │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │ 多模态    │ → │ 风险距离     │ → │ 意图预测     │        │
│  │ 感知融合  │   │ 3D 估计      │   │ GNN+时序     │        │
│  └──────────┘   └──────────────┘   └──────┬───────┘        │
│                                              │              │
│                    ┌─────────────────────────▼────────┐     │
│                    │     ISO/TS 15066 安全验证器       │     │
│                    └────────────┬────────────────────┘     │
│                                 │                          │
│            ┌────────────────────┼────────────────────┐     │
│            ▼                    ▼                    ▼     │
│       ┌─────────┐        ┌──────────┐         ┌────────┐  │
│       │ 速度限制 │        │ 急停     │         │ 声光报警│  │
│       │ 0.3x 减速│        │ 安全位   │         │ 舵机LED │  │
│       └─────────┘        └──────────┘         └────────┘  │
└──────────┬──────────────────┬──────────────────────────────┘
           │                  │
    ┌──────▼───┐       ┌────▼────────┐
    │ ESP32-S3 │       │ Gazebo 仿 真 │
    │ 实物执行  │       │ 闭环验证    │
    │ 20ms调度  │       │ 场景库 10+  │
    └──────────┘       └─────────────┘
```

---

## 🚀 快速开始 / Quick Start

### 环境要求

```bash
Python >= 3.9
pip install opencv-python numpy mediapipe ultralytics
```

### 运行

```bash
# 模拟模式（无需硬件，推荐先试）
python hand_safety_system.py --simulate

# 真实相机模式（需 Orbbec Gemini 2）
python hand_safety_system.py --visualize

# 生产模式（纯信号输出）
python hand_safety_system.py
```

### 安全区划分

| 区域 | 颜色 | 距离 | 响应 |
|------|------|------|------|
| 安全区 | 🟢 | > 100cm | 正常运行 |
| 警告区 | 🟡 | 60–100cm | 机器人减速 0.3x |
| 危险区 | 🔴 | < 30cm | 急停 + 报警 |
| 锁定区 | 🟣 | — | 需手动重置 |

按键：`q` 退出 / `r` 重置

---

## 📂 项目结构 / Project Structure

```
welding-safety-system/
├── hand_safety_system.py       # ⭐ 核心系统 v5.0 (1557 行)
│                               #   CameraThread → InferenceThread → MainThread
│                               #   三线程架构: 相机采集/推理计算/可视化展示
├── generate_aruco_markers.py    # ArUco 标定板生成器
├── aruco_mount_ring.scad       # 标定板安装支架 (OpenSCAD)
├── README.md                   # 本文件
└── .gitignore                  # 排除规则(大文件/虚拟环境/缓存)
```

> 💡 **说明**: 模型权重（YOLO `.pt` / MediaPipe `.task`）因体积较大未纳入本仓库。
> 运行时自动下载或从 [Releases](../../releases) 获取。

---

## 🛠️ 技术栈 / Tech Stack

| 层级 | 技术 |
|------|------|
| **感知** | Python · PyTorch · MediaPipe HandLandmarker (21关键点) · OpenCV · Orbbec Gemini 2 · 毫米波雷达 |
| **算法** | YOLOv8/v11 目标检测 · GNN 图神经网络 · 时序预测 · 多模态特征融合 |
| **机器人** | ROS 2 · Gazebo 仿真 · ESP32-S3 嵌入式 · 三自由度舵机组 |
| **工程** | FastAPI · WebSocket · TCP JSON Lines · Docker |
| **安全标准** | [ISO/TS 15066](https://www.iso.org/standard/67846.html) 协作机器人规范 |

---

## 📊 性能指标 / Benchmarks

| 指标 | 数值 | 说明 |
|------|------|------|
| 手部推理延迟 | **8–12 ms** | MediaPipe HandLandmarker on GPU |
| 多模态融合延迟 | **10 ms** | 含深度图对齐+距离计算 |
| 意图预测准确率 | **90%+** | 自建 5,500 片段数据集 (GNN+时序) |
| 急停响应时间 | **< 20 ms** | ESP32-S3 调度 + 舵机 + LED |
| 急停频率下降 | **−78%** | 对比无安全系统基线 |
| 碰撞率 | **< 5%** | 仿真 + 实测综合 |

---

## 📄 License

本项目仅供学习交流使用。部分组件（MediaPipe、YOLOv8 等）遵循各自开源协议。

---

<p align="center">
  <sub>If this project helped you, give it a ⭐!</sub>
</p>
