"""
生成 ArUco 二维码标记（打印、裁剪、贴在焊头上）

用法：
  python generate_aruco_markers.py

输出：
  在当前目录生成 4 个 PNG 文件：
    aruco_ID0.png   aruco_ID1.png   aruco_ID2.png   aruco_ID3.png

贴在焊头安装环的 4 个平面上即可实现 360° 追踪。
"""
import cv2
import numpy as np
import os

# ── 参数 ──────────────────────────────────────────────────────

# 必须与 hand_safety_system.py 中的 ARUCO_DICT 一致
DICT_NAME = cv2.aruco.DICT_6X6_250

# 需要生成的标记 ID（系统会检测 ID=0~3 中的任意一个）
TARGET_IDS = [0, 1, 2, 3]

# 输出图片尺寸（像素）— 打印时按实际尺寸缩放即可
OUTPUT_SIZE_PX = 1200        # 图片边长（大一点打印更清晰）
MARKER_SIZE_PX = 800         # 标记本身边长（留白边便于裁剪）
BORDER_BITS = 1              # ArUco 自带白边宽度（单位：格子数）

# ── 生成并保存 ────────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))
dictionary = cv2.aruco.getPredefinedDictionary(DICT_NAME)

for marker_id in TARGET_IDS:
    # 生成 ArUco 标记图像（cv2.aruco.generateImageMarker 返回的是单色位图）
    marker_img = cv2.aruco.generateImageMarker(
        dictionary, marker_id, MARKER_SIZE_PX, borderBits=BORDER_BITS
    )

    # 放入白色画布（方便打印裁剪）
    canvas = np.ones((OUTPUT_SIZE_PX, OUTPUT_SIZE_PX), dtype=np.uint8) * 255
    offset = (OUTPUT_SIZE_PX - MARKER_SIZE_PX) // 2
    canvas[offset:offset + MARKER_SIZE_PX, offset:offset + MARKER_SIZE_PX] = marker_img

    # 保存
    out_path = os.path.join(script_dir, f"aruco_ID{marker_id}.png")
    cv2.imwrite(out_path, canvas)
    print(f"  [OK] {out_path}  ({OUTPUT_SIZE_PX}x{OUTPUT_SIZE_PX} px)")

print("\n生成完毕！打印时请确保：")
print("  1. 打印时**不要缩放**（100% 原尺寸）")
print("  2. 裁剪后贴在安装环的 4 个平面上")
print("  3. 保证标记表面平整、无反光\n")
