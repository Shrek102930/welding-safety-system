// ═══════════════════════════════════════════════════════════════
// 焊头 ArUco 多面安装环 — 3D 打印参数化模型
//
// 用法：
//   1. 用游标卡尺量焊头直径，修改 WELD_TIP_DIAMETER
//   2. 用 OpenSCAD 打开此文件
//   3. 渲染 (F6) → 导出 STL (F7) → 3D 打印
//   4. 打印后把 4 个 ArUco 码（ID=0~3, 6×6_250）贴在 4 个平面上
//
// ═══════════════════════════════════════════════════════════════

// ── 用户参数 ──────────────────────────────────────────────────

/* [焊头尺寸] */
// 焊头圆柱体直径（mm）— 用游标卡尺量
WELD_TIP_DIAMETER = 20;

/* [环尺寸] */
// 环壁厚（mm）
WALL_THICKNESS = 3;
// 环高度（mm）
RING_HEIGHT = 25;
// 二维码贴纸边长（mm）— 打印 AruCo 时设为同样尺寸
MARKER_SIZE = 18;

/* [打印机设置] */
// 层高（mm）— 0.2 普通, 0.12 精细
LAYER_HEIGHT = 0.2;
// 是否加底边（提高平台附着力）
ADD_BRIM = true;

// ── 计算公式 ──────────────────────────────────────────────────

TIP_RADIUS = WELD_TIP_DIAMETER / 2;
OUTER_RADIUS = TIP_RADIUS + WALL_THICKNESS;
FLAT_WIDTH = MARKER_SIZE + 4;  // 贴纸 + 2mm 边框
// 由外径和贴纸宽度推导平面距离圆心的缩进
FLAT_INSET = sqrt(OUTER_RADIUS * OUTER_RADIUS - (FLAT_WIDTH / 2) * (FLAT_WIDTH / 2));

// ── 主模型 ────────────────────────────────────────────────────

module aruco_mount_ring() {
    difference() {
        union() {
            // 基础环
            difference() {
                cylinder(h = RING_HEIGHT, r = OUTER_RADIUS, $fn = 64);
                translate([0, 0, -0.1])
                    cylinder(h = RING_HEIGHT + 0.2, r = TIP_RADIUS, $fn = 64);
            }

            // 4 个平面的贴纸平台（凸块）
            for (angle = [0:90:270]) {
                rotate([0, 0, angle]) {
                    translate([FLAT_INSET, 0, 0]) {
                        // 凸块 — 给贴纸一个平整的矩形面
                        cube([MARKER_SIZE + 2, MARKER_SIZE + 4, RING_HEIGHT],
                             center = true);
                    }
                }
            }
        }
    }

    // ── 4 个面的 ArUco 定位标记 ──
    // 打印时可以用这些浅槽定位贴纸位置
    %for (angle = [0:90:270]) {
        rotate([0, 0, angle]) {
            translate([FLAT_INSET + 1, 0, 0]) {
                // 浅槽区域（0.4mm 深），辅助贴纸定位
                color("Gray", 0.3)
                    cube([MARKER_SIZE, MARKER_SIZE, 0.4], center = true);
            }
        }
    }
}

// ── 底边（可选）───────────────────────────────────────────────

module brim() {
    if (ADD_BRIM) {
        linear_extrude(height = LAYER_HEIGHT, convexity = 4) {
            offset(r = 3) {
                circle(r = OUTER_RADIUS + WALL_THICKNESS, $fn = 64);
            }
        }
    }
}

// ── 渲染 ──────────────────────────────────────────────────────

$fn = 64;

brim();
aruco_mount_ring();

// ═══════════════════════════════════════════════════════════════
// 打印说明
// ═══════════════════════════════════════════════════════════════
//
// 材料推荐：
//   PLA 或 PETG（普通 FDM 打印）
//   不需要支撑（因为 4 个凸块是水平延伸的）
//
// 后处理：
//   1. 打磨 4 个平面（平整贴纸）
//   2. 裁剪 4 个 ArUco 码（6×6_250, ID=0,1,2,3），尺寸 = MARKER_SIZE 对应
//   3. 用双面胶或胶水贴在 4 个平面上
//
// ArUco 码生成：
//   搜索 "ArUco marker generator"
//   选 6×6 bits, ID=0~3, size=18×18mm
//   打印后用剪刀裁剪
//
// 安装：
//   套在焊头上，紧配合（如果太松，用胶带绑一圈）
//   确保 4 个面中至少有一个朝向相机方向
//
// ═══════════════════════════════════════════════════════════════
