# pyright: reportAttributeAccessIssue=false
# pyright: reportUnusedExpression=false

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from _test_init import DATA_ROOT
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from PIL.Image import Image as ImageType
from shapely import Point
from shapely.geometry import LineString, Polygon
from shapely.ops import split

from core.color_extraction import *

np.random.seed(42)


@dataclass
class SplitPlane:
    point: NDArray  # (3,)
    direction: NDArray  # (3,)
    region: Optional[Polygon] = None  # ab 平面的有效区域
    depth: int = 0
    label: str = ""


def split_colors_svd(
    pixels: NDArray,
    depth: int = 0,
    max_depth: int = 4,
    var_theshold: float = 0,
):
    """递归切分，在PC1轴上最大化类间方差切分"""
    if len(pixels) == 0:
        return

    variance: float = np.average(pixels.var(axis=0), weights=[1, 1, 1])  # type: ignore
    if variance < var_theshold or depth >= max_depth:
        return

    # 投影到 PC1 上并排序
    X = pixels.astype(np.float64)
    # 中心化
    X_centered = X - X.mean(axis=0)
    # SVD分解得到主轴
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    pc1 = Vt[0]  # 第一主成分方向
    projections = X_centered @ pc1  # 投影值
    projections = project_onto_pc1(pixels)
    sorted_indices = np.argsort(projections)
    pixels_sorted = pixels[sorted_indices]

    # 中位切分(最大化类间方差)
    best_index = bcv_best_split_index(projections[sorted_indices])

    yield (pixels_sorted[best_index], pc1, X.mean(axis=0))

    yield from split_colors_svd(pixels_sorted[:best_index], depth + 1, max_depth, var_theshold)
    yield from split_colors_svd(pixels_sorted[best_index:], depth + 1, max_depth, var_theshold)


def bcv_best_split_index(projections: NDArray[np.float64], num_steps: int = 150) -> int:
    """
    在 PC1 投影值上寻找使类间方差最大的切分点索引。
    输入数组必须已排序。
    返回最佳切分点的索引（在 projections 中的位置）。
    """
    n = len(projections)
    assert n >= 2, "投影值至少需要两个元素"

    # 1. 计算1/4分位和3/4分位
    q1 = np.percentile(projections, 25)
    q3 = np.percentile(projections, 75)

    # 2. 在 q1 和 q3 之间均匀采样切分值
    thresholds = np.linspace(q1, q3, num_steps)
    best_score = -np.inf
    best_index = -1

    # 3. 穷举所有候选切分点
    for t in thresholds:
        idx = np.searchsorted(projections, t, side="right")

        if idx <= 0 or idx >= n:
            continue  # 跳过边缘切分

        left = projections[:idx]
        right = projections[idx:]

        w0 = len(left) / n
        w1 = 1 - w0
        mu0 = left.mean()
        mu1 = right.mean()

        bcv = w0 * w1 * (mu0 - mu1) ** 2

        if bcv > best_score:
            best_score = bcv
            best_index = idx

    return best_index


def plot_in_ab(
    ax: Axes,
    lab: NDArray,
    L_display_range: Tuple[int, int] = (30, 80),  # 这是0-100的标准范围
):
    a, b = lab[:, 1], lab[:, 2]

    # 统计每个 (a, b) 对应的像素个数
    hist = np.zeros((256, 256), dtype=np.uint32)
    for aa, bb in zip(a.flatten(), b.flatten()):
        hist[aa, bb] += 1

    # 提取非零项位置
    aa_coords, bb_coords = np.nonzero(hist)
    counts: NDArray = hist[aa_coords, bb_coords].astype(float)

    # 处理极端大值
    t = np.percentile(counts, 60)
    k = 3 / t
    counts = 1 / (1 + np.exp(-k * (counts - t)))

    # 归一化计数映射到 L 范围
    L_min, L_max = L_display_range
    counts_norm: NDArray[np.floating] = counts / counts.max()
    mapped_L = (1 - counts_norm) * (L_max - L_min) + L_min  # shape: (N,)
    mapped_L = mapped_L * 255.0 / 100.0

    # 构造 Lab 点 → (N, 3)
    lab_pts = np.stack([mapped_L, aa_coords, bb_coords], axis=1).astype(np.uint8)
    rgb_pts = LAB2RGB(lab_pts)

    # 绘图
    ax.scatter(aa_coords - 128, bb_coords - 128, c=rgb_pts / 255.0, s=15)
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.grid(True)
    ax.axis("equal")


def plot_pc1(ax, lab, S, Vt):
    # 绘制 PCA 主轴方向
    mean_lab = lab.mean(axis=0)
    for i in range(3):
        if S[i] < 1:
            continue
        ab_dir = Vt[i][1:]  # (a, b)
        max_S = S.max()
        scale = 50  # 全局缩放
        length = S[i] / max_S * scale
        ax, bx = ab_dir * length
        plt.arrow(
            mean_lab[1] - 128,
            mean_lab[2] - 128,
            ax,
            bx,
            color=f"C{i}",
            width=0.5,
            # head_width=4.5,
            # length_includes_head=True,
            label=f"PC{i+1} (σ={S[i]:.1f})",
        )
    plt.legend()


def plot_split_line(ax: Axes, lab: NDArray):
    # 初始 ab 平面区域
    region_stack = [Point(128, 128).buffer(128)]

    for center, direction, mean in split_colors_svd(lab, max_depth=3):
        if not region_stack:
            break
        current_region = region_stack.pop()

        ab_origin = mean[1:]
        ab_proj = direction[1:]
        ab_proj /= np.linalg.norm(ab_proj)
        ab_dir = np.array([-ab_proj[1], ab_proj[0]])
        ab_dir /= np.linalg.norm(ab_dir)
        if np.allclose(ab_dir, 0):
            continue
        ab_dir = ab_dir / np.linalg.norm(ab_dir)
        ab_point = ab_origin + np.dot(center[1:] - ab_origin, ab_proj) * ab_proj
        # ax.scatter(ab_point[0] - 128, ab_point[1] - 128)

        if not current_region.contains(Point(*center[1:])):
            region_stack.pop()
            current_region = region_stack.pop()
            # 说明这个递归分支已经终止，清理当前 + 上一个多余区域

        # 用你的 clip 函数裁剪线段
        clipped = clip_line_in_polygon(ab_point, ab_dir, current_region)
        if clipped is None:
            continue
        p1, p2 = clipped

        # 绘制分割线
        ax.plot(
            [p1[0] - 128, p2[0] - 128],
            [p1[1] - 128, p2[1] - 128],
            linestyle="--",
            linewidth=2,
            color="black",
            # alpha=0.6,
        )

        # 分割当前 polygon
        full_line = LineString([ab_point - ab_dir * 1000, ab_point + ab_dir * 1000])

        try:
            result = split(current_region, full_line)
        except Exception:
            print("Error")
            continue  # 容错处理

        if len(result.geoms) < 2:
            continue  # 切不出两个区域，跳过

        polys = list(result.geoms)
        centers = [np.array(poly.centroid.coords[0]) for poly in polys]

        def is_right_region(point):
            vec = point - ab_point  # ab_point 是切分点
            return np.dot(vec, ab_proj) > 0

        left_region = None
        right_region = None
        for poly, c in zip(polys, centers):
            if is_right_region(c):
                right_region = poly
            else:
                left_region = poly

        # 默认按顺序压栈：右先入，左后入，保持 DFS 顺序
        region_stack.append(right_region)
        region_stack.append(left_region)


def clip_line_in_polygon(center: NDArray, direction: NDArray, polygon: Polygon):
    """
    给定 ab 平面上的中心点 + 方向向量 + 区域（Polygon），
    返回该方向直线在区域内的裁剪线段两个端点。
    """
    # 构造一条很长的线穿过 center
    point1 = center - direction * 1000
    point2 = center + direction * 1000
    full_line = LineString([point1, point2])

    # 与区域相交
    clipped = polygon.intersection(full_line)

    if clipped.is_empty:
        return None  # 无交线
    elif isinstance(clipped, LineString):
        coords = list(clipped.coords)
        return np.array(coords[0]), np.array(coords[1])
    else:
        # 多段或点，取最远两个点
        coords = []
        for geom in clipped.geoms:
            if isinstance(geom, LineString):
                coords += list(geom.coords)
        if len(coords) < 2:
            return None
        coords.sort(key=lambda p: np.linalg.norm(np.array(p) - center))
        return np.array(coords[0]), np.array(coords[-1])


def show_color_distribution(
    image_rgb: np.ndarray,
    image_lab: np.ndarray,
):
    h, w, _ = image_rgb.shape
    total = h * w
    num_samples = 50000  # 采样点数
    idx = np.random.choice(total, size=min(num_samples, total), replace=False)
    pixels_rgb = image_rgb.reshape(-1, 3)[idx]
    pixels_lab = image_lab.reshape(-1, 3)[idx]

    fig = plt.figure(figsize=(4, 4))

    # RGB 分布
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore
    ax.scatter(pixels_rgb[:, 0], pixels_rgb[:, 1], pixels_rgb[:, 2], c=pixels_rgb / 255.0, marker=".", s=1)  # type: ignore
    ax.set_title("RGB Color Distribution")
    ax.set_xlabel("R"), ax.set_ylabel("G"), ax.set_zlabel("B")

    # Lab 分布（注意不能直接用 Lab 值设为颜色）
    fig2 = plt.figure(figsize=(4, 4))
    ax2: Axes3D = fig2.add_subplot(111, projection="3d")  # type: ignore
    ax2.scatter(
        pixels_lab[:, 1].astype(float) - 128,
        pixels_lab[:, 2].astype(float) - 128,
        pixels_lab[:, 0].astype(float) * 100 / 255.0,  # type: ignore
        c=pixels_rgb / 255.0,
        marker=".",
        s=1,
    )
    ax2.set_title("Lab Color Distribution")
    ax2.set_xlabel("a"), ax2.set_ylabel("b"), ax2.set_zlabel("L")


def main():
    # image_path = DATA_ROOT / "svd/1.png"
    image_path = DATA_ROOT / "svd/0.jpeg"
    image_path = DATA_ROOT / "svd/Cover.jpg"
    image: ImageType = Image.open(image_path).convert("RGB")
    image.thumbnail((500, 500), Image.Resampling.LANCZOS)
    img_rgb = np.array(image)
    img_lab = RGB2LAB(img_rgb)

    rgb_flat = img_rgb.reshape(-1, 3)  # shape: (H*W, 3)
    lab_flat = img_lab.reshape(-1, 3)  # shape: (H*W, 3)

    # mean_lab = lab_flat.mean(axis=0)
    # X = lab_flat - mean_lab  # 零中心化
    # U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # print(S)
    # show_color_distribution(img_rgb, img_lab)

    lab_flat[:, 0] = 128  # 去掉L维度
    mean_lab = lab_flat.mean(axis=0)
    U, S, Vt = np.linalg.svd(lab_flat - mean_lab, full_matrices=False)
    fig, ax = plt.subplots()
    plot_in_ab(ax, lab_flat)
    # plot_pc1(ax, lab_flat, S, Vt)
    plot_split_line(ax, lab_flat)

    plt.show()


if __name__ == "__main__":
    main()
