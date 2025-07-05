from collections import defaultdict
from typing import Callable, List, Tuple

import numpy as np
from color_space import *
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as ImageType
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist

ColorExtractor = Callable[[ImageType], Tuple[str, List[RGBColor]]]


def filter_saturated_colors(pixels: NDArray[np.uint8], threshold: int = 20) -> NDArray[np.uint8]:
    """在LAB空间中过滤低饱和度颜色"""
    a = pixels[..., 1].astype(np.int16) - 128
    b = pixels[..., 2].astype(np.int16) - 128
    saturation = np.sqrt(a**2 + b**2)  # 计算颜色的饱和度
    return pixels[saturation > threshold]  # 只保留高饱和度颜色


def filter_colors_hls(
    pixels: NDArray[np.uint8], l_range: Tuple[float, float] = (0.2, 0.8), s_threshold: float = 0.3
) -> NDArray[np.bool]:
    """
    使用 HLS 空间过滤颜色，排除过黑、过白和过灰的颜色。

    Args:
        pixels (np.ndarray): 形状为 (N, 3) 的 HLS 颜色数组，范围 [0, 255]。
        l_range (tuple): Lightness 的合理范围 (min, max)，范围是 [0, 1]。
        s_threshold (float): 饱和度阈值，低于此值视为灰色，范围 [0, 1]。

    Returns:
        np.ndarray: 筛选后的掩码数组。
    """
    l_channel = pixels[:, 1]  # 亮度
    s_channel = pixels[:, 2]  # 饱和度

    # 创建掩码：亮度和饱和度条件同时满足
    mask = (
        (l_channel >= l_range[0] * 255)
        & (l_channel <= l_range[1] * 255)  # 亮度范围
        & (s_channel >= s_threshold * 255)  # 饱和度条件
    )

    return mask


def rgb_average(image: ImageType) -> Tuple[str, List[RGBColor]]:
    """RGB 颜色空间的平均值"""
    np_img = np.array(image)
    avg_color = np_img.mean(axis=(0, 1))
    r, g, b = map(int, avg_color[:3])
    return "RGB 均值", [RGBColor(r, g, b)]


def lab_average(image: ImageType) -> Tuple[str, List[RGBColor]]:
    """LAB 颜色空间的平均值，转换回 RGB"""
    np_img = np.array(image)
    np_img = RGB2LAB(np_img)
    avg_color = np_img.mean(axis=(0, 1))
    rgb_color = LAB2RGB(avg_color)
    r, g, b = map(int, rgb_color)
    return "LAB 均值", [RGBColor(r, g, b)]


def median_cut_lab(image: ImageType) -> Tuple[str, List[RGBColor]]:
    """
    在 LAB 颜色空间内执行中位切分算法提取主色（包含去除低饱和像素）。

    Args:
        image (ImageType): 输入 PIL 图像

    Returns:
        Tuple[str, List[RGBColor]]: (方法名, 提取的 RGB 颜色)
    """
    num_colors: int = 5
    rgb_array = np.array(image)
    lab_array = RGB2LAB(rgb_array)
    hls_array = RGB2HLS(rgb_array).reshape(-1, 3)
    pixels = lab_array.reshape(-1, 3)  # 转换为 (N, 3)

    # pixels = filter_saturated_colors(pixels)
    pixels = pixels[filter_colors_hls(hls_array)]

    # 递归切分
    def split_colors(pixels: NDArray[np.uint8], depth: int = 0) -> List[Tuple[NDArray[np.float64], float]]:
        if len(pixels) == 0:
            return []
        if len(pixels) <= len(lab_array) // num_colors or depth >= np.log2(num_colors):
            mean_color = pixels.mean(axis=0)  # 计算均值
            variance: float = np.average(pixels.var(axis=0), weights=[1 / 6.5, 1, 1])  # type: ignore
            return [(mean_color, variance)]  # 返回 (均值, 方差) 元组

        # 找到方差最大的通道
        var_l, var_a, var_b = np.var(pixels, axis=0)
        max_var_index = np.argmax([var_l / 6.5, var_a, var_b])  # cv2中L有缩放，将其权重调整一下(2.55**2)

        # 按该通道的中位数切分
        pixels = pixels[pixels[:, max_var_index].argsort()]
        median_index = len(pixels) // 2
        return split_colors(pixels[:median_index], depth + 1) + split_colors(pixels[median_index:], depth + 1)

    # 获取主色
    dominant_lab_colors = sorted(split_colors(pixels), key=lambda x: x[1])

    # 转换为 RGB
    dominant_rgb_colors = [
        tuple(map(int, LAB2RGB(np.array(color[0], dtype=np.uint8)))) for color in dominant_lab_colors
    ]

    if len(dominant_lab_colors) != 0:
        return "中位切分+前处理", [RGBColor.from_array(dominant_rgb_colors[0][:3])]  # type: ignore
    else:
        raise IndexError("没有提取到主色")


def final_solution(image: ImageType) -> Tuple[str, List[RGBColor]]:
    """
    最终方案：中位切分 + HLS前后处理

    Args:
        image (ImageType): 输入 PIL 图像

    Returns:
        Tuple[str, List[RGBColor]]: (方法名, 提取的 RGB 颜色)
    """
    num_colors: int = 5
    rgb_array = np.array(image)
    lab_array = RGB2LAB(rgb_array)
    pixels = lab_array.reshape(-1, 3)  # 转换为 (N, 3)
    hls_array = RGB2HLS(rgb_array).reshape(-1, 3)

    # pixels_filted = filter_saturated_colors(pixels)
    pixels_filted = pixels[filter_colors_hls(hls_array)]
    pixels_filted = pixels

    if len(pixels_filted) <= 20:  # 绝大部分色彩都被过滤了，所以不再使用中位切分，直接平均
        avg_color = pixels.mean(axis=0)
        rgb_color = LAB2RGB(avg_color)
        r, g, b = map(int, rgb_color)
        return "中位切分+前后处理", [RGBColor(r, g, b)]
    else:
        pixels = pixels_filted  # 历史遗留的变量名替换，懒得改

        # 递归切分
        def split_colors(pixels: NDArray[np.uint8], depth: int = 0) -> List[Tuple[NDArray[np.float64], float]]:
            if len(pixels) == 0:
                return []
            if len(pixels) <= len(lab_array) // num_colors or depth >= np.log2(num_colors):
                mean_color = pixels.mean(axis=0)  # 计算均值
                variance: float = np.average(pixels.var(axis=0), weights=[0.1, 1, 1])  # type: ignore
                return [(mean_color, variance)]  # 返回 (均值, 方差) 元组

            # 找到方差最大的通道
            var_l, var_a, var_b = np.var(pixels, axis=0)
            # cv2中L有缩放，将其权重调整一下(2.55**2)
            # cv2使用0-255的数值表示0-100的L
            # 但是实际的权重更小，因为不太想区分亮度
            max_var_index = np.argmax([var_l / 10, var_a, var_b])

            # 按该通道的中位数切分
            pixels = pixels[pixels[:, max_var_index].argsort()]
            median_index = len(pixels) // 2
            return split_colors(pixels[:median_index], depth + 1) + split_colors(pixels[median_index:], depth + 1)

        # 获取主色
        # 按方差排序
        dominant_lab_colors = sorted(split_colors(pixels), key=lambda x: x[1])
        color_variances = [item[1] for item in dominant_lab_colors]

        # 获取最小方差值
        min_variance = color_variances[0]
        # 过滤出方差在 50 以内的颜色
        close_colors = [color for color, variance in dominant_lab_colors if variance < min(min_variance * 2, 50)]
        close_colors = [color for color, variance in dominant_lab_colors]
        # 对 close_colors 按鲜艳度排序（从高到低）
        sorted_colors = sorted(close_colors, key=lambda x: LAB2HSV(x.astype(np.uint8))[1], reverse=True)

        # 转换为 RGB 列表
        dominant_rgb_colors = [LAB2RGB(color.astype(np.uint8)) for color in sorted_colors]

        return "中位切分+前后", [RGBColor.from_array(rgb) for rgb in dominant_rgb_colors]


def project_onto_pc1(pixels: NDArray[np.uint8]) -> NDArray[np.float64]:
    """对像素做 PCA，并返回其在第一主成分方向上的投影值"""
    X = pixels.astype(np.float64)
    # 中心化
    X_centered = X - X.mean(axis=0)
    # SVD分解得到主轴
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    pc1 = Vt[0]  # 第一主成分方向
    projections = X_centered @ pc1  # 投影值
    return projections


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


def split_colors_svd(pixels: NDArray, depth: int = 0) -> List[Tuple[NDArray[np.float64], float, int]]:
    if len(pixels) == 0:
        return []

    variance: float = np.average(pixels.var(axis=0), weights=[1, 1, 1])  # type: ignore
    if variance < 50 or depth >= 200:
        mean_color = pixels.mean(axis=0)  # 计算均值
        return [(mean_color, variance, len(pixels))]  # 返回 (均值, 方差, 数量) 元组

    # 投影到 PC1 上并排序
    projections = project_onto_pc1(pixels)
    sorted_indices = np.argsort(projections)
    pixels_sorted = pixels[sorted_indices]

    # 中位切分(最大化类间方差)
    best_index = bcv_best_split_index(projections[sorted_indices])
    left = split_colors_svd(pixels_sorted[:best_index], depth + 1)
    right = split_colors_svd(pixels_sorted[best_index:], depth + 1)
    return left + right


def adaptive_cluster_count(color_centers: NDArray, method: str = "ward") -> NDArray[np.float64]:
    """
    对颜色中心进行层次聚类，自动确定聚类数，并返回最终聚类后的主色调数组。

    返回值：
        palette: k × 3 的 LAB 主色数组
    """
    Z = linkage(color_centers, method=method)
    # plt.cla()
    # dendrogram(Z, no_labels=True)
    # plt.show()

    # 提取合并距离
    merge_dists = Z[:, 2]
    diff = np.diff(merge_dists)

    # 找到跳跃最大的地方（差值最大的位置）
    best_k = 1
    max_jump_idx = np.argmax(diff)
    best_k = len(color_centers) - (max_jump_idx + 1)

    cut_index = np.argmax(Z[:, 2] > 25)
    best_k = len(color_centers) - cut_index

    # 聚类（截断）
    labels = fcluster(Z, t=best_k, criterion="maxclust")

    # 聚类后计算每类的平均颜色
    cluster_dict = defaultdict(list)
    for color, label in zip(color_centers, labels):
        cluster_dict[label].append(color)

    palette = np.array([np.mean(colors, axis=0) for colors in cluster_dict.values()])

    return palette


def median_with_svd(image: ImageType) -> Tuple[str, List[RGBColor]]:
    """
    中位切分 + PCA提取主轴

    Args:
        image (ImageType): 输入 PIL 图像

    Returns:
        Tuple[str, List[RGBColor]]: (方法名, 提取的 RGB 颜色)
    """
    image.thumbnail((500, 500), Image.Resampling.LANCZOS)
    rgb_array = np.array(image)
    lab_array = lab_cv2std(RGB2LAB(rgb_array))
    pixels = lab_array.reshape(-1, 3)  # 转换为 (N, 3)

    # 递归切分
    median_split_colors = split_colors_svd(pixels)
    means = np.array([mean for mean, _, _ in median_split_colors])
    # 根据权重重复
    counts = np.array([count for _, _, count in median_split_colors])
    min_count = np.min(counts)
    repeats = np.floor_divide(counts, min_count)
    repeats = np.maximum(repeats, 1)  # 避免出现 0 次
    color_centers = np.repeat(means, repeats, axis=0)
    print(f"{min_count=}")

    cluster_result = adaptive_cluster_count(color_centers)

    # 转换为 RGB 列表
    sorted_colors = sorted(
        lab_std2cv(cluster_result), key=lambda x: LAB2HSV(lab_std2cv(x).astype(np.uint8))[1], reverse=True
    )
    sorted_colors = cluster_result
    dominant_rgb_colors = [LAB2RGB(lab_std2cv(color).astype(np.uint8)) for color in sorted_colors]

    return "中位切分+SVD", [RGBColor.from_array(rgb) for rgb in dominant_rgb_colors]
