from collections import defaultdict
from typing import Callable, List, Tuple

import numpy as np
from color_space import *
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as ImageType
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

ColorExtractor = Callable[[ImageType], Tuple[str, List[RGBColor]]]


def filter_colors_lab(pixels: NDArray[np.uint8], threshold: int = 20) -> NDArray[np.uint8]:
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


def split_colors_median(
    pixels: NDArray,
    depth: int = 0,
    max_depth: int = 3,
) -> List[Tuple[NDArray[np.float64], float, int]]:
    """递归中位切分"""
    if len(pixels) == 0:
        return []

    if depth >= max_depth:
        variance: float = np.average(pixels.var(axis=0), weights=[1, 1, 1])  # type: ignore
        mean_color = pixels.mean(axis=0)  # 计算均值
        return [(mean_color, variance, len(pixels))]  # 返回 (均值, 方差, 数量) 元组

    # 找到方差最大的通道
    var_l, var_a, var_b = np.var(pixels, axis=0)
    # 可调权重，主要是L的权重
    max_var_index = np.argmax([var_l, var_a, var_b])

    # 按该通道的中位数切分
    pixels = pixels[pixels[:, max_var_index].argsort()]
    median_index = len(pixels) // 2
    left = split_colors_median(pixels[:median_index], depth + 1, max_depth)
    right = split_colors_median(pixels[median_index:], depth + 1, max_depth)
    return left + right


def split_colors_svd(
    pixels: NDArray,
    depth: int = 0,
    max_depth: int = 200,
    var_theshold: float = 50,
) -> List[Tuple[NDArray[np.float64], float, int]]:
    """递归切分，在PC1轴上最大化类间方差切分"""
    if len(pixels) == 0:
        return []

    variance: float = np.average(pixels.var(axis=0), weights=[1, 1, 1])  # type: ignore
    if variance < var_theshold or depth >= max_depth:
        mean_color = pixels.mean(axis=0)  # 计算均值
        return [(mean_color, variance, len(pixels))]  # 返回 (均值, 方差, 数量) 元组

    # 投影到 PC1 上并排序
    projections = project_onto_pc1(pixels)
    sorted_indices = np.argsort(projections)
    pixels_sorted = pixels[sorted_indices]

    # 中位切分(最大化类间方差)
    best_index = bcv_best_split_index(projections[sorted_indices])
    left = split_colors_svd(pixels_sorted[:best_index], depth + 1, max_depth, var_theshold)
    right = split_colors_svd(pixels_sorted[best_index:], depth + 1, max_depth, var_theshold)
    return left + right


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


def adaptive_cluster_count(color_centers: NDArray, show_tree: bool = False) -> NDArray[np.float64]:
    """
    对颜色中心进行层次聚类，自动确定聚类数，并返回最终聚类后的主色调数组。

    Returns:
        palette: k × 3 的 LAB 主色数组
    """
    Z = linkage(color_centers, method="average")
    if show_tree:
        plt.cla()
        dendrogram(
            Z,
            # color_threshold=0.1,
            no_labels=True,
            truncate_mode="level",  # 或 'lastp'，按节点数或层数剪枝
            p=20,  # 保留最后的20个簇（你可以调）
        )
        plt.show(block=False)

    # 截断，Lab空间距离10表示颜色大致可以被认为是不同色调
    cut_index = np.argmax(Z[:, 2] > 10)
    best_k = len(color_centers) - cut_index
    labels = fcluster(Z, t=best_k, criterion="maxclust")

    # 聚类后计算每类的平均颜色
    cluster_dict = defaultdict(list)
    for color, label in zip(color_centers, labels):
        cluster_dict[label].append(color)
    palette = np.array([np.mean(colors, axis=0) for colors in cluster_dict.values()])
    return palette


def just_average(image: ImageType) -> Tuple[str, List[RGBColor]]:
    """RGB/LAB  颜色空间的平均值"""
    np_rgb = np.array(image)
    np_lab = RGB2LAB(np_rgb)

    avg_rgb = np_rgb.mean(axis=(0, 1))
    avg_lab = np_lab.mean(axis=(0, 1))
    r1, g1, b1 = map(int, avg_rgb[:3])
    r2, g2, b2 = map(int, LAB2RGB(avg_lab))
    return "RGB/LAB 均值", [RGBColor(r1, g1, b1), RGBColor(r2, g2, b2)]


def single_solution(
    image: ImageType,
    use_pca: bool = False,
    use_pre: bool = True,
) -> Tuple[str, List[RGBColor]]:
    """
    提取单颜色的最终方案：递归切分 + HLS前后处理

    Args:
        image (ImageType): 输入 PIL 图像
        use_pca (bool): 用于比较两种切分策略
        use_pre (bool): 用于比较前处理的作用

    Returns:
        Tuple[str, List[RGBColor]]: (方法名, 提取的 RGB 颜色)
    """
    method_name = ("PC1切分" if use_pca else "中位切分") + ("+前处理" if use_pre else "")

    rgb_array = np.array(image)
    lab_array = lab_cv2std(RGB2LAB(rgb_array))
    pixels = lab_array.reshape(-1, 3)  # 转换为 (N, 3)

    if use_pre:
        hls_array = RGB2HLS(rgb_array).reshape(-1, 3)
        pixels = pixels[filter_colors_hls(hls_array)]

        if len(pixels) <= 20:  # 绝大部分色彩都被过滤了，所以不再使用中位切分，直接平均
            avg_color = pixels.mean(axis=0)
            rgb_color = LAB2RGB(avg_color)
            r, g, b = map(int, rgb_color)
            return method_name, [RGBColor(r, g, b)]

    # 按方差排序
    dominant_lab_colors = sorted(split_colors_median(pixels), key=lambda x: x[1])
    color_variances = [item[1] for item in dominant_lab_colors]

    # 过滤出方差在 50 以内的颜色
    min_var = color_variances[0]
    close_colors = [lab_std2cv(c) for c, var, _ in dominant_lab_colors if var < 50]
    # 另一个方案，适用于具有极端一致的大片颜色，也是最终采用的方案
    close_colors = [lab_std2cv(c) for c, var, _ in dominant_lab_colors if var < min(min_var * 2, 50)]

    # 按饱和度排序
    sorted_colors = sorted(close_colors, key=lambda x: LAB2HSV(x.astype(np.uint8))[1], reverse=True)
    dominant_rgb_colors = [LAB2RGB(color.astype(np.uint8)) for color in sorted_colors]

    # 虽然最后就选第一个作为结果了，但是为了方便测试，此处把经过方差筛选的颜色都输出了
    return method_name, [RGBColor.from_array(rgb) for rgb in dominant_rgb_colors]


def multi_solution(image: ImageType, show_tree: bool = False) -> Tuple[str, List[RGBColor]]:
    """
    提取多颜色的最终方案：PC1主轴递归切分（不进行前处理）后层次聚类，自适应确定簇数

    Args:
        image (ImageType): 输入 PIL 图像
        show_tree (bool): 是否显示聚类树

    Returns:
        Tuple[str, List[RGBColor]]: (方法名, 提取的 RGB 颜色)
    """
    image.thumbnail((500, 500), Image.Resampling.LANCZOS)
    rgb_array = np.array(image)
    lab_array = lab_cv2std(RGB2LAB(rgb_array))
    pixels = lab_array.reshape(-1, 3)  # 转换为 (N, 3)

    # 递归切分
    median_split_colors = split_colors_svd(pixels)
    point_filter = np.array([count > 8 for _, _, count in median_split_colors])  # 拒绝孤立点
    means = np.array([mean for mean, _, _ in median_split_colors])[point_filter]
    # 根据权重重复
    # TODO: 或许可以直接将距离计算修改为带权重版本
    counts = np.array([count for _, _, count in median_split_colors])[point_filter]
    min_count = np.min(counts)
    repeats = np.floor_divide(counts, min_count)
    repeats = np.maximum(repeats, 1)  # 避免出现 0 次
    color_centers = np.repeat(means, repeats, axis=0)
    if len(color_centers) != 1:
        cluster_result = adaptive_cluster_count(color_centers, show_tree=show_tree)
    else:
        cluster_result = color_centers

    # 转换为 RGB 列表
    sorted_colors = sorted(
        lab_std2cv(cluster_result), key=lambda x: LAB2HSV(lab_std2cv(x).astype(np.uint8))[1], reverse=True
    )
    sorted_colors = cluster_result
    dominant_rgb_colors = [LAB2RGB(lab_std2cv(color).astype(np.uint8)) for color in sorted_colors]

    return "PC1切分+层次聚类", [RGBColor.from_array(rgb) for rgb in dominant_rgb_colors]


def just_cluster(image: ImageType) -> Tuple[str, List[RGBColor]]:
    """
    测试一下单纯聚类的效果如何，结果效率很低，效果也不太好

    Args:
        image (ImageType): 输入 PIL 图像

    Returns:
        Tuple[str, List[RGBColor]]: (方法名, 提取的 RGB 颜色)
    """
    image.thumbnail((500, 500), Image.Resampling.LANCZOS)
    rgb_array = np.array(image)
    lab_array = lab_cv2std(RGB2LAB(rgb_array))
    pixels = lab_array.reshape(-1, 3)  # 转换为 (N, 3)

    cluster_result = adaptive_cluster_count(pixels)

    # 转换为 RGB 列表
    sorted_colors = sorted(
        lab_std2cv(cluster_result), key=lambda x: LAB2HSV(lab_std2cv(x).astype(np.uint8))[1], reverse=True
    )
    sorted_colors = cluster_result
    dominant_rgb_colors = [LAB2RGB(lab_std2cv(color).astype(np.uint8)) for color in sorted_colors]

    return "just_cluster", [RGBColor.from_array(rgb) for rgb in dominant_rgb_colors]
