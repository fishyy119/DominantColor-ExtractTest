from PIL import Image
import numpy as np
import cv2

from typing import Callable, Tuple, List
from numpy.typing import NDArray

RGBColor = Tuple[int, int, int]
ColorExtractor = Callable[[Image.Image], Tuple[str, RGBColor]]
ColorConverter = Callable[[NDArray[np.uint8]], NDArray[np.uint8]]


def ColorConvertBase(img: NDArray[np.uint8], cvCOLOR: int) -> NDArray[np.uint8]:
    """
    进行颜色空间转换

    Args:
        img (NDArray[np.uint8]): 形状为 (H, W, 3) / (L, 3) / (3,), 表示三通道颜色数据
        cvCOOLOR (int): cv2的类型转换标识符

    Returns:
        output_img (NDArray[np.uint8]): 形状与输入相同，但颜色通道转换。
    """
    img = img.astype(np.uint8)
    if img.ndim == 1:
        # (3,) -> (1, 1, 3)
        expand_array = np.expand_dims(np.expand_dims(img, axis=0), axis=0)
        convert_array = cv2.cvtColor(expand_array, cvCOLOR).astype(np.uint8)
        return convert_array[0, 0, :]  # 转回 (3,)
    elif img.ndim == 2:
        # (L, 3) -> (L, 1, 3)
        expand_array = np.expand_dims(img, axis=1)
        convert_array = cv2.cvtColor(expand_array, cvCOLOR).astype(np.uint8)
        return convert_array[:, 0, :]  # 转回 (L, 3)
    else:
        return cv2.cvtColor(img, cvCOLOR).astype(np.uint8)


RGB2LAB: ColorConverter = lambda img: ColorConvertBase(img, cv2.COLOR_RGB2LAB)
LAB2RGB: ColorConverter = lambda img: ColorConvertBase(img, cv2.COLOR_LAB2RGB)

RGB2HLS: ColorConverter = lambda img: ColorConvertBase(img, cv2.COLOR_RGB2HLS)
RGB2HSV: ColorConverter = lambda img: ColorConvertBase(img, cv2.COLOR_RGB2HSV)
HLS2RGB: ColorConverter = lambda img: ColorConvertBase(img, cv2.COLOR_HLS2RGB)

LAB2HLS: ColorConverter = lambda img: RGB2HLS(LAB2RGB(img))
LAB2HSV: ColorConverter = lambda img: RGB2HSV(LAB2RGB(img))
HLS2LAB: ColorConverter = lambda img: RGB2LAB(HLS2RGB(img))


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


def rgb_average(image: Image.Image) -> Tuple[str, RGBColor]:
    """RGB 颜色空间的平均值"""
    np_img = np.array(image)
    avg_color = np_img.mean(axis=(0, 1))
    r, g, b = map(int, avg_color[:3])
    return "RGB 均值", (r, g, b)


def lab_average(image: Image.Image) -> Tuple[str, RGBColor]:
    """LAB 颜色空间的平均值，转换回 RGB"""
    np_img = np.array(image)
    np_img = RGB2LAB(np_img)
    avg_color = np_img.mean(axis=(0, 1))
    rgb_color = LAB2RGB(avg_color)
    r, g, b = map(int, rgb_color)
    return "LAB 均值", (r, g, b)


def median_cut_lab(image: Image.Image) -> Tuple[str, Tuple[int, int, int]]:
    """
    在 LAB 颜色空间内执行中位切分算法提取主色。

    Args:
        image (Image.Image): 输入 PIL 图像

    Returns:
        List[Tuple[str, Tuple[int, int, int]]]: (方法名, 提取的 RGB 颜色)
    """
    num_colors: int = 5
    rgb_array = np.array(image)
    lab_array = RGB2LAB(rgb_array)
    pixels = lab_array.reshape(-1, 3)  # 转换为 (N, 3)

    pixels = filter_saturated_colors(pixels)

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
        return "LAB 中位切分", dominant_rgb_colors[0][:3]  # type: ignore
    else:
        raise IndexError("没有提取到主色")


def final_solution(image: Image.Image) -> Tuple[str, Tuple[int, int, int]]:
    """
    最终方案

    Args:
        image (Image.Image): 输入 PIL 图像

    Returns:
        List[Tuple[str, Tuple[int, int, int]]]: (方法名, 提取的 RGB 颜色)
    """
    num_colors: int = 5
    rgb_array = np.array(image)
    lab_array = RGB2LAB(rgb_array)
    pixels = lab_array.reshape(-1, 3)  # 转换为 (N, 3)
    hls_array = RGB2HLS(rgb_array).reshape(-1, 3)

    # pixels_filted = filter_saturated_colors(pixels)
    pixels_filted = pixels[filter_colors_hls(hls_array)]

    if len(pixels_filted) <= 20:  # 绝大部分色彩都被过滤了，所以不再使用中位切分，直接平均
        avg_color = pixels.mean(axis=0)
        rgb_color = LAB2RGB(avg_color)
        r, g, b = map(int, rgb_color)
        return "最终方案", (r, g, b)
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
        print(color_variances)
        print("-----------------------------------------")

        # 获取最小方差值
        min_variance = color_variances[0]
        # 过滤出方差在 50 以内的颜色
        close_colors = [color for color, variance in dominant_lab_colors if variance < min(min_variance * 2, 50)]

        if len(close_colors) > 1:
            # **选择最鲜艳的颜色**
            result = max(
                close_colors, key=lambda x: float(np.linalg.norm(x[1:].astype(np.int16) - np.array([128, 128])))
            )
            result = max(close_colors, key=lambda x: LAB2HSV(x.astype(np.uint8))[1])
        else:
            # **选择方差最小的颜色**
            result = dominant_lab_colors[0][0]

        # 转换为 RGB
        dominant_rgb_colors = LAB2RGB(result.astype(np.uint8))

        return "最终方案", tuple(dominant_rgb_colors)
