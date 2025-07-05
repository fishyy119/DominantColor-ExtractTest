from typing import Callable, NamedTuple, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

CArray = NDArray[np.uint8]
ColorConverter = Callable[[CArray], CArray]


class RGBColor(NamedTuple):
    R: int
    G: int
    B: int

    @classmethod
    def from_array(cls, array: Sequence[int] | NDArray[np.uint8]) -> "RGBColor":
        return cls(array[0], array[1], array[2])


def ColorConvertBase(img: CArray, cvCOLOR: int) -> CArray:
    """
    进行颜色空间转换

    Args:
        img (CArray): 形状为 (H, W, 3) / (L, 3) / (3,), 表示三通道颜色数据
        cvCOOLOR (int): cv2的类型转换标识符

    Returns:
        output_img (CArray): 形状与输入相同，但颜色通道转换。
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

# cv2未提供直接转换，使用RGB中转
LAB2HLS: ColorConverter = lambda img: RGB2HLS(LAB2RGB(img))
LAB2HSV: ColorConverter = lambda img: RGB2HSV(LAB2RGB(img))
HLS2LAB: ColorConverter = lambda img: RGB2LAB(HLS2RGB(img))

"""
cv2中的Lab：
L: 0~255，由标准的0~100线性映射而来
a: 0~255，相对于标准的-128~127存在一个偏移
b: 同上
"""


def lab_cv2std(lab: NDArray[np.uint8]) -> NDArray[np.float32]:
    """
    将cv2的Lab转换为标准Lab

    Args:
        lab (CArray): 形状为 (H, W, 3) / (L, 3) / (3,), 表示三通道颜色数据

    Returns:
        std_lab (CArray): 转换后的标准Lab
    """
    lab_ = lab.astype(np.float32)
    lab_[..., 0] = lab_[..., 0] * 100 / 255.0  # L: 0~255 -> 0~100
    lab_[..., 1] -= 128.0  # a: 0~255 -> -128~127
    lab_[..., 2] -= 128.0  # b: 同上
    return lab_


def lab_std2cv(lab: NDArray[np.floating]) -> NDArray[np.uint8]:
    """
    将标准Lab转换为cv2的Lab

    Args:
        lab (CArray): 形状为 (H, W, 3) / (L, 3) / (3,), 表示三通道颜色数据

    Returns:
        cv2_lab (CArray): 转换后的cv2 Lab
    """
    lab_ = lab.astype(np.float32)
    lab_[..., 0] = lab_[..., 0] * 255.0 / 100.0  # L: 0~100 -> 0~255
    lab_[..., 1] += 128.0  # a: -128~127 -> 0~255
    lab_[..., 2] += 128.0  # b: 同上
    return lab_.astype(np.uint8)
