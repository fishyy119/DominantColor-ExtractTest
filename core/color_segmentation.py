from color_space import RGB2LAB, LAB2RGB
from PIL import Image
from skimage.segmentation import felzenszwalb
import numpy as np

from PIL.Image import Image as ImageType


def felzenszwalb_lab_segmentation(
    image: ImageType,
    scale: float = 200,
    sigma: float = 0.8,
    min_size: int = 200,
) -> ImageType:
    """
    使用 Felzenszwalb 算法对图像进行 LAB 空间分割，并用每个区域的 LAB 均值重新上色。

    参数:
        image: 输入的 PIL 图像 (RGB)。
        scale: Felzenszwalb 分割的 scale 参数，控制区域大小。
        sigma: Felzenszwalb 分割的 sigma 参数，用于高斯模糊。
        min_size: Felzenszwalb 分割的 min_size 参数，最小区域大小。

    返回:
        处理后的 PIL 图像。
    """
    img_rgb = np.array(image)  # H x W x 3，dtype=uint8

    # 转LAB色彩空间（0~255，uint8）
    lab_img_uint8 = RGB2LAB(img_rgb)

    # 转float32并归一化到 0~1
    lab_img = lab_img_uint8.astype(np.float32) / 255.0

    # 执行Felzenszwalb分割，使用传入参数
    segments = felzenszwalb(lab_img, scale=scale, sigma=sigma, min_size=min_size)

    # 创建一个新的LAB图像用于存储均值重建图像
    new_lab_img = np.zeros_like(lab_img)

    for seg_val in np.unique(segments):
        mask = segments == seg_val
        for ch in range(3):
            avg = lab_img[:, :, ch][mask].mean()
            new_lab_img[:, :, ch][mask] = avg

    # 反归一化并转回 uint8
    new_lab_img_uint8 = np.clip(new_lab_img * 255.0, 0, 255).astype(np.uint8)

    # LAB 转 RGB（使用已有 LAB2RGB 函数）
    result_rgb = LAB2RGB(new_lab_img_uint8)

    return Image.fromarray(result_rgb)
