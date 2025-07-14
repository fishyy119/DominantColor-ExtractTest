import _test_init
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage.color import lab2rgb


def plot_lab_ab_circle(L_value=50, radius=128):
    """绘制 CIELab 空间中 L 固定、ab 为圆形的色域分布"""
    size = 512  # 图像尺寸
    a = np.linspace(-radius, radius, size)
    b = np.linspace(-radius, radius, size)
    aa, bb = np.meshgrid(a, b)

    # 生成圆形 mask
    mask = aa**2 + bb**2 <= radius**2

    # 创建 Lab 平面，默认黑
    lab = np.zeros((size, size, 3), dtype=np.float32)
    lab[..., 0] = L_value
    lab[..., 1] = aa
    lab[..., 2] = bb

    # 转换为 RGB
    rgb = color.lab2rgb(lab)
    rgb[~mask] = 1.0  # 圆外填白色（或你可以设为 np.nan）

    # 显示图像
    plt.figure(figsize=(5, 5))
    plt.imshow(rgb, extent=[-radius, radius, -radius, radius], origin="lower")
    plt.xlabel("a")
    plt.ylabel("b")
    plt.title(f"ab 平面 (L = {L_value})")
    plt.grid(False)
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_lab_ab_circle(L_value=75, radius=128)
