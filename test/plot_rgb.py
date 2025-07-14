import _test_init
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 网格点的步长
step = 16
levels = np.arange(0, 256, step)
n = len(levels)

# 生成六个面的小立方体的坐标
# 立方体范围是 [0, 255]


def create_face_voxels():
    # 初始化空数组，False表示无立方体
    voxels = np.zeros((n, n, n), dtype=bool)
    colors = np.zeros((n, n, n, 3))

    # 6个面
    # 面1: R=0
    voxels[0, :, :] = True
    colors[0, :, :, 0] = 0 / 255
    colors[0, :, :, 1] = levels[:, None] / 255
    colors[0, :, :, 2] = levels[None, :] / 255

    # 面2: R=255 (最大)
    voxels[-1, :, :] = True
    colors[-1, :, :, 0] = 255 / 255
    colors[-1, :, :, 1] = levels[:, None] / 255
    colors[-1, :, :, 2] = levels[None, :] / 255

    # 面3: G=0
    voxels[:, 0, :] = True
    colors[:, 0, :, 0] = levels[:, None] / 255
    colors[:, 0, :, 1] = 0 / 255
    colors[:, 0, :, 2] = levels[None, :] / 255

    # 面4: G=255
    voxels[:, -1, :] = True
    colors[:, -1, :, 0] = levels[:, None] / 255
    colors[:, -1, :, 1] = 255 / 255
    colors[:, -1, :, 2] = levels[None, :] / 255

    # 面5: B=0
    voxels[:, :, 0] = True
    colors[:, :, 0, 0] = levels[:, None] / 255
    colors[:, :, 0, 1] = levels[None, :] / 255
    colors[:, :, 0, 2] = 0 / 255

    # 面6: B=255
    voxels[:, :, -1] = True
    colors[:, :, -1, 0] = levels[:, None] / 255
    colors[:, :, -1, 1] = levels[None, :] / 255
    colors[:, :, -1, 2] = 255 / 255

    return voxels, colors


def plot1():
    voxels, colors = create_face_voxels()

    fig = plt.figure(figsize=(8, 8))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    ax.voxels(voxels, facecolors=colors, edgecolors="gray", linewidth=0.3, shade=False)

    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, n)
    ax.set_axis_off()
    ax.axis("equal")

    plt.tight_layout()


def plot2(g_value: int = 128):
    """绘制 RGB 空间中 G 分量固定的平面 (R-B 变化)，展示感知不均匀性"""
    size = 32  # 控制分辨率
    r = np.linspace(0, 255, size, dtype=int)
    b = np.linspace(0, 255, size, dtype=int)

    # 生成图像矩阵：shape = (size, size, 3)
    img = np.zeros((size, size, 3), dtype=np.uint8)

    for i in range(size):
        for j in range(size):
            img[j, i] = [r[i], g_value, b[j]]  # 注意：行-列与x-y顺序相反

    # 显示图像
    plt.figure(figsize=(5, 5))
    plt.imshow(img, extent=[0, 255, 0, 255], origin="lower")
    plt.xlabel("Red")
    plt.ylabel("Blue")
    plt.title(f"RGB平面（G = {g_value}）")

    # 可选：绘制参考线
    plt.grid(color="white", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()


if __name__ == "__main__":
    # plot1()
    plot2(128 + 64 + 32)
    plt.show()
