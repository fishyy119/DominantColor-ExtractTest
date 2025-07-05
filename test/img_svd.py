# pyright: reportAttributeAccessIssue=false
# pyright: reportUnusedExpression=false

import matplotlib.pyplot as plt
import numpy as np
from _test_init import DATA_ROOT
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from PIL.Image import Image as ImageType

from core.color_extraction import *

np.random.seed(42)


def plot_in_ab(
    lab: NDArray,
    S: NDArray,
    Vt: NDArray,
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
    plt.figure(figsize=(8, 8))
    plt.scatter(aa_coords - 128, bb_coords - 128, c=rgb_pts / 255.0, s=15)
    plt.xlabel("a")
    plt.ylabel("b")
    plt.grid(True)
    plt.axis("equal")

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

    fig = plt.figure(figsize=(12, 6))

    # RGB 分布
    ax: Axes3D = fig.add_subplot(1, 2, 1, projection="3d")  # type: ignore
    ax.scatter(pixels_rgb[:, 0], pixels_rgb[:, 1], pixels_rgb[:, 2], c=pixels_rgb / 255.0, marker=".", s=1)  # type: ignore
    ax.set_title("RGB Color Distribution")
    ax.set_xlabel("R"), ax.set_ylabel("G"), ax.set_zlabel("B")

    # Lab 分布（注意不能直接用 Lab 值设为颜色）
    ax2: Axes3D = fig.add_subplot(1, 2, 2, projection="3d")  # type: ignore
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

    mean_lab = lab_flat.mean(axis=0)
    X = lab_flat - mean_lab  # 零中心化
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    print(S)
    show_color_distribution(img_rgb, img_lab)

    lab_flat[:, 0] = 128  # 去掉L维度
    mean_lab = lab_flat.mean(axis=0)
    U, S, Vt = np.linalg.svd(lab_flat - mean_lab, full_matrices=False)
    plot_in_ab(lab_flat, S, Vt)

    plt.show()


if __name__ == "__main__":
    main()
