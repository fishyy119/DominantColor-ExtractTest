import matplotlib.pyplot as plt
import numpy as np
from skimage.color import lab2rgb

# 创建 a-b 网格
res = 256  # 分辨率（图像大小）
a = np.linspace(-128, 127, res)
b = np.linspace(-128, 127, res)
aa, bb = np.meshgrid(a, b)

# 构建 L=0 的 Lab 图像：shape (res, res, 3)
lab = np.zeros((res, res, 3), dtype=np.float64)
lab[..., 0] = 50  # L = 0
lab[..., 1] = aa  # a
lab[..., 2] = bb  # b

# 转换为 RGB
rgb = lab2rgb(lab)

# 屏蔽非法颜色（lab2rgb 会裁剪，但我们可以高亮不可显示的区域）
# 可视化方式：用灰色或透明遮罩
valid_mask = np.all((rgb >= 0) & (rgb <= 1), axis=-1)

# 显示
plt.figure(figsize=(6, 6))
plt.imshow(rgb, extent=[-128, 127, -128, 127])
plt.title("L=0 plane in CIELAB space (a-b)")
plt.xlabel("a")
plt.ylabel("b")

# 可选：加遮罩（灰色区域表示非法）
rgb_invalid = np.ones_like(rgb)
rgb_invalid[..., :] = 0.2  # 暗灰
rgb[~valid_mask] = rgb_invalid[~valid_mask]

plt.show()
