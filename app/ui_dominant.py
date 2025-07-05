import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import List

from _project_init import WS_ROOT, Config
from PIL import Image, ImageTk
from PIL.Image import Image as ImageType

from core.color_extraction import *


class ColorExtractionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Color Extraction Test")
        self.root.geometry("600x500")

        self.current_index = 0
        self.image_files = []
        self.origin_image: ImageTk.PhotoImage | None = None

        # 不同的提取器，统一定义到列表中
        self.extractors: List[ColorExtractor] = [
            rgb_average,
            lab_average,
            median_cut_lab,
            final_solution,
            median_with_svd,
        ]

        ###########################################################################################
        # 顶部按钮区
        top_frame = tk.Frame(root)
        top_frame.grid(row=0, column=0, columnspan=2, pady=5)

        self.btn_open = tk.Button(top_frame, text="选择文件夹", command=self.load_folder)
        self.btn_open.grid(row=0, column=0, padx=5)

        self.btn_prev = tk.Button(top_frame, text="上一张", command=self.prev_image)
        self.btn_prev.grid(row=0, column=1, padx=5)

        self.btn_next = tk.Button(top_frame, text="下一张", command=self.next_image)
        self.btn_next.grid(row=0, column=2, padx=5)

        # 创建一个 StringVar 变量，绑定到 Label
        self.image_count_var = tk.StringVar()
        self.label_imgcount = tk.Label(top_frame, textvariable=self.image_count_var)
        self.label_imgcount.grid(row=0, column=3, padx=20)
        self.update_label()

        #######################################################################################################
        # 底部区域，显示原始图片与颜色提取结果
        # 左侧（显示原始图片）
        self.left_frame = tk.Frame(root, width=400, height=400, bg="gray")
        self.left_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.left_frame.grid_propagate(False)

        self.image_label: tk.Label = tk.Label(self.left_frame, bg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # 右侧（颜色提取结果）
        self.right_frame = tk.Frame(root, width=300, height=400, bg="white")
        self.right_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.right_frame.grid_propagate(False)

        self.color_rows: List[tk.Frame] = []  # 每行一个 Frame，对应一个提取器
        self.color_labels: List[List[tk.Label]] = []  # 每个 label 的二维数组

        for i, extractor in enumerate(self.extractors):
            row_frame = tk.Frame(self.right_frame)
            row_frame.grid(row=i, column=0, padx=5, pady=5, sticky="w")
            self.color_rows.append(row_frame)
            self.color_labels.append([])  # 初始化为空

        ##################################################################################################
        # 各行列的增长权重
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.minsize(600, 400)

    def update_label(self):
        """更新 Label 的显示内容"""
        total = len(self.image_files)
        current = self.current_index + 1 if total > 0 else 0
        self.image_count_var.set(f"图片：{current}/{total}")

    def load_folder(self):
        """选择文件夹并加载图片"""
        folder = filedialog.askdirectory(initialdir=WS_ROOT / "data")
        if not folder:
            return

        # 只获取图片文件
        self.image_files = [f for f in Path(folder).iterdir() if f.suffix.lower() in Config.img_suffixes]

        if self.image_files:
            self.current_index = 0
            self.show_image()

    def show_image(self):
        """显示当前索引的图片"""
        if not self.image_files:
            return

        image_path = self.image_files[self.current_index]
        image: ImageType = Image.open(image_path)

        # 获取 Label 的尺寸
        self.root.update_idletasks()  # 确保能获取最新的尺寸
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()
        image.thumbnail((label_width, label_height))  # 调整图片大小
        self.origin_image = ImageTk.PhotoImage(image)

        self.image_label.config(image=self.origin_image)
        self.update_label()

        for i, extractor in enumerate(self.extractors):
            row_frame = self.color_rows[i]

            # 清空之前的显示
            for label in self.color_labels[i]:
                label.destroy()
            self.color_labels[i].clear()

            method_name, colors = extractor(image)
            # 添加方法名标签
            method_label = tk.Label(
                row_frame,
                text=method_name,
                width=12,
                anchor="w",
            )
            method_label.pack(side=tk.LEFT, padx=(0, 8))  # 稍微拉开一点间距
            self.color_labels[i].append(method_label)

            for color in colors:
                label = tk.Label(row_frame, width=2, height=1, relief=tk.RIDGE)
                label.pack(side=tk.LEFT, padx=2)
                # label.config(text=f"{method_name}:{color.R},{color.G},{color.B}")  # 显示方法名
                label.config(bg=f"#{color.R:02x}{color.G:02x}{color.B:02x}")  # 设置 Label 背景颜色为提取颜色
                # brightness = 0.299 * color.R + 0.587 * color.G + 0.114 * color.B
                # label.config(fg="#000000" if brightness >= 128 else "#FFFFFF")  # 自动切换前景色
                self.color_labels[i].append(label)

    def prev_image(self):
        """显示上一张图片"""
        if self.image_files:
            if self.current_index > 0:
                self.current_index -= 1
            else:
                self.current_index = len(self.image_files) - 1
            self.show_image()

    def next_image(self):
        """显示下一张图片"""
        if self.image_files:
            if self.current_index < len(self.image_files) - 1:
                self.current_index += 1
            else:
                self.current_index = 0
            self.show_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorExtractionApp(root)
    root.mainloop()
