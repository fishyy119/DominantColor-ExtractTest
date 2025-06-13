import _project_init
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from pathlib import Path
from core.color_segmentation import felzenszwalb_lab_segmentation

from typing import List
from PIL.Image import Image as ImageType

WS_ROOT = _project_init.WS_ROOT
Config = _project_init.Config


class ImageSplitApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Color Extraction Test")
        self.root.geometry("600x500")

        self.current_index = 0
        self.last_index = -1  # 用于记录上一次显示的图片索引
        self.image_files = []
        self.origin_photo: ImageTk.PhotoImage | None = None
        self.result_photo: ImageTk.PhotoImage | None = None

        ###########################################################################################
        # 顶部按钮区
        top_frame = tk.Frame(root)
        top_frame.grid(row=0, column=0, pady=5)

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

        ###########################################################################################
        # 参数控制区：使用滑块和输入框来控制参数
        control_frame = tk.Frame(root)
        control_frame.grid(row=1, column=0, pady=5)

        # 参数变量（用于滑块和输入同步控制）
        self.scale_var = tk.DoubleVar(value=200)
        self.sigma_var = tk.DoubleVar(value=0.8)
        self.minsize_var = tk.IntVar(value=200)
        self.zoom_var = tk.DoubleVar(value=1.0)

        self._add_param_control(control_frame, "Scale", self.scale_var, 0, 1500, 50, 0)
        self._add_param_control(control_frame, "Sigma", self.sigma_var, 0.0, 5.0, 0.1, 1)
        self._add_param_control(control_frame, "MinSize", self.minsize_var, 0, 1000, 50, 2)
        self._add_param_control(control_frame, "Zoom", self.zoom_var, 0.01, 1.0, 0.01, 3)

        ###########################################################################################
        # 底部图像显示区域：原图 + 分割图
        self.img_frame = tk.Frame(root, width=400, height=400, bg="gray")
        self.img_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        self.img_frame.grid_propagate(False)

        self.origin_label = tk.Label(self.img_frame, bg="white")
        self.origin_label.grid(row=0, column=0, sticky="nsew")
        self.origin_label.grid_propagate(False)  # 禁止自动调整大小

        self.segmented_label = tk.Label(self.img_frame, bg="white")
        self.segmented_label.grid(row=0, column=1, sticky="nsew")
        self.segmented_label.grid_propagate(False)  # 禁止自动调整大小

        self.img_frame.grid_rowconfigure(0, weight=1)
        self.img_frame.grid_columnconfigure(0, weight=1)
        self.img_frame.grid_columnconfigure(1, weight=1)

        ##################################################################################################
        # 各行列的增长权重
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.minsize(600, 400)

    def _add_param_control(
        self,
        parent: tk.Widget,
        label_text: str,
        var: tk.DoubleVar | tk.IntVar,
        minval: float,
        maxval: float,
        step: float,
        row: int,
    ) -> None:
        """辅助函数：创建参数控制（横向排列 Label、Scale、Entry）"""
        frame = tk.Frame(parent)
        frame.grid(row=row, column=0, sticky="w", pady=2)

        # 横向布局：Label + Scale + Entry
        label = tk.Label(frame, text=label_text, width=8, anchor="w")
        label.pack(side=tk.LEFT, padx=(0, 5))

        scale = tk.Scale(
            frame,
            from_=minval,
            to=maxval,
            resolution=step,
            orient=tk.HORIZONTAL,
            variable=var,
            length=120,
            showvalue=False,
        )
        scale.pack(side=tk.LEFT, padx=(0, 5))

        entry = tk.Entry(frame, textvariable=var, width=4)
        entry.pack(side=tk.LEFT)

        scale.bind("<ButtonRelease-1>", lambda e: self.show_image())  # 松开鼠标时更新图片
        entry.bind("<ButtonRelease-1>", lambda e: self.show_image())  # 松开鼠标时更新图片

    def update_label(self):
        """更新 label_imgcount 的显示内容"""
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
        """显示当前索引的图片（原图与处理图）"""
        if not self.image_files:
            return

        image_path = self.image_files[self.current_index]
        image: ImageType = Image.open(image_path).convert("RGB")  # 确保为RGB格式
        # 切换到新图片时，进行的操作：
        # 1. 显示原图
        # 2. 预计算缩放值，使其长宽尺寸不超过250
        if self.current_index != self.last_index:
            self.origin_photo = self.get_resized_photoimage(image)
            self.origin_label.config(image=self.origin_photo)
            self.update_label()
            orig_w, orig_h = image.size
            factor = min(250 / orig_w, 250 / orig_h, 1.0)
            self.zoom_var.set(round(factor, 2))  # 设置缩放比例（限制最大为1.0）

        scale = self.zoom_var.get()
        new_size = (int(image.width * scale), int(image.height * scale))
        image_resized = image.resize(new_size, Image.Resampling.LANCZOS)

        # 对图像进行处理，并显示处理后的结果
        try:
            result_image: ImageType = felzenszwalb_lab_segmentation(
                image_resized,
                scale=self.scale_var.get(),
                sigma=self.sigma_var.get(),
                min_size=self.minsize_var.get(),
            )
            self.result_photo = self.get_resized_photoimage(result_image)
            self.segmented_label.config(image=self.result_photo)
        except Exception as e:
            print(f"图像处理出错：{e}")
            self.segmented_label.config(image="")  # 清空

        self.last_index = self.current_index  # 记录上一次的索引

    def get_resized_photoimage(
        self,
        image: ImageType,
    ) -> ImageTk.PhotoImage:
        """根据Label尺寸调整大小，因为有引用保持的问题，此处不进行设置"""
        self.root.update_idletasks()  # 确保能获取最新的尺寸
        label_width = self.img_frame.winfo_width() // 2 - 5
        label_height = self.img_frame.winfo_height()
        # !这个计算不对
        # label_width = label.winfo_width()
        # label_height = label.winfo_height()

        orig_w, orig_h = image.size
        scale = min(label_width / orig_w, label_height / orig_h)  # 计算缩放比例，放大或缩小均可
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        new_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(new_image)

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
    app = ImageSplitApp(root)
    root.mainloop()
