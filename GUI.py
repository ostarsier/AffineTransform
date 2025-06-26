import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageTk


# 初始化全局变量
clicked_points = []  # 鼠标点击的图像坐标
robot_coordinates = []  # 对应的机械臂坐标
image = None
image_display = None
canvas = None
photo = None
affine_matrix = None  # 仿射矩阵


def load_image():
    """加载标定板图片"""
    global image, image_display, photo, canvas
    filepath = filedialog.askopenfilename(
        title="选择标定板图片",
        filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if not filepath:
        return

    # 加载图片
    image = cv2.imread(filepath)
    if image is None:
        messagebox.showerror("错误", "加载图片失败。")
        return

    image_display = image.copy()

    # 将 OpenCV 图像转换为 PIL 图像
    image_rgb = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    photo = ImageTk.PhotoImage(image_pil)

    # 更新画布
    canvas.config(width=photo.width(), height=photo.height())
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)


def on_canvas_click(event):
    """记录点击的像素点并可视化"""
    global image_display, photo, clicked_points, robot_coordinates

    x, y = event.x, event.y
    clicked_points.append((x, y))

    # 在图像上绘制点击点
    cv2.circle(image_display, (x, y), radius=2, color=(255, 255, 255), thickness=-1)

    # 更新显示
    image_rgb = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    photo = ImageTk.PhotoImage(image_pil)
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)

    # 在标签中显示点击的坐标
    label_clicks.config(text=f"已点击的点: {clicked_points}")


def load_excel():
    """加载机械臂坐标数据"""
    global robot_coordinates
    filepath = filedialog.askopenfilename(
        title="选择机械臂坐标 Excel 文件",
        filetypes=[("Excel 文件", "*.xlsx;*.xls")]
    )
    if not filepath:
        return None

    try:
        robot_data = pd.read_excel(filepath)
        if not all(col in robot_data.columns for col in ["x", "y"]):
            messagebox.showerror("错误", "Excel 文件必须包含 'x', 'y' 列。")
            return None
        robot_coordinates = robot_data[["x", "y"]].values
        label_robot_coords.config(text=f"机械臂坐标: {robot_coordinates}")
        messagebox.showinfo("成功", "机械臂坐标已加载。")
    except Exception as e:
        messagebox.showerror("错误", f"加载 Excel 文件失败: {e}")
        return None


def manual_input_coordinates():
    """手动输入机械臂坐标"""
    def save_input():
        try:
            input_x = float(entry_x.get())
            input_y = float(entry_y.get())
            robot_coordinates.append((input_x, input_y))
            label_robot_coords.config(text=f"机械臂坐标: {robot_coordinates}")
            top.destroy()  # 关闭输入窗口
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字。")

    top = tk.Toplevel(root)
    top.title("输入机械臂坐标")

    tk.Label(top, text="输入 x 坐标:").pack()
    entry_x = tk.Entry(top)
    entry_x.pack()

    tk.Label(top, text="输入 y 坐标:").pack()
    entry_y = tk.Entry(top)
    entry_y.pack()

    tk.Button(top, text="保存", command=save_input).pack()
    top.mainloop()


def calculate_affine_matrix():
    """计算仿射矩阵并验证"""
    global affine_matrix

    if not clicked_points:
        messagebox.showwarning("警告", "请在图像上点击至少一个点。")
        return

    robot_end_xy = np.array(robot_coordinates)
    if len(clicked_points) != len(robot_end_xy):
        messagebox.showerror("错误", "图像坐标与机械臂坐标点数不匹配。")
        return

    affine_matrix, _ = cv2.estimateAffine2D(np.array(clicked_points), robot_end_xy)
    if affine_matrix is None:
        messagebox.showerror("错误", "计算仿射矩阵失败。")
        return

    messagebox.showinfo("成功", f"已计算仿射矩阵:\n{affine_matrix}")
    label_matrix.config(text=f"仿射矩阵:\n{affine_matrix}")


def test_conversion():
    """验证最后一个点击点的转换"""
    global affine_matrix

    if affine_matrix is None:
        messagebox.showwarning("警告", "仿射矩阵尚未计算。")
        return

    if not clicked_points:
        messagebox.showwarning("警告", "请在图像上点击至少一个点。")
        return

    test_x, test_y = clicked_points[-1]
    robot_x = (affine_matrix[0][0] * test_x) + (affine_matrix[0][1] * test_y) + affine_matrix[0][2]
    robot_y = (affine_matrix[1][0] * test_x) + (affine_matrix[1][1] * test_y) + affine_matrix[1][2]
    messagebox.showinfo("转换结果", f"最后一个点 ({test_x}, {test_y}) 转换为机械臂坐标: ({robot_x}, {robot_y})")

def export_to_txt():
    """导出点击的图像坐标和对应的机械臂坐标、仿射矩阵以及验证数据到 txt 文件"""
    if len(clicked_points) == 0 or len(robot_coordinates) == 0 or affine_matrix is None:
        messagebox.showwarning("警告", "没有足够的数据导出，或仿射矩阵尚未计算。")
        return

    filepath = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("文本文件", "*.txt")],
        title="保存为"
    )
    if not filepath:
        return

    with open(filepath, "w") as file:
        # 写入仿射矩阵
        file.write("仿射矩阵:\n")
        for row in affine_matrix:
            file.write(" ".join([f"{elem:.6f}" for elem in row]) + "\n")

        # 写入点击的图像坐标和对应的机械臂坐标
        file.write("\n点击的图像坐标和对应的机械臂坐标:\n")
        for img_point, robot_point in zip(clicked_points, robot_coordinates):
            file.write(f"图像坐标: {img_point} -> 机械臂坐标: {robot_point}\n")

        # 验证数据：图像坐标转换后的机械臂坐标
        file.write("\n验证数据（图像坐标转换到机械臂坐标）：\n")
        for img_point in clicked_points:
            x_camera, y_camera = img_point
            robot_x = (affine_matrix[0][0] * x_camera) + (affine_matrix[0][1] * y_camera) + affine_matrix[0][2]
            robot_y = (affine_matrix[1][0] * x_camera) + (affine_matrix[1][1] * y_camera) + affine_matrix[1][2]
            file.write(f"图像坐标: {img_point} 转换后的机械臂坐标: ({robot_x:.6f}, {robot_y:.6f})\n")

    messagebox.showinfo("成功", f"数据已导出至 {filepath}")

# 创建主窗口
root = tk.Tk()
root.title("标定工具")

# 创建画布
canvas = tk.Canvas(root, bg="gray")
canvas.pack(fill=tk.BOTH, expand=True)

# 鼠标点击事件绑定
canvas.bind("<Button-1>", on_canvas_click)

# 创建按钮
frame = tk.Frame(root)
frame.pack(fill=tk.X, side=tk.BOTTOM)

btn_load_image = tk.Button(frame, text="加载图片", command=load_image)
btn_load_image.pack(side=tk.LEFT, padx=5, pady=5)

btn_load_excel = tk.Button(frame, text="加载机械臂坐标 Excel", command=load_excel)
btn_load_excel.pack(side=tk.LEFT, padx=5, pady=5)

btn_manual_input = tk.Button(frame, text="手动输入机械臂坐标", command=manual_input_coordinates)
btn_manual_input.pack(side=tk.LEFT, padx=5, pady=5)

btn_calculate_matrix = tk.Button(frame, text="计算仿射矩阵", command=calculate_affine_matrix)
btn_calculate_matrix.pack(side=tk.LEFT, padx=5, pady=5)

btn_test_conversion = tk.Button(frame, text="验证转换", command=test_conversion)
btn_test_conversion.pack(side=tk.LEFT, padx=5, pady=5)

btn_export = tk.Button(frame, text="导出为 TXT", command=export_to_txt)
btn_export.pack(side=tk.LEFT, padx=5, pady=5)

btn_quit = tk.Button(frame, text="退出", command=root.quit)
btn_quit.pack(side=tk.RIGHT, padx=5, pady=5)

# 创建标签显示已点击的点和仿射矩阵
label_clicks = tk.Label(root, text="已点击的点: []", anchor="w", justify="left")
label_clicks.pack(fill=tk.X, padx=10, pady=5)

label_robot_coords = tk.Label(root, text="机械臂坐标: []", anchor="w", justify="left")
label_robot_coords.pack(fill=tk.X, padx=10, pady=5)

label_matrix = tk.Label(root, text="仿射矩阵: ", anchor="w", justify="left")
label_matrix.pack(fill=tk.X, padx=10, pady=5)

# 运行主循环
root.mainloop()
