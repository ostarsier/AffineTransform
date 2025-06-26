import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageTk


# --- 全局变量初始化 ---
# 用于存储用户在图像上点击的点的像素坐标 (x, y)
clicked_points = []
# 用于存储与点击的像素点相对应的机械臂世界坐标 (x, y)
robot_coordinates = []
# 存储加载的原始OpenCV图像对象
image = None
# 存储用于显示的图像对象，会在上面绘制点击点等信息
image_display = None
# Tkinter的画布组件，用于显示图像和接收鼠标点击
canvas = None
# Tkinter兼容的图像对象，用于在画布上显示
photo = None
# 存储计算出的2x3仿射变换矩阵
affine_matrix = None


def load_image():
    """
    通过文件对话框加载一张用于标定的图片。
    加载后，图片会显示在主窗口的画布上。
    """
    global image, image_display, photo, canvas
    # 打开文件选择对话框，让用户选择图片文件
    filepath = filedialog.askopenfilename(
        title="选择标定板图片",
        filetypes=[("图片文件", "*.*")]
    )
    # 如果用户取消选择，则直接返回
    if not filepath:
        return

    # 使用OpenCV读取图片文件
    # cv2.imread返回一个BGR格式的numpy数组
    image = cv2.imread(filepath)
    if image is None:
        messagebox.showerror("错误", "加载图片失败。")
        return

    # 复制一份原始图像用于显示和绘制，以保留原始图像不变
    image_display = image.copy()

    # --- 将OpenCV图像转换为Tkinter兼容的格式 ---
    # 1. OpenCV的图像格式是BGR，PIL和Tkinter需要RGB格式，因此进行颜色空间转换
    image_rgb = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)
    # 2. 从numpy数组创建PIL图像对象
    image_pil = Image.fromarray(image_rgb)
    # 3. 将PIL图像对象转换为Tkinter的PhotoImage对象
    photo = ImageTk.PhotoImage(image_pil)

    # --- 更新画布 ---
    # 调整画布大小以适应新加载的图片
    canvas.config(width=photo.width(), height=photo.height())
    # 在画布的左上角(0,0)位置显示图片
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)


def on_canvas_click(event):
    """
    画布的鼠标点击事件处理函数。
    记录点击的像素坐标，并在图像上绘制一个点以进行可视化。

    Args:
        event: Tkinter事件对象，包含点击坐标 (event.x, event.y)。
    """
    global image_display, photo, clicked_points

    # 从事件对象中获取鼠标点击的x, y坐标
    x, y = event.x, event.y
    # 将坐标添加到全局列表
    clicked_points.append((x, y))

    # 在显示的图像上绘制一个红色的小圆点来标记点击位置
    cv2.circle(image_display, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

    # --- 更新画布上的图像 ---
    # 同样需要将修改后的OpenCV图像转换为Tkinter格式
    image_rgb = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    photo = ImageTk.PhotoImage(image_pil)
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)

    # 更新UI标签，显示所有已点击的点的列表
    label_clicks.config(text=f"已点击的点: {clicked_points}")


def load_excel():
    """
    通过文件对话框加载包含机械臂坐标的Excel文件。
    文件应包含'x'和'y'两列。
    """
    global robot_coordinates
    filepath = filedialog.askopenfilename(
        title="选择机械臂坐标 Excel 文件",
        filetypes=[("Excel 文件", "*.*")]
    )
    if not filepath:
        return

    try:
        # 使用pandas读取Excel文件
        robot_data = pd.read_excel(filepath)
        # 检查文件中是否包含必需的'x'和'y'列
        if not all(col in robot_data.columns for col in ["x", "y"]):
            messagebox.showerror("错误", "Excel 文件必须包含 'x', 'y' 列。")
            return
        # 提取x, y列的值，并将其转换为列表
        robot_coordinates = robot_data[["x", "y"]].values.tolist()
        # 更新UI标签显示加载的坐标
        label_robot_coords.config(text=f"机械臂坐标: {robot_coordinates}")
        messagebox.showinfo("成功", "机械臂坐标已加载。")
    except Exception as e:
        messagebox.showerror("错误", f"加载 Excel 文件失败: {e}")


def manual_input_coordinates():
    """
    弹出一个新窗口，允许用户手动输入单个机械臂坐标。
    """
    # 创建一个新的顶级窗口
    top = tk.Toplevel(root)
    top.title("手动输入机械臂坐标")

    # --- 内部函数：保存输入 --- #
    def save_input():
        """验证并保存在输入框中输入的坐标。"""
        try:
            # 从输入框获取文本并转换为浮点数
            input_x = float(entry_x.get())
            input_y = float(entry_y.get())
            # 将新坐标添加到全局列表中
            robot_coordinates.append((input_x, input_y))
            # 更新UI标签
            label_robot_coords.config(text=f"机械臂坐标: {robot_coordinates}")
            # 关闭输入窗口
            top.destroy()
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字。", parent=top)

    # --- 创建输入窗口的UI组件 --- #
    tk.Label(top, text="输入 x 坐标:").pack(pady=5)
    entry_x = tk.Entry(top)
    entry_x.pack(pady=5, padx=10)

    tk.Label(top, text="输入 y 坐标:").pack(pady=5)
    entry_y = tk.Entry(top)
    entry_y.pack(pady=5, padx=10)

    tk.Button(top, text="保存", command=save_input).pack(pady=10)

    # 将焦点设置在第一个输入框，方便用户直接输入
    entry_x.focus_set()
    # 使输入窗口成为模态窗口，阻止与主窗口的交互直到此窗口关闭
    top.transient(root)
    top.grab_set()
    root.wait_window(top)


def calculate_affine_matrix():
    """
    使用至少三个对应的图像坐标点和机械臂坐标点来计算仿射变换矩阵。
    需要确保点击的图像点数量和加载的机械臂坐标点数量一致且至少为3。
    """
    global affine_matrix

    # 检查是否有足够的点来进行计算
    if len(clicked_points) < 3:
        messagebox.showwarning("警告", "请至少在图像上点击三个点。")
        return

    # 检查图像坐标和机械臂坐标的数量是否匹配
    if len(clicked_points) != len(robot_coordinates):
        messagebox.showerror("错误", f"坐标点数不匹配!\n图像点: {len(clicked_points)}, 机械臂点: {len(robot_coordinates)}")
        return

    # 准备OpenCV所需格式的源点（图像像素坐标）和目标点（机械臂世界坐标）
    # 使用前三个点进行计算
    src_pts = np.array(clicked_points[:3], dtype=np.float32)
    dst_pts = np.array(robot_coordinates[:3], dtype=np.float32)

    # 使用OpenCV的getAffineTransform函数计算2x3的仿射矩阵
    affine_matrix = cv2.getAffineTransform(src_pts, dst_pts)

    if affine_matrix is not None:
        # 将计算出的矩阵显示在UI标签上
        matrix_str = "\n".join(["  ".join([f"{elem:8.4f}" for elem in row]) for row in affine_matrix])
        label_matrix.config(text=f"仿射矩阵:\n{matrix_str}")
        messagebox.showinfo("成功", "仿射矩阵计算成功。")
    else:
        messagebox.showerror("错误", "计算仿射矩阵失败。可能是因为三个点共线。")


def test_conversion():
    """
    使用已计算的仿射矩阵，验证最后一个点击的图像点到机械臂坐标的转换。
    并将转换结果与实际对应的机械臂坐标（如果存在）进行比较。
    """
    if affine_matrix is None:
        messagebox.showwarning("警告", "请先计算仿射矩阵。")
        return

    if not clicked_points:
        messagebox.showwarning("警告", "请在图像上点击至少一个点以进行验证。")
        return

    # 获取最后一个点击的图像点
    last_image_point = np.array([[clicked_points[-1]]], dtype=np.float32)

    # 使用cv2.transform进行坐标转换，这是更推荐的方式
    transformed_point = cv2.transform(last_image_point, affine_matrix)
    robot_x, robot_y = transformed_point[0][0]

    # 准备显示的信息
    result_message = f"最后一个图像点 {last_image_point[0][0]} 转换为机械臂坐标: ({robot_x:.4f}, {robot_y:.4f})"

    # 如果有对应的实际机械臂坐标，则计算并显示误差
    if len(robot_coordinates) >= len(clicked_points):
        actual_robot_point = robot_coordinates[len(clicked_points) - 1]
        error = np.sqrt((robot_x - actual_robot_point[0])**2 + (robot_y - actual_robot_point[1])**2)
        result_message += f"\n\n对应的实际机械臂坐标: {actual_robot_point}"
        result_message += f"\n转换误差: {error:.4f}"

    messagebox.showinfo("转换验证结果", result_message)


def export_to_txt():
    """
    将所有标定相关数据（仿射矩阵、坐标对应关系、验证结果）导出到一个文本文件中。
    """
    if affine_matrix is None:
        messagebox.showwarning("警告", "仿射矩阵尚未计算，无法导出。")
        return

    if not clicked_points or not robot_coordinates:
        messagebox.showwarning("警告", "没有足够的坐标数据可以导出。")
        return

    # 弹出文件保存对话框
    filepath = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("文本文件", "*.txt")],
        title="导出标定数据为"
    )
    if not filepath:
        return

    try:
        with open(filepath, "w") as file:
            # 写入仿射矩阵
            file.write("--- 仿射变换矩阵 ---\n")
            np.savetxt(file, affine_matrix, fmt='%.8f')

            # 写入坐标对应关系和验证
            file.write("\n\n--- 坐标点对应关系及验证 ---\n")
            file.write(f"{'Index':<8}{'Image Coords':<20}{'Robot Coords (Actual)':<25}{'Robot Coords (Calculated)':<30}{'Error':<10}\n")
            file.write("-" * 95 + "\n")

            # 遍历所有匹配的点
            num_points = min(len(clicked_points), len(robot_coordinates))
            for i in range(num_points):
                img_pt = np.array([[clicked_points[i]]], dtype=np.float32)
                robot_pt_actual = robot_coordinates[i]

                # 进行转换
                robot_pt_calc = cv2.transform(img_pt, affine_matrix)[0][0]

                # 计算误差
                error = np.linalg.norm(np.array(robot_pt_actual) - robot_pt_calc)

                # 格式化输出字符串
                img_str = f"({img_pt[0][0][0]:.1f}, {img_pt[0][0][1]:.1f})"
                robot_actual_str = f"({robot_pt_actual[0]:.4f}, {robot_pt_actual[1]:.4f})"
                robot_calc_str = f"({robot_pt_calc[0]:.4f}, {robot_pt_calc[1]:.4f})"

                file.write(f"{i+1:<8}{img_str:<20}{robot_actual_str:<25}{robot_calc_str:<30}{error:<10.4f}\n")

        messagebox.showinfo("成功", f"数据已成功导出至 {filepath}")
    except Exception as e:
        messagebox.showerror("错误", f"导出文件失败: {e}")

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
