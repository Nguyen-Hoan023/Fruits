import tkinter as tk  # Thư viện tạo giao diện
from PIL import Image, ImageTk, ImageFilter  # Thư viện xử lý hình ảnh
import cv2
from tkinter import filedialog  # Hộp thoại mở file
from ultralytics import YOLO  # Nhận diện YOLO
import os
import numpy as np  # Thư viện tính toán
import random

# Lớp SubWindow kế thừa từ lớp Toplevel để tạo cửa sổ nhận diện
class SubWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Nhận dạng và phân loại")
        self.geometry("600x600")

        # Định nghĩa tên lớp đối tượng
        self.class_names = {
            0: "Tao",
            1: "Chuoi",
            2: "Cam",
        }

        # Nhãn hiển thị kết quả
        self.object_result_label = tk.Label(self, text="", font=("Arial", 12))
        self.object_result_label.pack(pady=10)

        # Khung chứa ảnh nhận diện
        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)

        # Nút tải ảnh lên
        self.upload_button = tk.Button(self, text="Tải ảnh lên", command=self.upload_image, bg="orange", font=("Arial", 12))
        self.upload_button.pack(pady=10)

        # Nút đóng cửa sổ
        self.close_button = tk.Button(self, text="Đóng", command=self.destroy, bg="red", font=("Arial", 12))
        self.close_button.pack(pady=10)

        # Load mô hình YOLO
        self.load_model()

    def load_model(self):
        self.yolo_model = YOLO("C:/Fruits10/runs/detect/train/weights/best.pt")

    def upload_image(self):
        """Mở hộp thoại chọn ảnh và thực hiện nhận diện."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return

        # Đọc ảnh
        image = cv2.imread(file_path)
        orig = image.copy()

        # Nhận diện ảnh với YOLO
        yolo_results = self.yolo_model(file_path)

        # Đếm số lượng từng loại trái cây
        object_counts = self.count_objects(yolo_results)
        self.display_object_counts(object_counts)

        # Vẽ bounding box và nhãn lên ảnh
        for result in yolo_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ bounding box
                class_id = int(box.cls[0])  # Lấy ID lớp
                label = self.class_names.get(class_id, "Unknown")  # Tra tên lớp

                # Vẽ bounding box và nhãn
                cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(orig, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Lưu ảnh đã xử lý
        save_dir = "output_fruits"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(file_path))
        cv2.imwrite(save_path, orig)

        # Hiển thị ảnh
        self.display_detected_image(save_path)

    def count_objects(self, results):
        """Đếm số lượng đối tượng theo từng loại."""
        object_counts = {}
        for result in results:
            class_ids = result.boxes.cls.numpy().astype(int)
            for class_id in class_ids:
                object_class = self.class_names.get(class_id, "Unknown")
                object_counts[object_class] = object_counts.get(object_class, 0) + 1
        return object_counts

    def display_object_counts(self, object_counts):
        """Hiển thị số lượng loại trái cây đã nhận diện."""
        result_text = "Loại quả đã nhận dạng:\n"
        for object_class, count in object_counts.items():
            result_text += f"{count} : {object_class}\n"
        self.object_result_label.configure(text=result_text)

    def display_detected_image(self, image_path):
        """Hiển thị ảnh đã nhận diện trong cửa sổ."""
        img = Image.open(image_path)
        img = img.resize((400, 400), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk  # Giữ tham chiếu tránh bị thu hồi

# Hàm mở cửa sổ phụ
def open_main_program():
    sub_window = SubWindow(window)

# Hàm kết thúc chương trình
def end_program():
    window.destroy()

# Tạo giao diện chính
window = tk.Tk()
window.title("Nhận dạng và phân loại trái cây")
window.geometry("600x600")

# Load ảnh nền
background_image = Image.open(r"C:\Fruits10\final\nen.jpg")
background_image = background_image.resize((600, 600), Image.LANCZOS)
background_image = background_image.filter(ImageFilter.SHARPEN)  # Làm nét ảnh

background_photo = ImageTk.PhotoImage(background_image)

# Gán ảnh nền vào label
background_label = tk.Label(window, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Nút chạy chương trình
execute_button = tk.Button(window, text="Chạy chương trình", command=open_main_program, bg="green", font=("Arial", 12))
execute_button.place(x=220, y=450)

# Nút kết thúc chương trình
end_button = tk.Button(window, text="Kết thúc chương trình", command=end_program, bg="red", font=("Arial", 12))
end_button.place(x=220, y=500)

# Chạy giao diện
window.mainloop()
