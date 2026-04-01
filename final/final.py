import tkinter as tk  # Thư viện tạo giao diện
from PIL import Image, ImageTk, ImageFilter  # Thư viện xử lý hình ảnh
import cv2
from tkinter import filedialog, messagebox  # Hộp thoại mở file
from ultralytics import YOLO  # Nhận diện YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Lớp SubWindow kế thừa từ lớp Toplevel để tạo cửa sổ nhận diện
class SubWindow(tk.Toplevel):
    _shared_model = None

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
        model_path = os.path.join(PROJECT_ROOT, "runs", "detect", "train", "weights", "best.pt")
        if not os.path.exists(model_path):
            messagebox.showerror("Lỗi mô hình", f"Không tìm thấy mô hình: {model_path}")
            self.destroy()
            return

        if SubWindow._shared_model is None:
            SubWindow._shared_model = YOLO(model_path)

        self.yolo_model = SubWindow._shared_model

    def upload_image(self):
        """Mở hộp thoại chọn ảnh và thực hiện nhận diện."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path or not hasattr(self, "yolo_model"):
            return

        # Đọc ảnh
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Lỗi ảnh", "Không thể đọc ảnh đã chọn.")
            return

        orig = image.copy()

        # Nhận diện ảnh với YOLO
        yolo_results = self.yolo_model(orig, verbose=False)

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
        save_dir = os.path.join(PROJECT_ROOT, "output_fruits")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(file_path))
        cv2.imwrite(save_path, orig)

        # Hiển thị ảnh trực tiếp để tránh I/O trung gian
        self.display_detected_image(orig)

    def count_objects(self, results):
        """Đếm số lượng đối tượng theo từng loại."""
        object_counts = {}
        for result in results:
            if result.boxes is None or result.boxes.cls is None:
                continue
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for class_id in class_ids:
                object_class = self.class_names.get(class_id, "Unknown")
                object_counts[object_class] = object_counts.get(object_class, 0) + 1
        return object_counts

    def display_object_counts(self, object_counts):
        """Hiển thị số lượng loại trái cây đã nhận diện."""
        result_text = "Loại quả đã nhận dạng:\n"
        if not object_counts:
            result_text += "Không phát hiện được trái cây nào."
        else:
            for object_class, count in object_counts.items():
                result_text += f"{count} : {object_class}\n"
        self.object_result_label.configure(text=result_text)

    def display_detected_image(self, image_bgr):
        """Hiển thị ảnh đã nhận diện trong cửa sổ."""
        rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_img)
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
background_path = os.path.join(BASE_DIR, "nen.jpg")
background_image = Image.open(background_path)
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
