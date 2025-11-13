import cv2
import threading
import time
from roboflow import Roboflow
from PIL import Image, ImageTk
import tkinter as tk

# === Roboflow Setup ===
rf = Roboflow(api_key="oRhVKcgEoIwPHg4VZWJ6")
project = rf.workspace().project("fruit02")  # tên project
model = project.version(4).model             # version model (đổi nếu bạn đang dùng version khác)

# === GUI Setup ===
window = tk.Tk()
window.title("Fruit Detection - Roboflow")

label = tk.Label(window)
label.pack()

# === Camera Setup ===
cap = cv2.VideoCapture(0)

# === Biến chia sẻ giữa threads ===
latest_frame = None
latest_prediction = None
lock = threading.Lock()

# === Hàm luồng camera live ===
def update_frame():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = cv2.resize(frame, (640, 640))
        time.sleep(0.03)  # ~30 FPS

# === Hàm luồng nhận diện riêng ===
def detect_loop():
    global latest_prediction
    while True:
        time.sleep(1.0)  # gửi ảnh mỗi 1 giây
        with lock:
            if latest_frame is not None:
                cv2.imwrite("temp.jpg", latest_frame.copy())
        try:
            prediction = model.predict("temp.jpg", confidence=40, overlap=30).json()
            with lock:
                latest_prediction = prediction
        except Exception as e:
            print("Lỗi dự đoán:", e)

# === Hàm hiển thị GUI ===
def update_gui():
    with lock:
        frame_to_show = latest_frame.copy() if latest_frame is not None else None
        preds = latest_prediction

    if frame_to_show is not None and preds:
        for det in preds["predictions"]:
            x, y, w, h = int(det["x"]), int(det["y"]), int(det["width"]), int(det["height"])
            class_name = det["class"]
            cv2.rectangle(frame_to_show, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
            cv2.putText(frame_to_show, class_name, (x - w // 2, y - h // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        label.imgtk = imgtk
        label.configure(image=imgtk)

    window.after(50, update_gui)

# === Khởi động các luồng ===
threading.Thread(target=update_frame, daemon=True).start()
threading.Thread(target=detect_loop, daemon=True).start()

# === Bắt đầu GUI loop ===
update_gui()
window.mainloop()

cap.release()
