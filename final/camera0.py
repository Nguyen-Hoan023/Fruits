import cv2
import threading
import time
from ultralytics import YOLO
from PIL import Image, ImageTk
import tkinter as tk
import os

# === Load mô hình YOLOv8 (đã train xong) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "train", "weights", "best.pt")
model = YOLO(MODEL_PATH)

# === GUI Setup ===
window = tk.Tk()
window.title("Fruit Detection - YOLOv8")

label = tk.Label(window)
label.pack()

# === Camera Setup ===
cap = cv2.VideoCapture(0)

# === Biến chia sẻ giữa threads ===
latest_frame = None
latest_prediction = None
lock = threading.Lock()
stop_event = threading.Event()

# === Hàm luồng camera live ===
def update_frame():
    global latest_frame
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = cv2.resize(frame, (640, 480))
        else:
            time.sleep(0.02)

# === Hàm luồng nhận diện riêng (YOLO local) ===
def detect_loop():
    global latest_prediction
    while not stop_event.is_set():
        time.sleep(0.15)
        with lock:
            frame_for_infer = latest_frame.copy() if latest_frame is not None else None

        if frame_for_infer is not None:
            results = model(frame_for_infer, verbose=False)
            with lock:
                latest_prediction = results[0]

# === Hàm hiển thị GUI ===
def update_gui():
    if stop_event.is_set():
        return

    with lock:
        frame_to_show = latest_frame.copy() if latest_frame is not None else None
        result = latest_prediction

    if frame_to_show is not None and result is not None:
        boxes = result.boxes
        names = model.names

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                label_text = f"{names[class_id]} {conf:.2f}"

                cv2.rectangle(frame_to_show, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_to_show, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        label.imgtk = imgtk
        label.configure(image=imgtk)

    window.after(50, update_gui)


def on_close():
    stop_event.set()
    if cap.isOpened():
        cap.release()
    window.destroy()

# === Khởi động các luồng ===
threading.Thread(target=update_frame, daemon=True).start()
threading.Thread(target=detect_loop, daemon=True).start()

# === Bắt đầu GUI loop ===
window.protocol("WM_DELETE_WINDOW", on_close)
update_gui()
window.mainloop()

if cap.isOpened():
    cap.release()
