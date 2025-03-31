import tkinter as tk
from tkinter import Label, Button, messagebox
from PIL import Image, ImageTk
import cv2
import subprocess
import uuid
from datetime import datetime

# Automatisch Kamera mit höchster Auflösung auswählen
def select_best_camera(max_cameras=5):
    best_cam = None
    best_resolution = (0, 0)
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width * height > best_resolution[0] * best_resolution[1]:
                best_resolution = (width, height)
                best_cam = i
            cap.release()
    if best_cam is None:
        messagebox.showerror("Error", "Keine Kamera gefunden.")
        exit()
    return cv2.VideoCapture(best_cam)

cap = select_best_camera()

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kamera App")
        self.root.geometry("800x600")
        self.root.configure(bg='black')

        self.frame_label = Label(root, bg='black')
        self.frame_label.pack(fill=tk.BOTH, expand=True)

        self.overlay_label = Label(root, bg='black')
        self.overlay_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.overlay_label.lower()

        btn_style = {'font': ('Helvetica', 30, 'bold'), 'width': 6, 'height': 2, 'borderwidth': 0}

        self.accept_btn = Button(root, text="✔️", command=self.print_image, bg='#4CAF50', fg='white', **btn_style)
        self.reject_btn = Button(root, text="❌", command=self.close_overlay, bg='#F44336', fg='white', **btn_style)

        self.accept_btn.configure(relief="flat", highlightthickness=0, bd=0)
        self.reject_btn.configure(relief="flat", highlightthickness=0, bd=0)

        self.current_image = None

        self.update_frame()
        self.root.bind('<space>', self.capture_image)
        self.root.bind('<Configure>', self.resize_event)

    def resize_event(self, event):
        if self.current_image is not None:
            self.display_frame(self.current_image, self.overlay_label)

    def update_frame(self):
        ret, frame = cap.read()
        if ret:
            self.display_frame(frame, self.frame_label)
        self.root.after(10, self.update_frame)

    def capture_image(self, event):
        ret, frame = cap.read()
        if ret:
            self.current_image = frame
            self.display_frame(frame, self.overlay_label)

            self.overlay_label.lift()
            self.accept_btn.lift()
            self.reject_btn.lift()

            self.accept_btn.place(relx=0.35, rely=0.85, anchor=tk.CENTER)
            self.reject_btn.place(relx=0.65, rely=0.85, anchor=tk.CENTER)

            self.save_image(frame)

    def close_overlay(self):
        self.overlay_label.lower()
        self.accept_btn.place_forget()
        self.reject_btn.place_forget()

    def display_frame(self, frame, widget):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        w, h = self.root.winfo_width(), self.root.winfo_height()
        if w <= 1 or h <= 1:
            return

        ratio = min(w / img.width, h / img.height)
        new_size = (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
        img = img.resize(new_size, Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        widget.imgtk = imgtk
        widget.configure(image=imgtk)

    def save_image(self, frame):
        filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(filename, frame)

    def print_image(self):
        filename = f"print_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(filename, self.current_image)
        subprocess.run(["lp", filename])
        self.close_overlay()

if __name__ == '__main__':
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
    cap.release()
