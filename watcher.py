import tkinter as tk
from tkinter import Label, Button, Toplevel, Canvas
from PIL import Image, ImageTk
import subprocess
import uuid
import os
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import time

CANON_PICS_FOLDER = r"C:\workspace\python\CANON PICS"

class PhotoHandler(FileSystemEventHandler):
    def __init__(self, app):
        self.app = app

    def on_created(self, event):
        if event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            self.app.add_photo(event.src_path)

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.withdraw()

        self.fullscreen_win = Toplevel(root)
        self.fullscreen_win.attributes('-fullscreen', True)
        self.fullscreen_win.configure(bg='black')
        self.fullscreen_win.withdraw()

        self.photo_label = Label(self.fullscreen_win, bg='black')
        self.photo_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        btn_style = {
            'font': ('Helvetica', 24),
            'width': 4,
            'height': 2,
            'borderwidth': 0,
            'highlightthickness': 0,
            'relief': 'flat'
        }

        self.accept_btn = Button(self.fullscreen_win, text="✓", command=self.print_image,
                                 bg='#4CAF50', fg='white', **btn_style)
        self.reject_btn = Button(self.fullscreen_win, text="×", command=self.close_overlay,
                                 bg='#F44336', fg='white', **btn_style)

        self.prev_btn = Button(self.fullscreen_win, text="←", command=self.prev_photo,
                               bg='gray', fg='white', **btn_style)
        self.next_btn = Button(self.fullscreen_win, text="→", command=self.next_photo,
                               bg='gray', fg='white', **btn_style)

        self.timer_canvas = Canvas(self.fullscreen_win, width=100, height=100, bg='black', highlightthickness=0)
        self.timer_arc = self.timer_canvas.create_arc(10, 10, 90, 90, start=90, extent=360, style='arc', outline='white', width=6)

        self.photos = []
        self.current_index = -1
        self.timer_running = False

        event_handler = PhotoHandler(self)
        observer = Observer()
        observer.schedule(event_handler, CANON_PICS_FOLDER, recursive=True)
        observer.start()

    def add_photo(self, photo_path):
        self.photos.append(photo_path)
        self.current_index = len(self.photos) - 1
        self.display_current_photo()

    def display_current_photo(self):
        if not self.photos:
            return

        self.timer_running = False
        photo_path = self.photos[self.current_index]
        img = Image.open(photo_path)

        w, h = self.fullscreen_win.winfo_screenwidth(), self.fullscreen_win.winfo_screenheight()
        ratio = min(w / img.width, h / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(img)
        self.photo_label.configure(image=imgtk, bg='black')
        self.photo_label.imgtk = imgtk

        self.fullscreen_win.deiconify()
        self.photo_label.lift()
        self.accept_btn.place(relx=0.4, rely=0.9, anchor=tk.CENTER)
        self.reject_btn.place(relx=0.6, rely=0.9, anchor=tk.CENTER)
        self.prev_btn.place(relx=0.05, rely=0.5, anchor=tk.CENTER)
        self.next_btn.place(relx=0.95, rely=0.5, anchor=tk.CENTER)

        self.timer_canvas.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
        self.timer_canvas.lift()
        self.start_timer()

    def close_overlay(self):
        self.fullscreen_win.withdraw()
        self.accept_btn.place_forget()
        self.reject_btn.place_forget()
        self.prev_btn.place_forget()
        self.next_btn.place_forget()
        self.timer_canvas.place_forget()
        self.timer_running = False

    def print_image(self):
        if self.current_index >= 0:
            subprocess.run(["lp", self.photos[self.current_index]])
        self.close_overlay()

    def start_timer(self):
        self.timer_running = True

        def countdown():
            total = 60
            for i in range(total):
                if not self.timer_running:
                    return
                angle = 360 * (1 - i / total)
                self.timer_canvas.itemconfig(self.timer_arc, extent=angle)
                time.sleep(1)
            if self.timer_running:
                self.close_overlay()

        threading.Thread(target=countdown, daemon=True).start()

    def next_photo(self):
        if self.current_index < len(self.photos) - 1:
            self.current_index += 1
            self.display_current_photo()

    def prev_photo(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_photo()

if __name__ == '__main__':
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
