import tkinter as tk
from tkinter import Label, Button, Toplevel, Canvas
from PIL import Image, ImageTk
import win32gui
import win32ui
import win32con
import cv2
import numpy as np
import os
import time
import subprocess
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import argparse

WINDOW_TITLE = "Remote Live View-Fenster"  # Title from EOS Utility Live View window
CANON_PICS_FOLDER = r"C:\workspace\python\CANON PICS"
image_region = None
debug_windows = {}
DEBUG_MODE = False  # Set to False by default for production

class PhotoHandler(FileSystemEventHandler):
    def __init__(self, app):
        self.app = app

    def on_created(self, event):
        if event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            self.app.add_photo(event.src_path)

class BoothApp:
    def __init__(self, root, printable=False):
        self.root = root
        self.root.title("Photo Booth")
        self.printable = printable  # Store the printable flag
        
        # Define color scheme
        self.colors = {
            'bg': '#1A1A1A',           # Dark background
            'primary': '#3A7CA5',      # Soft blue
            'secondary': '#2F4858',    # Dark blue-gray
            'accent': '#16BAC5',       # Turquoise
            'success': '#2E7D32',      # Dark green
            'warning': '#E65100',      # Dark orange
            'text': '#FFFFFF',         # White
            'text_secondary': '#B0BEC5' # Light gray
        }
        
        # Configure root window for fullscreen
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg=self.colors['bg'])
        
        # Initialize state variables
        self.current_mode = "live"  # "live" or "review"
        self.photos = []
        self.current_index = -1
        self.timer_running = False
        self.timer_id = None  # Store the after() ID
        self.initialized = False
        self.current_image = None
        self.start_time = time.time()  # Add start time tracking
        
        # Create main display area
        self.display_label = Label(self.root, bg=self.colors['bg'])
        self.display_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Create timer window
        self.timer_window = tk.Toplevel(self.root)
        self.timer_window.withdraw()  # Hide initially
        self.timer_window.overrideredirect(True)  # Remove window decorations
        self.timer_window.attributes('-alpha', 0.6)  # Set transparency
        
        # Define UI constants
        self.BOTTOM_AREA_HEIGHT = 150  # Height of reserved bottom area
        
        # Create bottom panel with dark background
        self.bottom_panel = tk.Frame(
            self.root,
            bg=self.colors['bg'],
            height=self.BOTTOM_AREA_HEIGHT
        )
        self.bottom_panel.place(relx=0, rely=1, relwidth=1, anchor='sw')
        
        # Create timer in the bottom panel (removed rounded rectangle code)
        self.timer_canvas = Canvas(
            self.bottom_panel,
            width=150,
            height=150,
            bg=self.colors['bg'],
            highlightthickness=0
        )
        
        # Create outer circle
        self.timer_canvas.create_oval(10, 10, 140, 140, outline='#333333', width=8)
        
        # Create progress arc
        self.timer_arc = self.timer_canvas.create_arc(
            10, 10, 140, 140,
            start=90,
            extent=360,
            style='arc',
            outline=self.colors['accent'],
            width=8
        )
        
        # Create countdown text
        self.timer_text = self.timer_canvas.create_text(
            75, 75,
            text="60",
            font=('Helvetica', 42, 'bold'),
            fill='white'
        )
        
        # Create control panel
        self.create_control_panel()
        
        # Setup file watcher
        event_handler = PhotoHandler(self)
        observer = Observer()
        observer.schedule(event_handler, CANON_PICS_FOLDER, recursive=True)
        observer.start()
        
        # Start live view
        self.update_stream()
        
        # Add keyboard binding
        self.root.bind('<Control-r>', self.recalculate_view_zone)
        self.root.bind('<Escape>', self.close_app)  # Add ESC binding
        
        # Add a flag to allow recalculation
        self.allow_recalculation = True
    
    def create_control_panel(self):
        # Updated button styles with modern look
        btn_style = {
            'font': ('Helvetica', 24),
            'width': 4,
            'height': 2,
            'borderwidth': 0,
            'highlightthickness': 0,
            'relief': 'flat',
            'justify': 'center',
            'cursor': 'hand2'  # Hand cursor on hover
        }
        
        # Create control frame with rounded corners
        self.control_frame = tk.Frame(
            self.root,
            bg=self.colors['bg'],
            padx=20,
            pady=20
        )
        self.control_frame.place(relx=0.5, rely=0.9, anchor=tk.CENTER)
        
        # Live view and review control frames
        self.live_controls = tk.Frame(self.control_frame, bg=self.colors['bg'])
        self.review_controls = tk.Frame(self.control_frame, bg=self.colors['bg'])
        
        # Update review control buttons with modern styling
        self.accept_btn = Button(
            self.review_controls,
            text="✔️",
            command=self.print_image,
            bg=self.colors['success'],
            fg=self.colors['text'],
            activebackground=self.colors['success'],
            activeforeground=self.colors['text'],
            **btn_style
        )
        
        self.back_btn = Button(
            self.review_controls,
            text="🔙",
            command=self.back_to_live,
            bg=self.colors['warning'],
            fg=self.colors['text'],
            activebackground=self.colors['warning'],
            activeforeground=self.colors['text'],
            **btn_style
        )
        
        # Navigation buttons with updated styling
        nav_btn_style = {
            'font': ('Helvetica', 36),
            'width': 3,
            'height': 3,
            'borderwidth': 0,
            'highlightthickness': 0,
            'relief': 'flat',
            'justify': 'center',
            'bg': self.colors['secondary'],
            'fg': self.colors['text'],
            'activebackground': self.colors['primary'],
            'activeforeground': self.colors['text'],
            'cursor': 'hand2'
        }
        
        # Create navigation buttons with modern styling
        self.prev_btn = Button(self.root, text="←", command=self.prev_photo, **nav_btn_style)
        self.next_btn = Button(self.root, text="→", command=self.next_photo, **nav_btn_style)
        
        # Initially show live controls
        self.show_live_controls()
    
    def show_live_controls(self):
        self.review_controls.pack_forget()
        self.live_controls.pack()
        self.timer_canvas.place_forget()
        self.prev_btn.place_forget()
        self.next_btn.place_forget()
    
    def show_review_controls(self):
        self.live_controls.pack_forget()
        self.review_controls.pack()
        
        # Only show accept (print) button if printable flag is set
        if self.printable:
            self.accept_btn.pack(side=tk.LEFT, padx=10)
        self.back_btn.pack(side=tk.LEFT, padx=10)
        
        # Position navigation buttons on the sides
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Place navigation buttons on the sides, vertically centered
        self.prev_btn.place(relx=0.05, rely=0.5, anchor=tk.CENTER)
        self.next_btn.place(relx=0.95, rely=0.5, anchor=tk.CENTER)
        
        self.timer_canvas.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
    
    def back_to_live(self):
        self.current_mode = "live"
        self.show_live_controls()
        self.stop_timer()  # Stop any running timer
        self.display_current_photo()  # This will show live view
    
    def add_photo(self, photo_path):
        self.photos.append(photo_path)
        self.current_index = len(self.photos) - 1
        self.current_mode = "review"
        self.show_review_controls()
        self.display_current_photo()
    
    def display_current_photo(self):
        if self.current_mode == "live":
            # Live view will be handled by update_stream
            return
        
        if not self.photos or self.current_index < 0:
            return
        
        photo_path = self.photos[self.current_index]
        img = Image.open(photo_path)
        
        # Resize image to fit screen while maintaining aspect ratio
        w, h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        ratio = min(w / img.width, h / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        
        imgtk = ImageTk.PhotoImage(img)
        self.display_label.configure(image=imgtk, bg=self.colors['bg'])
        self.display_label.imgtk = imgtk
        
        self.start_timer()
    
    def print_image(self):
        if self.current_index >= 0:
            subprocess.run(["lp", self.photos[self.current_index]])
        self.back_to_live()
    
    def discard_photo(self):
        if self.current_index >= 0:
            # Remove the photo from the list
            self.photos.pop(self.current_index)
            self.current_index = min(self.current_index, len(self.photos) - 1)
            if self.current_index >= 0:
                self.display_current_photo()
            else:
                self.back_to_live()
    
    def prev_photo(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_photo()
    
    def next_photo(self):
        if self.current_index < len(self.photos) - 1:
            self.current_index += 1
            self.display_current_photo()
    
    def start_timer(self):
        # Stop any existing timer
        self.stop_timer()
        
        # Reset the timer arc and text
        self.timer_canvas.itemconfig(self.timer_arc, extent=360)
        self.timer_canvas.itemconfig(self.timer_text, text="60")
        
        # Update timer visual style
        self.timer_canvas.itemconfig(
            self.timer_arc,
            outline=self.colors['accent'],
            width=10
        )
        self.timer_canvas.itemconfig(
            self.timer_text,
            fill=self.colors['text'],
            font=('Helvetica', 42, 'bold')
        )
        
        # Start timer
        self.timer_running = True
        self.timer_count = 0
        self.update_timer()
    
    def stop_timer(self):
        """Stop any running timer and cancel scheduled updates"""
        self.timer_running = False
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None
    
    def update_timer(self):
        if not self.timer_running:
            return
            
        total = 60
        self.timer_count += 1
        
        if self.timer_count > total:
            self.stop_timer()
            self.back_to_live()
            return
            
        # Update arc with smoother color transitions
        angle = 360 * (1 - self.timer_count / total)
        self.timer_canvas.itemconfig(self.timer_arc, extent=angle)
        
        # Update text
        remaining = total - self.timer_count
        self.timer_canvas.itemconfig(self.timer_text, text=str(remaining))
        
        # Smoother color transitions for timer
        if remaining <= 10:
            color = '#F44336'  # Red
        elif remaining <= 20:
            color = self.colors['warning']  # Orange
        else:
            color = self.colors['accent']  # Default accent color
            
        self.timer_canvas.itemconfig(self.timer_arc, outline=color)
        
        self.timer_id = self.root.after(1000, self.update_timer)
    
    def capture_window(self, title):
        """Capture a screenshot of the specified window"""
        hwnd = win32gui.FindWindow(None, title)
        if not hwnd:
            return None
        
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        width, height = right - left, bottom - top
        if width == 0 or height == 0:
            return None
        
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bitmap)
        
        save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)
        
        bmpinfo = save_bitmap.GetInfo()
        bmpstr = save_bitmap.GetBitmapBits(True)
        
        img = Image.frombuffer('RGB',
                              (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                              bmpstr, 'raw', 'BGRX', 0, 1)
        
        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        
        return img
    
    def update_stream(self):
        global image_region
        if self.current_mode != "live":
            self.root.after(33, self.update_stream)
            return
        
        try:
            img = self.capture_window(WINDOW_TITLE)
            if img:
                # Only try to detect frames if necessary
                if not self.initialized and ((time.time() - self.start_time) < 5):
                    crop_rect = self.detect_frames(img)
                    if crop_rect:
                        image_region = crop_rect
                        self.initialized = True
                
                # Crop to the detected region if available
                if image_region:
                    x, y, w, h = image_region
                    cropped_img = img.crop((x, y, x + w, y + h))
                    self.current_image = cropped_img
                else:
                    self.current_image = img

                # Get screen dimensions
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                
                # Calculate scaling ratio while maintaining aspect ratio
                img_width, img_height = self.current_image.size
                width_ratio = screen_width / img_width
                height_ratio = screen_height / img_height
                scale_ratio = min(width_ratio, height_ratio)
                
                # Calculate new dimensions
                new_width = int(img_width * scale_ratio)
                new_height = int(img_height * scale_ratio)
                
                # Resize image
                resized_img = self.current_image.resize(
                    (new_width, new_height),
                    Image.Resampling.LANCZOS
                )
                
                # Create PhotoImage
                imgtk = ImageTk.PhotoImage(resized_img)
                self.display_label.imgtk = imgtk
                self.display_label.configure(image=imgtk)
            
        except Exception as e:
            print(f"Error in update_stream: {e}")
        
        self.root.after(33, self.update_stream)
    
    def detect_frames(self, pil_img):
        """Detect the actual content area of the live view display"""
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        original_img = img_cv.copy()
        
        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour as outer frame
        outer_rect = None
        max_area = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > max_area and w > 300 and h > 300:
                outer_rect = (x, y, w, h)
                max_area = area
        
        if not outer_rect:
            return None
        
        # Extract the region within the outer frame
        x, y, w, h = outer_rect
        roi = img_cv[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to find dark regions
        _, threshold = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up the threshold
        kernel = np.ones((5, 5), np.uint8)
        threshold_cleaned = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        threshold_cleaned = cv2.morphologyEx(threshold_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the cleaned threshold
        contours_thresh, _ = cv2.findContours(threshold_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest dark contour
        content_rect = None
        max_content_area = 0
        
        for cnt in contours_thresh:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            area = rw * rh
            if area > 10000 and min(rw, rh) > 50:
                if area > max_content_area:
                    content_rect = (x + rx, y + ry, rw, rh)
                    max_content_area = area
        
        return content_rect

    def recalculate_view_zone(self, event=None):
        """Recalculate the view zone from the live view window"""
        global image_region
        
        if self.current_mode != "live":
            return
            
        # Reset initialization flags
        self.initialized = False
        image_region = None
        self.start_time = time.time()
        
        # Force an immediate frame capture and analysis
        img = self.capture_window(WINDOW_TITLE)
        if img:
            crop_rect = self.detect_frames(img)
            if crop_rect:
                image_region = crop_rect
                self.initialized = True
                print("View zone recalculated successfully")
            else:
                print("Could not detect view zone")

    def close_app(self, event=None):
        """Close the application when ESC is pressed"""
        self.root.quit()

    def create_rounded_rectangle(self, canvas, x1, y1, x2, y2, radius=25, **kwargs):
        """Helper function to create a rounded rectangle"""
        points = [
            x1 + radius, y1,
            x1 + radius, y1,
            x2 - radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1 + radius,
            x1, y1
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Photo Booth Application')
    parser.add_argument('--printable', action='store_true', 
                       help='Enable printing functionality')
    args = parser.parse_args()
    
    root = tk.Tk()
    app = BoothApp(root, printable=args.printable)
    root.mainloop() 