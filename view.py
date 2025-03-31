import tkinter as tk
from PIL import Image, ImageTk
import win32gui
import win32ui
import win32con
import time
import cv2
import numpy as np
import os

WINDOW_TITLE = "Remote Live View-Fenster"  # Titel vom EOS Utility Live View Fenster
image_region = None  # (left, top, width, height) des tatsächlichen Bildes
debug_windows = {}  # Dictionary to store debug windows
DEBUG_MODE = True  # Set to False to disable debug windows


def detect_frames(pil_img):
    """Detect the actual content area of the live view display"""
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    original_img = img_cv.copy()
    
    # STEP 1: First, detect the overall UI window (gray outer frame)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Debug - show edge detection
    if DEBUG_MODE:
        edge_img = Image.fromarray(cv2.cvtColor(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB))
        show_debug_window(edge_img, "1. Edge Detection")
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour as outer frame
    outer_rect = None
    max_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > max_area and w > 300 and h > 300:  # Minimum size filtering
            outer_rect = (x, y, w, h)
            max_area = area
    
    if not outer_rect:
        print("Could not detect outer frame")
        return None
    
    # Draw outer rectangle on debug image
    outer_debug = img_cv.copy()
    x, y, w, h = outer_rect
    cv2.rectangle(outer_debug, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if DEBUG_MODE:
        outer_debug_pil = Image.fromarray(cv2.cvtColor(outer_debug, cv2.COLOR_BGR2RGB))
        show_debug_window(outer_debug_pil, "2. Outer Frame")
    
    # STEP 2: Now detect the actual content area (the dark area with the photo)
    # Extract the region within the outer frame
    roi = img_cv[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to find dark regions (the photo area)
    # For a dark area in a lighter UI, we want to find pixels darker than the UI
    _, threshold = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Debug - show threshold
    if DEBUG_MODE:
        threshold_img = Image.fromarray(cv2.cvtColor(cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB))
        show_debug_window(threshold_img, "3. Dark Area Threshold")
    
    # Clean up the threshold with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    threshold_cleaned = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    threshold_cleaned = cv2.morphologyEx(threshold_cleaned, cv2.MORPH_OPEN, kernel)
    
    if DEBUG_MODE:
        cleaned_img = Image.fromarray(cv2.cvtColor(cv2.cvtColor(threshold_cleaned, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB))
        show_debug_window(cleaned_img, "4. Cleaned Threshold")
    
    # Find contours in the cleaned threshold
    contours_thresh, _ = cv2.findContours(threshold_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw all contours from threshold on debug image
    thresh_contour_img = roi.copy()
    cv2.drawContours(thresh_contour_img, contours_thresh, -1, (0, 0, 255), 2)
    if DEBUG_MODE:
        thresh_contour_pil = Image.fromarray(cv2.cvtColor(thresh_contour_img, cv2.COLOR_BGR2RGB))
        show_debug_window(thresh_contour_pil, "5. Dark Area Contours")
    
    # Find the largest dark contour that's likely to be the photo area
    content_rect = None
    max_content_area = 0
    
    for cnt in contours_thresh:
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        area = rw * rh
        
        # Filter out small areas and areas that aren't somewhat rectangular
        if area < 10000 or min(rw, rh) < 50:  # Minimum size filtering
            continue
            
        aspect_ratio = max(rw, rh) / min(rw, rh)
        if aspect_ratio > 3:  # Skip very elongated shapes
            continue
            
        # Check if this is a potentially good content area
        if area > max_content_area:
            content_rect = (x + rx, y + ry, rw, rh)
            max_content_area = area
            print(f"Found potential content area: {rx},{ry} {rw}x{rh}, area={area}")
    
    # STEP 3: Alternative approach - try to find the content area using color properties
    if not content_rect:
        print("Trying alternative detection method...")
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate the average saturation in grid cells
        # Content areas typically have higher saturation than UI elements
        cell_size = 20
        rows, cols = roi.shape[0] // cell_size, roi.shape[1] // cell_size
        saturation_map = np.zeros((rows, cols), dtype=np.float32)
        
        for r in range(rows):
            for c in range(cols):
                r1, r2 = r * cell_size, (r + 1) * cell_size
                c1, c2 = c * cell_size, (c + 1) * cell_size
                cell = roi_hsv[r1:r2, c1:c2, 1]  # Saturation channel
                saturation_map[r, r] = np.mean(cell)
        
        # Normalize and threshold saturation map
        if np.max(saturation_map) > 0:
            norm_sat = (saturation_map / np.max(saturation_map) * 255).astype(np.uint8)
            _, sat_thresh = cv2.threshold(norm_sat, 150, 255, cv2.THRESH_BINARY)
            
            # Find contours in saturation map
            sat_thresh_large = cv2.resize(sat_thresh, (roi.shape[1], roi.shape[0]), 
                                          interpolation=cv2.INTER_NEAREST)
            sat_contours, _ = cv2.findContours(sat_thresh_large, cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
            
            # Find largest saturation contour
            for cnt in sat_contours:
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                area = rw * rh
                if area > max_content_area:
                    content_rect = (x + rx, y + ry, rw, rh)
                    max_content_area = area
    
    # STEP 4: Try using variance as a feature to detect image region
    if not content_rect:
        print("Trying variance-based detection...")
        # Calculate local variance (high in detailed image areas, low in UI elements)
        cell_size = 20
        rows, cols = roi.shape[0] // cell_size, roi.shape[1] // cell_size
        variance_map = np.zeros((rows, cols), dtype=np.float32)
        
        for r in range(rows):
            for c in range(cols):
                r1, r2 = r * cell_size, (r + 1) * cell_size
                c1, c2 = c * cell_size, (c + 1) * cell_size
                if r2 <= roi_gray.shape[0] and c2 <= roi_gray.shape[1]:
                    cell = roi_gray[r1:r2, c1:c2]
                    variance_map[r, c] = np.var(cell)
        
        # Normalize and threshold variance map
        variance_map_norm = variance_map.copy()
        if np.max(variance_map) > 0:
            variance_map_norm = (variance_map / np.max(variance_map) * 255).astype(np.uint8)
        
        # Visualize variance map
        if DEBUG_MODE:
            variance_vis = cv2.resize(variance_map_norm, (roi.shape[1], roi.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            variance_vis_color = cv2.applyColorMap(variance_vis, cv2.COLORMAP_JET)
            variance_vis_pil = Image.fromarray(cv2.cvtColor(variance_vis_color, cv2.COLOR_BGR2RGB))
            show_debug_window(variance_vis_pil, "6. Variance Map")
            
        # Threshold variance map to find high variance regions (likely the content)
        _, var_thresh = cv2.threshold(variance_map_norm, 100, 255, cv2.THRESH_BINARY)
        var_thresh_large = cv2.resize(var_thresh, (roi.shape[1], roi.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        var_thresh_large = cv2.morphologyEx(var_thresh_large, cv2.MORPH_CLOSE, kernel)
        var_thresh_large = cv2.morphologyEx(var_thresh_large, cv2.MORPH_OPEN, kernel)
        
        var_contours, _ = cv2.findContours(var_thresh_large, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in var_contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            area = rw * rh
            if area > 10000 and min(rw, rh) > 50:
                if area > max_content_area:
                    content_rect = (x + rx, y + ry, rw, rh)
                    max_content_area = area
                    print(f"Found by variance: {rx},{ry} {rw}x{rh}, area={area}")
    
    # STEP 5: As a last resort, estimate content location based on the outer frame
    if not content_rect and outer_rect:
        print("Using estimated content location based on outer frame...")
        # Assuming content is centered in the window with standard margins
        x, y, w, h = outer_rect
        # Apply conservative margins from each side
        margin_x = int(w * 0.10)  # 10% margin from left/right
        margin_y = int(h * 0.10)  # 10% margin from top/bottom
        content_rect = (x + margin_x, y + margin_y, w - 2*margin_x, h - 2*margin_y)
    
    # Debug - final visualization
    final_debug = original_img.copy()
    
    if outer_rect:
        ox, oy, ow, oh = outer_rect
        cv2.rectangle(final_debug, (ox, oy), (ox + ow, oy + oh), (255, 0, 0), 2)
    
    if content_rect:
        cx, cy, cw, ch = content_rect
        cv2.rectangle(final_debug, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)
    
    if DEBUG_MODE:
        final_debug_pil = Image.fromarray(cv2.cvtColor(final_debug, cv2.COLOR_BGR2RGB))
        show_debug_window(final_debug_pil, "7. Final Detection")
    
    return content_rect


def show_debug_window(img, title="Debug"):
    """Display a debug window with the provided image"""
    global debug_windows
    
    if not DEBUG_MODE:
        return
    
    # Resize image if too large
    width, height = img.size
    if width > 800 or height > 600:
        ratio = min(800 / width, 600 / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Create or update window
    if title in debug_windows:
        win, label = debug_windows[title]
        if not win.winfo_exists():
            # Window was closed, create new one
            win = tk.Toplevel()
            win.title(title)
            label = tk.Label(win)
            label.pack()
            debug_windows[title] = (win, label)
    else:
        win = tk.Toplevel()
        win.title(title)
        label = tk.Label(win)
        label.pack()
        debug_windows[title] = (win, label)
    
    # Update image
    tk_img = ImageTk.PhotoImage(img)
    label = debug_windows[title][1]
    label.configure(image=tk_img)
    label.image = tk_img  # Keep reference


def capture_window(title):
    """Capture a screenshot of the specified window"""
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        print(f"Window '{title}' not found")
        return None
    
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    width, height = right - left, bottom - top
    if width == 0 or height == 0:
        print("Window has zero size")
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


class LiveViewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EOS LiveView Mirror")
        
        # Create frame for buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add buttons
        self.calibrate_button = tk.Button(self.button_frame, 
                                          text="Recalibrate", 
                                          command=self.recalibrate)
        self.calibrate_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.toggle_debug_button = tk.Button(self.button_frame, 
                                            text="Toggle Debug", 
                                            command=self.toggle_debug)
        self.toggle_debug_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.save_button = tk.Button(self.button_frame, 
                                    text="Save Image", 
                                    command=self.save_current_image)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Add status label
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Initializing...")
        self.status_label = tk.Label(self.button_frame, 
                                    textvariable=self.status_var, 
                                    bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.RIGHT, fill=tk.X, padx=5, pady=5, expand=True)
        
        # Image display
        self.label = tk.Label(self.root)
        self.label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.initialized = False
        self.current_image = None
        self.update_stream()
    
    def toggle_debug(self):
        global DEBUG_MODE
        DEBUG_MODE = not DEBUG_MODE
        self.status_var.set(f"Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")
        
        # Close all debug windows if turning off
        if not DEBUG_MODE:
            for title in list(debug_windows.keys()):
                win, _ = debug_windows[title]
                if win.winfo_exists():
                    win.destroy()
    
    def save_current_image(self):
        if self.current_image:
            # Create directory if it doesn't exist
            if not os.path.exists("saved_images"):
                os.makedirs("saved_images")
                
            # Save with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"saved_images/eos_live_{timestamp}.png"
            self.current_image.save(filename)
            self.status_var.set(f"Image saved as {filename}")
        else:
            self.status_var.set("No image to save")
    
    def recalibrate(self):
        global image_region
        image_region = None
        self.initialized = False
        self.status_var.set("Status: Recalibrating...")
    
    def update_stream(self):
        global image_region
        start = time.time()
        
        try:
            img = capture_window(WINDOW_TITLE)
            if img:
                # Try to detect frames if not initialized or every 60 seconds
                if not self.initialized or (time.time() % 60 < 1) or not image_region:
                    self.status_var.set("Status: Detecting frame...")
                    crop_rect = detect_frames(img)
                    if crop_rect:
                        image_region = crop_rect
                        self.initialized = True
                        self.status_var.set(f"Status: Content area detected at {crop_rect}")
                    else:
                        self.status_var.set("Status: Failed to detect content area")
                
                # Crop to the detected region if available
                if image_region:
                    x, y, w, h = image_region
                    cropped_img = img.crop((x, y, x + w, y + h))
                    self.current_image = cropped_img
                    imgtk = ImageTk.PhotoImage(cropped_img)
                else:
                    self.current_image = img
                    imgtk = ImageTk.PhotoImage(img)
                    
                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)
                
                # Adjust window size
                button_height = self.button_frame.winfo_reqheight()
                self.root.geometry(f"{imgtk.width()}x{imgtk.height()+button_height}")
            else:
                self.status_var.set("Status: Cannot capture window")
        
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Exception: {e}")
        
        # Calculate delay to maintain frame rate
        delay = max(1, int((1 / 30 - (time.time() - start)) * 1000))
        self.root.after(delay, self.update_stream)


if __name__ == "__main__":
    root = tk.Tk()
    app = LiveViewApp(root)
    root.mainloop()
