"""
Mask Creator - Image Annotation Tool

A Python/OpenCV-based tool for quickly creating binary masks by drawing on images.
Features:
- Left-click to add mask areas
- Right-click to erase mask areas
- Arrow key navigation between images
- Real-time drawing preview
- Auto-save masks as <filename>.mask.png
- Aspect ratio preservation
- Window resizing support
- Top menu bar: All White, All Black, Reset, Undo, Redo

Usage: python mask_creator.py <image_folder>
"""

import cv2
import numpy as np
import os
import sys

IS_WINDOWS = sys.platform.startswith('win32')

if IS_WINDOWS:
    import win32gui
    import win32con

class MaskCreator:
    def __init__(self, image_folder):
        # --- Initialization ---
        # Normalize the image folder path and extract folder name
        self.image_folder = os.path.normpath(image_folder.strip('\'"'))
        self.folder_name = os.path.basename(self.image_folder)

        # Load image filenames, excluding mask files
        self.images = sorted([
            f for f in os.listdir(self.image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            and not f.endswith('.mask.png')
        ])
        if not self.images:
            raise ValueError("No images found.")
        self.current_index = 0

        # Initialize drawing state
        self.drawing = False
        self.erase_mode = False
        self.points = []

        # Initialize mask and history
        self.original_image = None
        self.current_mask = None
        self.mask_exists = False
        self.history = []
        self.redo_stack = []
        self.mask_modified = False  # Flag for cache invalidation

        # UI configuration
        self.menu_items = ['All White', 'All Black', 'Undo', 'Redo', 'Remove Mask']
        self.menu_height = 30
        self.status_height = 30
        self.ui_padding = 10
        self.menu_bounds = []

        # Caching for performance
        self.cached_combined = None
        self.cached_win_size = None
        self.cached_menu = None
        self.cached_status = None
        self.cached_win_size_ui = None

        # Set up OpenCV window
        cv2.namedWindow('Mask Creator', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Mask Creator', 1280, 720)
        cv2.setMouseCallback('Mask Creator', self.handle_mouse)
        
        if IS_WINDOWS:
            self._set_minimum_window_size(800, 600)

        # Load the first image
        self.load_image()
    
    def _truncate_text(self, text, max_width, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, thickness=1):
        """
        Truncate text to fit within a specified width, adding ellipsis if necessary.
        """
        ellipsis = "..."
        ellipsis_width = cv2.getTextSize(ellipsis, font, font_scale, thickness)[0][0]
        
        if cv2.getTextSize(text, font, font_scale, thickness)[0][0] <= max_width:
            return text
        
        # Binary search for optimal truncation
        low = 0
        high = len(text)
        while low <= high:
            mid = (low + high) // 2
            candidate = ellipsis + text[mid:]
            candidate_width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
            
            if candidate_width < max_width:
                high = mid - 1
            else:
                low = mid + 1
        
        best_candidate = ellipsis + text[low:]
        if cv2.getTextSize(best_candidate, font, font_scale, thickness)[0][0] > max_width:
            return ellipsis  # Fallback if even this is too wide
        
        return best_candidate

    def _set_minimum_window_size(self, min_width, min_height):
        """
        Set a minimum size for the OpenCV window using pywin32.
        """
        hwnd = win32gui.FindWindow(None, 'Mask Creator')
        if hwnd:
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
            win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style | win32con.WS_THICKFRAME)
            win32gui.SetWindowPos(hwnd, None, 0, 0, 0, 0,
                                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | 
                                win32con.SWP_NOZORDER | win32con.SWP_FRAMECHANGED)
            min_rect = win32gui.GetWindowRect(hwnd)
            win32gui.MoveWindow(hwnd, min_rect[0], min_rect[1], 
                              max(min_width, min_rect[2] - min_rect[0]), 
                              max(min_height, min_rect[3] - min_rect[1]), True)

    def load_image(self):
        """
        Load the current image and its corresponding mask, if available.
        """
        path = os.path.join(self.image_folder, self.images[self.current_index])
        self.original_image = cv2.imread(path)
        h, w = self.original_image.shape[:2]
        mask_path = os.path.splitext(path)[0] + '.mask.png'
        # remember whether a mask file was present on disk at load time
        self.initial_mask_existed = os.path.exists(mask_path)
        self.mask_exists = self.initial_mask_existed
        self.current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if self.mask_exists else np.zeros((h, w), dtype=np.uint8)
        self.history.clear()
        self.redo_stack.clear()
        self.mask_modified = True  # Invalidate cache
        self.update_display()

    def get_window_and_display(self):
        """
        Calculate window and display dimensions, scaling, and offsets.
        Return:
          win_w, win_h,
          scale,
          (disp_w, disp_h),
          (off_x, off_y)
        """
        _, _, win_w, win_h = cv2.getWindowImageRect('Mask Creator')
        img_h, img_w = self.original_image.shape[:2]
        avail_h = win_h - self.menu_height - self.status_height
        scale = min(win_w / img_w, avail_h / img_h) if img_w and img_h else 1.0
        disp_w, disp_h = int(img_w * scale), int(img_h * scale)
        off_x, off_y = (win_w - disp_w) // 2, self.menu_height + (avail_h - disp_h) // 2
        return win_w, win_h, scale, (disp_w, disp_h), (off_x, off_y)

    def convert_coordinates(self, x, y):
        """
        Convert window coordinates to image coordinates or menu interaction.
        If y < menu_height: return ('MENU', x).
        Else map to image coords using exact int(xr/scale) — no rounding.
        """
        # Menu click region stays as-is
        if y < self.menu_height:
            return ('MENU', x)

        # Compute relative mouse pos, then project into image coords
        win_w, win_h, scale, (dw, dh), (off_x, off_y) = self.get_window_and_display()
        fx = (x - off_x) / scale
        fy = (y - (self.menu_height + (win_h - self.menu_height - self.status_height - dh)//2)) / scale

        # Clamp to [0..width-1] and [0..height-1]
        ih, iw = self.original_image.shape[:2]
        ix = min(max(int(fx),  0), iw - 1)
        iy = min(max(int(fy),  0), ih - 1)
        return (ix, iy)

    def push_history(self):
        """
        Save the current mask state to the history stack.
        """
        # store both the mask bitmap and whether it existed on disk
        self.history.append((self.current_mask.copy(), self.mask_exists))
        self.redo_stack.clear()

    def apply_fill(self, val):
        """
        Fill the entire mask with a specified value (e.g., 255 for white).
        """
        self.push_history()
        self.current_mask[:] = val
        self._save_and_refresh()

    def reset_mask(self):
        """
        Reset the mask to all black (empty).
        """
        self.apply_fill(0)

    def undo(self):
        """
        Undo the last mask modification.
        """
        if not self.history:
            return
        # save current state into redo stack
        self.redo_stack.append((self.current_mask.copy(), self.mask_exists))
        # restore last state
        restored_mask, restored_exists = self.history.pop()
        self.current_mask = restored_mask
        self.mask_exists = restored_exists

        # Only delete the file if it didn't originally exist; otherwise re-write it
        mask_path = os.path.splitext(
            os.path.join(self.image_folder, self.images[self.current_index])
        )[0] + '.mask.png'
        
        if self.mask_exists:
            cv2.imwrite(mask_path, self.current_mask)
        else:
            if os.path.exists(mask_path):
                os.remove(mask_path)
        
        self.mask_modified = True
        self._refresh_display()

    def redo(self):
        """
        Redo the last undone mask modification.
        """
        if not self.redo_stack:
            return
        # save current state
        self.history.append((self.current_mask.copy(), self.mask_exists))
        # restore next state
        restored_mask, restored_exists = self.redo_stack.pop()
        self.current_mask = restored_mask
        self.mask_exists = restored_exists

        # If redoing returns a blank mask, delete the file; otherwise save it
        # Only delete the file if it didn't originally exist; otherwise re-write it
        mask_path = os.path.splitext(
            os.path.join(self.image_folder, self.images[self.current_index])
        )[0] + '.mask.png'
        
        if self.mask_exists:
            cv2.imwrite(mask_path, self.current_mask)
        else:
            if os.path.exists(mask_path):
                os.remove(mask_path)
        
        self.mask_modified   = True
        self._refresh_display()

    def remove_mask(self):
        """
        Remove the current mask and delete the corresponding mask file.
        """
        self.push_history()
        self.current_mask[:] = 0
        mask_path = os.path.splitext(os.path.join(self.image_folder, self.images[self.current_index]))[0] + '.mask.png'
        
        # Delete the mask file if it exists
        if os.path.exists(mask_path):
            os.remove(mask_path)
        
        self.mask_exists = False  # Update the mask status
        self.mask_modified = True
        self._refresh_display()

    def _save_and_refresh(self):
        """
        Save the current mask to disk and refresh the display.
        """
        path = os.path.join(self.image_folder, self.images[self.current_index])
        cv2.imwrite(os.path.splitext(path)[0] + '.mask.png', self.current_mask)
        self.mask_exists = True
        self.mask_modified = True
        self._refresh_display()

    def _refresh_display(self):
        """
        Refresh the display by updating the UI and image preview.
        """
        self.update_display()

    def update_display(self):
        """
        Update the display, including the image, mask, and UI elements.
        """
        win_w, win_h, scale, (dw, dh), (off_x, off_y) = self.get_window_and_display()
        current_win_size = (win_w, win_h)

        # Update combined image cache when needed
        if self.mask_modified or current_win_size != self.cached_win_size:
            scaled_img = cv2.resize(self.original_image, (dw, dh))
            working_mask = self.current_mask if self.mask_exists else np.full_like(self.current_mask, 255)
            scaled_mask = cv2.resize(working_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
            mask_inv = cv2.bitwise_not(scaled_mask)
            darkened = cv2.convertScaleAbs(scaled_img, alpha=0.7, beta=0)
            scaled_mask_3ch = cv2.cvtColor(scaled_mask, cv2.COLOR_GRAY2BGR)
            mask_inv_3ch = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
            selected_part = cv2.bitwise_and(scaled_img, scaled_mask_3ch)
            non_selected_part = cv2.bitwise_and(darkened, mask_inv_3ch)
            self.cached_combined = cv2.add(selected_part, non_selected_part)
            self.cached_win_size = current_win_size
            self.mask_modified = False

        # Create canvas
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

        # Update UI elements cache
        if current_win_size != self.cached_win_size_ui:
            # Draw menu
            menu_canvas = np.zeros((self.menu_height, win_w, 3), dtype=np.uint8)
            cv2.rectangle(menu_canvas, (0, 0), (win_w, self.menu_height), (50,50,50), -1)
            x = self.ui_padding
            self.menu_bounds.clear()
            for label in self.menu_items:
                (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                x1, x2 = x, x + tw + 2*self.ui_padding
                cv2.putText(menu_canvas, label, (x + self.ui_padding//2, int(self.menu_height*0.7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                self.menu_bounds.append((label, x1, x2))
                x = x2 + self.ui_padding
            self.cached_menu = menu_canvas

            # Draw status bar
            status_canvas = np.zeros((self.status_height, win_w, 3), dtype=np.uint8)
            cv2.rectangle(status_canvas, (0, 0), (win_w, self.status_height), (50,50,50), -1)
            self.cached_status = status_canvas
            self.cached_win_size_ui = current_win_size

        # Apply cached UI elements
        canvas[0:self.menu_height, 0:win_w] = self.cached_menu
        canvas[win_h-self.status_height:win_h, 0:win_w] = self.cached_status

        # Draw combined image
        display_image = self.cached_combined.copy()
        if self.points:
            # Filter only valid integer points
            valid = [pt for pt in self.points if isinstance(pt[0], int) and isinstance(pt[1], int)]
            if valid:
                clr = (255,255,255) if not self.erase_mode else (0,0,255)
                # Scale and cast in one step
                pts_np = (np.array(valid, dtype=np.float32) * scale).astype(np.int32)
                if len(pts_np) > 1:
                    cv2.polylines(display_image, [pts_np], False, clr, 2)
        canvas[off_y:off_y+dh, off_x:off_x+dw] = display_image

        # Update dynamic status text
        self.draw_dynamic_status(canvas, win_w, win_h)
        cv2.imshow('Mask Creator', canvas)
    
    def draw_dynamic_status(self, canvas, win_w, win_h):
        """
        Draw dynamic status information (e.g., filename, mask status) on the canvas.
        """
        status_y = win_h - self.status_height
        # Left text
        filename = self.images[self.current_index]
        folder_display = self._truncate_text(self.folder_name, win_w//3)
        left_text = f"{folder_display} > {filename}"
        cv2.putText(canvas, left_text, (self.ui_padding, status_y + int(self.status_height*0.7)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # Center counter
        counter_text = f"[{self.current_index + 1}/{len(self.images)}]"
        (tw, th), _ = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(canvas, counter_text, ((win_w - tw)//2, status_y + int(self.status_height*0.7)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # Right status
        status_text = "No Mask" if not self.mask_exists else "Mask Exists"
        color = (0, 0, 255) if not self.mask_exists else (0, 255, 0)
        (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(canvas, status_text, (win_w - tw - self.ui_padding, status_y + int(self.status_height*0.7)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def handle_mouse(self, event, x, y, flags, _):
        coords = self.convert_coordinates(x, y)
        # Handle top‐bar clicks explicitly
        if event == cv2.EVENT_LBUTTONUP and coords and coords[0] == 'MENU':
            mx = coords[1]
            for label, x1, x2 in self.menu_bounds:
                if x1 <= mx <= x2:
                    if   label == 'All White':    self.apply_fill(255)
                    elif label == 'All Black':    self.reset_mask()
                    elif label == 'Undo':         self.undo()
                    elif label == 'Redo':         self.redo()
                    elif label == 'Remove Mask':  self.remove_mask()
                    # cancel any half-drawn line
                    self.drawing = False
                    self.points = []
                    return

        # If drawing, ignore any non‐image coords
        if self.drawing and (not isinstance(coords, tuple) or not isinstance(coords[0], int)):
            return

        # Unpack valid image coords for drawing
        ix = iy = None
        if isinstance(coords, tuple) and isinstance(coords[0], int):
            ix, iy = coords
                
        # Drawing motion
        if self.drawing and ix is not None:
            if event == cv2.EVENT_MOUSEMOVE:
                self.points.append((ix, iy))
                self.update_display()
            elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                self.finalize_drawing()
            return
        
        # Start drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing, self.erase_mode, self.points = True, False, [(ix, iy)]
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing, self.erase_mode, self.points = True, True, [(ix, iy)]

    def finalize_drawing(self):
        # Only keep truly numeric points
        valid_pts = [pt for pt in self.points if isinstance(pt[0], int) and isinstance(pt[1], int)]
        if len(valid_pts) >= 2:
            # close the loop
            valid_pts.append(valid_pts[0])
            mask_tmp = np.zeros_like(self.current_mask)
            # explicitly cast to int32
            cv2.fillPoly(mask_tmp, [np.array(valid_pts, dtype=np.int32)], 255)
            self.push_history()
            if self.erase_mode:
                self.current_mask = cv2.bitwise_and(self.current_mask, cv2.bitwise_not(mask_tmp))
            else:
                self.current_mask = cv2.bitwise_or(self.current_mask, mask_tmp)
            self._save_and_refresh()
        # reset drawing state
        self.drawing = False
        self.points = []

    def run(self):
        """
        Main application loop for handling user input and updating the display.
        """
        last_key_time = 0
        key_repeat_delay = 100  # ms between key repeats
        drawing_interrupt = False

        while True:
            self.update_display()
            
            key = cv2.waitKeyEx(1)
            now = cv2.getTickCount() * 1000 // cv2.getTickFrequency()
            
            # ESC to exit
            if key == 27:
                if self.drawing:
                    self.finalize_drawing()
                break
            
            # Handle key repeats with debouncing
            if key != -1 and (now - last_key_time) > key_repeat_delay:
                # Arrow key navigation
                if key in (ord('a'), 2424832):  # Left
                    drawing_interrupt = self.drawing
                    self.finalize_drawing()
                    self.current_index = (self.current_index - 1) % len(self.images)
                    self.load_image()
                    last_key_time = now
                    
                elif key in (ord('d'), 2555904):  # Right
                    drawing_interrupt = self.drawing
                    self.finalize_drawing()
                    self.current_index = (self.current_index + 1) % len(self.images)
                    self.load_image()
                    last_key_time = now
                    
                # Quick actions
                elif key == ord('r'):  # Reset mask
                    self.finalize_drawing()
                    self.reset_mask()
                    last_key_time = now
                    
                elif key == ord('z') and (cv2.getWindowProperty('Mask Creator', cv2.WND_PROP_MODALITY) >= 0):  # Undo
                    self.undo()
                    last_key_time = now
                    
                elif key == ord('y'):  # Redo
                    self.redo()
                    last_key_time = now
                    
                # Full mask operations    
                elif key == ord('w'):  # Fill white
                    self.apply_fill(255)
                    last_key_time = now
                    
                elif key == ord('b'):  # Fill black
                    self.reset_mask()
                    last_key_time = now

            # Handle drawing interruption
            if drawing_interrupt and not self.drawing:
                self.points = []
                drawing_interrupt = False

            # Continuous drawing mode
            if self.drawing and key == -1:
                self.update_display()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Entry point: Validate arguments and start the application
    if len(sys.argv) != 2:
        print("Usage: python mask_creator.py <image_folder>")
        sys.exit(1)
    MaskCreator(sys.argv[1]).run()