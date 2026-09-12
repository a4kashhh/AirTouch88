#!/usr/bin/env python3
"""
================================================================================
Project: Minimal 8x8 LED Matrix Controller & Hand Gesture Interface
Script:  cv_matrix_controller.py

Layout:
  - Left Side:  Natural Aspect-Ratio Camera View (No Squeeze) with
                Fixed 8x8 Control Box + Hand Brightness Zone
  - Right Side: Minimalist 8x8 LED Matrix Display with Brightness & Action Buttons
================================================================================
"""

import os
import sys
import math
import time
import socket
import argparse
import urllib.request
import cv2
import numpy as np

# Optional MediaPipe import
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[WARNING] MediaPipe not installed. Running in simulation/mouse mode.")

# Optional PySerial import
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# ==============================================================================
# MODEL ASSET DOWNLOAD HELPER
# ==============================================================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def ensure_model_asset():
    if not os.path.exists(MODEL_PATH):
        print(f"[DOWNLOAD] Downloading MediaPipe Hand Landmarker model...", flush=True)
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[DOWNLOAD] Model downloaded successfully.", flush=True)
        except Exception as e:
            print(f"[DOWNLOAD ERROR] Could not download model: {e}", flush=True)


# ==============================================================================
# 1. COMMUNICATION CLIENT (UDP & SERIAL)
# ==============================================================================
class MatrixCommunicator:
    """Manages low-latency UDP packet transmission to the ESP32 matrix."""
    def __init__(self, udp_ip="10.150.46.102", udp_port=8888, serial_port=None, baud_rate=115200,
                 transpose=True, invert=True):
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.transpose = transpose
        self.invert = invert
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

        self.ser = None
        self.serial_connected = False
        if serial_port and SERIAL_AVAILABLE:
            try:
                self.ser = serial.Serial(serial_port, baud_rate, timeout=0.1)
                self.serial_connected = True
                print(f"[SERIAL] Connected to {serial_port}", flush=True)
            except Exception as e:
                print(f"[SERIAL WARNING] Could not open {serial_port}: {e}", flush=True)

        self.last_brightness_send_time = 0
        self.last_sent_brightness = -1
        self.last_sent_cmd = "IDLE"

    def send_command(self, cmd_str):
        """Sends an ASCII command string to the ESP32-C3."""
        cmd_str = cmd_str.strip()
        if not cmd_str:
            return

        payload = (cmd_str + "\n").encode('utf-8')
        try:
            self.sock.sendto(payload, (self.udp_ip, self.udp_port))
        except Exception:
            pass

        if self.ser and self.ser.is_open:
            try:
                self.ser.write(payload)
            except Exception:
                pass

        self.last_sent_cmd = cmd_str
        print(f"[TX -> ESP32] {cmd_str}", flush=True)

    def send_brightness(self, brightness_val):
        """Throttled transmission for analog continuous brightness."""
        now = time.time()
        if (now - self.last_brightness_send_time >= 0.05) and (abs(brightness_val - self.last_sent_brightness) >= 2):
            self.last_brightness_send_time = now
            self.last_sent_brightness = brightness_val
            self.send_command(f"BRIGHTNESS:{brightness_val}")

    def send_toggle_dot(self, r, c, grid=None):
        """Toggles dot at row r, col c, sending complete frame to ensure sync."""
        global grid_dots
        target = grid if grid is not None else grid_dots
        self.send_frame(target)

    def send_frame(self, grid):
        """
        Sends complete 64-bit frame buffer to ESP32: FRAME:<64 bits>.
        Applies row<->col transposition and polarity inversion as required.
        """
        # 1. Transpose if hardware rows/cols are swapped
        if self.transpose:
            dev_grid = grid.T
        else:
            dev_grid = grid

        # 2. Build 64-bit bitstring with polarity correction
        bits = []
        for r in range(8):
            for c in range(8):
                val = dev_grid[r, c]
                if self.invert:
                    # Inverted: 1 (ON in UI) -> '0', 0 (OFF in UI) -> '1'
                    bit_char = '0' if val else '1'
                else:
                    bit_char = '1' if val else '0'
                bits.append(bit_char)

        bit_str = "".join(bits)
        self.send_command(f"FRAME:{bit_str}")


# ==============================================================================
# 2. ASPECT-RATIO PRESERVING FRAME RESIZER (ELIMINATES SQUEEZING)
# ==============================================================================
def fit_frame_cover(frame, target_w, target_h):
    """
    Scales and center-crops the camera frame to fit (target_w, target_h)
    preserving 100% true 1:1 aspect ratio with ZERO distortion or squeezing.
    Returns: cropped_frame, scale, start_x, start_y
    """
    h, w = frame.shape[:2]
    scale = max(target_w / float(w), target_h / float(h))
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    cropped = resized[start_y:start_y + target_h, start_x:start_x + target_w]

    return cropped, scale, start_x, start_y


# ==============================================================================
# 3. ANTI-JITTER & POINTER SMOOTHER
# ==============================================================================
class AntiJitterFilter:
    """Provides EMA continuous filtering for analog brightness."""
    def __init__(self, alpha=0.30):
        self.alpha = alpha
        self.filtered_val = 75.0

    def update(self, new_val):
        if self.filtered_val is None:
            self.filtered_val = float(new_val)
        else:
            self.filtered_val = self.alpha * new_val + (1.0 - self.alpha) * self.filtered_val
        return self.filtered_val

class PointerSmoother:
    """Filters fingertip coordinates to eliminate tremor and jitter."""
    def __init__(self, alpha=0.50):
        self.alpha = alpha
        self.smooth_x = None
        self.smooth_y = None

    def update(self, x, y):
        if self.smooth_x is None:
            self.smooth_x = float(x)
            self.smooth_y = float(y)
        else:
            self.smooth_x = self.alpha * x + (1.0 - self.alpha) * self.smooth_x
            self.smooth_y = self.alpha * y + (1.0 - self.alpha) * self.smooth_y
        return int(self.smooth_x), int(self.smooth_y)

    def reset(self):
        self.smooth_x = None
        self.smooth_y = None


# ==============================================================================
# 4. HIGH-ACCURACY HAND GESTURE & LANDMARK ANALYZER
# ==============================================================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

class HandGestureAnalyzer:
    """High-accuracy hand landmark detector with pixel-space geometry."""
    def __init__(self):
        self.available = False
        if not MEDIAPIPE_AVAILABLE:
            return

        ensure_model_asset()
        if not os.path.exists(MODEL_PATH):
            return

        try:
            base_options = python.BaseOptions(
                model_asset_path=MODEL_PATH,
                delegate=python.BaseOptions.Delegate.CPU
            )
            # High sensitivity thresholds for reliable detection
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=vision.RunningMode.IMAGE
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.available = True
        except Exception as e:
            print(f"[WARNING] MediaPipe init error: {e}", flush=True)

    def dist(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def analyze(self, frame_bgr):
        if not self.available:
            return False, "NONE", None, False

        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)

        hand_detected = False
        gesture_name = "NONE"
        is_pinching = False
        tip_raw_px = None

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand_detected = True
            lm_list = result.hand_landmarks[0]
            # True isotropic pixel coordinates
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]

            wrist = pts[0]
            thumb_tip = pts[4]
            index_tip = pts[8]
            middle_tip = pts[12]
            ring_tip = pts[16]
            pinky_tip = pts[20]

            tip_raw_px = index_tip

            # Palm size scale
            hand_scale = max(self.dist(wrist, pts[9]), 25.0)

            # Pinch detection in true pixel space
            pinch_dist_px = self.dist(thumb_tip, index_tip)
            pinch_dist_norm = pinch_dist_px / hand_scale
            is_pinching = (pinch_dist_norm < 0.28)

            # Finger extensions
            index_ext = self.dist(index_tip, wrist) > self.dist(pts[6], wrist) * 1.15
            middle_ext = self.dist(middle_tip, wrist) > self.dist(pts[10], wrist) * 1.15
            ring_ext = self.dist(ring_tip, wrist) > self.dist(pts[14], wrist) * 1.15
            pinky_ext = self.dist(pinky_tip, wrist) > self.dist(pts[18], wrist) * 1.15

            if is_pinching:
                gesture_name = "PINCH"
            elif index_ext and not middle_ext and not ring_ext and not pinky_ext:
                gesture_name = "POINT"
            elif index_ext and middle_ext and not ring_ext and not pinky_ext:
                gesture_name = "PEACE"
            elif index_ext and middle_ext and ring_ext and pinky_ext:
                gesture_name = "PALM"
            elif not index_ext and not middle_ext and not ring_ext and not pinky_ext:
                gesture_name = "FIST"
            else:
                gesture_name = "TRACK"

            # Clean skeleton overlay on camera view
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame_bgr, pts[start_idx], pts[end_idx], (0, 210, 255), 2, cv2.LINE_AA)
            for i, pt in enumerate(pts):
                pt_col = (0, 255, 180) if i in (4, 8) else (255, 140, 50)
                cv2.circle(frame_bgr, pt, 3, pt_col, -1)

            if is_pinching:
                pinch_mid = ((thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2)
                cv2.circle(frame_bgr, pinch_mid, 10, (0, 255, 255), -1)

        return hand_detected, gesture_name, tip_raw_px, is_pinching


# ==============================================================================
# 5. UI GEOMETRY & CONSTANTS
# ==============================================================================
WINDOW_W = 1260
WINDOW_H = 720

# LEFT SIDE: Camera Viewport (Widescreen 600x475)
CAM_X = 35
CAM_Y = 80
CAM_W = 600
CAM_H = 475

# Fixed 8x8 Control Box (Air-Pad) on Camera
control_area = {
    'x': CAM_X + 25,
    'y': CAM_Y + 45,
    'w': 380,
    'h': 380,
    'locked': True
}

# Dedicated Hand Brightness Control Zone (on Camera, to the right of 8x8 box)
BRIGHT_ZONE_W = 46
BRIGHT_ZONE_GAP = 22

# Toolbar below Camera View
AREA_BTN_Y = CAM_Y + CAM_H + 18
AREA_BTN_H = 32
AREA_BUTTONS = {
    'LOCK_TOGGLE': (CAM_X,       AREA_BTN_Y, 130, AREA_BTN_H),
    'CENTER':      (CAM_X + 145, AREA_BTN_Y, 90,  AREA_BTN_H),
    'CYCLE_SIZE':  (CAM_X + 250, AREA_BTN_Y, 110, AREA_BTN_H),
    'RESET':       (CAM_X + 375, AREA_BTN_Y, 90,  AREA_BTN_H)
}

# RIGHT SIDE: 8x8 LED Matrix Display
MATRIX_ORIGIN_X = 740
MATRIX_ORIGIN_Y = 135
DOT_SPACING = 54
DOT_RADIUS = 18

# 8x8 Dot Matrix State (1 = ON, 0 = OFF)
grid_dots = np.zeros((8, 8), dtype=np.uint8)

# Minimal Brightness Slider (on right side)
SLIDER_X = 740
SLIDER_Y = 575
SLIDER_W = 378
SLIDER_H = 6
current_brightness = 75
is_dragging_brightness = False
is_hand_adjusting_brightness = False

# 5x7 Typography Font (ASCII 32 ' ' to 90 'Z')
FONT_5X7 = {
    ' ': [0x00, 0x00, 0x00, 0x00, 0x00],
    '!': [0x00, 0x00, 0x5F, 0x00, 0x00],
    '"': [0x00, 0x07, 0x00, 0x07, 0x00],
    '#': [0x14, 0x7F, 0x14, 0x7F, 0x14],
    '$': [0x24, 0x2A, 0x7F, 0x2A, 0x12],
    '%': [0x23, 0x13, 0x08, 0x64, 0x62],
    '&': [0x36, 0x49, 0x55, 0x22, 0x50],
    "'": [0x00, 0x05, 0x03, 0x00, 0x00],
    '(': [0x00, 0x1C, 0x22, 0x41, 0x00],
    ')': [0x00, 0x41, 0x22, 0x1C, 0x00],
    '*': [0x14, 0x08, 0x3E, 0x08, 0x14],
    '+': [0x08, 0x08, 0x3E, 0x08, 0x08],
    ',': [0x00, 0x50, 0x30, 0x00, 0x00],
    '-': [0x08, 0x08, 0x08, 0x08, 0x08],
    '.': [0x00, 0x60, 0x60, 0x00, 0x00],
    '/': [0x20, 0x10, 0x08, 0x04, 0x02],
    '0': [0x3E, 0x51, 0x49, 0x45, 0x3E],
    '1': [0x00, 0x42, 0x7F, 0x40, 0x00],
    '2': [0x42, 0x61, 0x51, 0x49, 0x46],
    '3': [0x21, 0x41, 0x45, 0x4B, 0x31],
    '4': [0x18, 0x14, 0x12, 0x7F, 0x10],
    '5': [0x27, 0x45, 0x45, 0x45, 0x39],
    '6': [0x3C, 0x4A, 0x49, 0x49, 0x30],
    '7': [0x01, 0x71, 0x09, 0x05, 0x03],
    '8': [0x36, 0x49, 0x49, 0x49, 0x36],
    '9': [0x06, 0x49, 0x49, 0x29, 0x1E],
    ':': [0x00, 0x36, 0x36, 0x00, 0x00],
    ';': [0x00, 0x56, 0x36, 0x00, 0x00],
    '<': [0x08, 0x14, 0x22, 0x41, 0x00],
    '=': [0x14, 0x14, 0x14, 0x14, 0x14],
    '>': [0x00, 0x41, 0x22, 0x14, 0x08],
    '?': [0x02, 0x01, 0x51, 0x09, 0x06],
    '@': [0x32, 0x49, 0x79, 0x41, 0x3E],
    'A': [0x7E, 0x11, 0x11, 0x11, 0x7E],
    'B': [0x7F, 0x49, 0x49, 0x49, 0x36],
    'C': [0x3E, 0x41, 0x41, 0x41, 0x22],
    'D': [0x7F, 0x41, 0x41, 0x22, 0x1C],
    'E': [0x7F, 0x49, 0x49, 0x49, 0x41],
    'F': [0x7F, 0x09, 0x09, 0x09, 0x01],
    'G': [0x3E, 0x41, 0x49, 0x49, 0x7A],
    'H': [0x7F, 0x08, 0x08, 0x08, 0x7F],
    'I': [0x00, 0x41, 0x7F, 0x41, 0x00],
    'J': [0x20, 0x40, 0x41, 0x3F, 0x01],
    'K': [0x7F, 0x08, 0x14, 0x22, 0x41],
    'L': [0x7F, 0x40, 0x40, 0x40, 0x40],
    'M': [0x7F, 0x02, 0x0C, 0x02, 0x7F],
    'N': [0x7F, 0x04, 0x08, 0x10, 0x7F],
    'O': [0x3E, 0x41, 0x41, 0x41, 0x3E],
    'P': [0x7F, 0x09, 0x09, 0x09, 0x06],
    'Q': [0x3E, 0x41, 0x51, 0x21, 0x5E],
    'R': [0x7F, 0x09, 0x19, 0x29, 0x46],
    'S': [0x46, 0x49, 0x49, 0x49, 0x31],
    'T': [0x01, 0x01, 0x7F, 0x01, 0x01],
    'U': [0x3F, 0x40, 0x40, 0x40, 0x3F],
    'V': [0x1F, 0x20, 0x40, 0x20, 0x1F],
    'W': [0x3F, 0x40, 0x38, 0x40, 0x3F],
    'X': [0x63, 0x14, 0x08, 0x14, 0x63],
    'Y': [0x07, 0x08, 0x70, 0x08, 0x07],
    'Z': [0x61, 0x51, 0x49, 0x45, 0x43]
}

# Heartbeat Animation Bitmaps (Human physiological cycle)
HEART_LARGE = np.array([
    [0,0,0,0,0,0,0,0],
    [0,1,1,0,0,1,1,0],
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1],
    [0,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0]
], dtype=np.uint8)

HEART_SMALL = np.array([
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,1,0,0,1,0,0],
    [0,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0]
], dtype=np.uint8)

def build_scrolling_columns(msg_str):
    """Generates column-wise bit data for right-to-left scrolling text."""
    cols = []
    cols.extend([0x00] * 8)  # 8 blank lead-in columns
    for ch in msg_str.upper():
        char_cols = FONT_5X7.get(ch, [0x00, 0x00, 0x00, 0x00, 0x00])
        cols.extend(char_cols)
        cols.append(0x00)     # 1-column letter spacing
    cols.extend([0x00] * 8)  # 8 blank trailing columns
    return cols

def get_heartbeat_frame(now, start_time):
    """
    Simulates human resting heartbeat (~70 BPM, 0.86s cycle).
    Follows physiological 'Lub-Dub' double contraction rhythm:
      - 0.00 to 0.14s: Systole Peak 1 (Large Heart)
      - 0.14 to 0.22s: Brief Dip (Small Heart)
      - 0.22 to 0.36s: Systole Peak 2 (Large Heart)
      - 0.36 to 0.86s: Diastole Rest (Small Heart)
    """
    cycle = 0.86
    t = (now - start_time) % cycle
    is_large = (0.0 <= t < 0.14) or (0.22 <= t < 0.36)
    return (HEART_LARGE, True) if is_large else (HEART_SMALL, False)

# Typography Bar Geometry (Above 8x8 Matrix)
TYPO_Y = 74
TYPO_H = 34
TYPO_INPUT_RECT = (SLIDER_X, TYPO_Y, 266, TYPO_H)
TYPO_SCROLL_RECT = (SLIDER_X + 276, TYPO_Y, 102, TYPO_H)

# Animation State Variables
custom_message = "HELLO"
is_typing_mode = False
is_scroll_active = False
scroll_step = 0
last_scroll_time = 0.0
SCROLL_SPEED = 0.085  # 85ms per column
scroll_cols = []

is_heartbeat_active = False
heartbeat_start_time = 0.0
last_heart_state = None

# Minimal Action Buttons (5 buttons below slider)
BTN_W = 68
BTN_H = 34
BTN_GAP = 9
BTN_Y = 615
BUTTONS = {
    'CLEAR': (SLIDER_X,                           BTN_Y, BTN_W, BTN_H, "Clear"),
    'HEART': (SLIDER_X + (BTN_W + BTN_GAP)*1,     BTN_Y, BTN_W, BTN_H, "Heart"),
    'HELLO': (SLIDER_X + (BTN_W + BTN_GAP)*2,     BTN_Y, BTN_W, BTN_H, "Hello"),
    'SMILE': (SLIDER_X + (BTN_W + BTN_GAP)*3,     BTN_Y, BTN_W, BTN_H, "Smile"),
    'LOCK':  (SLIDER_X + (BTN_W + BTN_GAP)*4,     BTN_Y, BTN_W, BTN_H, "Lock")
}
is_master_locked = False

# Touch Point Dwell Delay & Anti-Bounce Settings
DWELL_TRIGGER_TIME = 0.75  # Deliberate 750ms dwell hold
dwell_dot = None
dwell_start_time = 0
dwell_lockout_dot = None   # Single-fire: won't re-toggle until finger leaves dot
last_air_click_time = 0
last_pinch_state = False
last_gesture_cmd_time = 0
click_ripple_anim = None   # (cx, cy, start_time)

# Mouse Interaction State
mouse_pos = (-1, -1)
is_dragging_area = False
is_resizing_area = False
drag_offset = (0, 0)


# ==============================================================================
# 6. COORDINATE MAPPING & HIT TESTING
# ==============================================================================
def is_inside_rect(px, py, rect):
    rx, ry, rw, rh = rect
    return (rx <= px <= rx + rw) and (ry <= py <= ry + rh)

def get_dot_at_xy(px, py):
    """Returns (r, c) if coordinate (px, py) falls inside any matrix dot."""
    for r in range(8):
        for c in range(8):
            cx = MATRIX_ORIGIN_X + c * DOT_SPACING
            cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
            if math.hypot(px - cx, py - cy) <= DOT_RADIUS + 8:
                return (r, c)
    return None

def get_dot_from_control_box(px, py):
    """
    Maps coordinate (px, py) inside the Fixed Control Box directly to (r, c).
    Returns None if (px, py) is outside the box.
    """
    bx = control_area['x']
    by = control_area['y']
    bw = control_area['w']
    bh = control_area['h']

    if bx <= px <= bx + bw and by <= py <= by + bh:
        c = int(((px - bx) / float(bw)) * 8)
        r = int(((py - by) / float(bh)) * 8)
        return (max(0, min(7, r)), max(0, min(7, c)))
    return None

def get_bright_zone_rect():
    """Returns (x, y, w, h) of the dedicated hand brightness zone on camera."""
    bx = control_area['x'] + control_area['w'] + BRIGHT_ZONE_GAP
    by = control_area['y']
    bw = BRIGHT_ZONE_W
    bh = control_area['h']
    return (bx, by, bw, bh)


# ==============================================================================
# 7. MOUSE EVENT HANDLER
# ==============================================================================
def on_mouse_event(event, x, y, flags, param):
    """Handles mouse click & drag for dots, slider, buttons, and area control."""
    global mouse_pos, current_brightness, is_dragging_brightness
    global is_dragging_area, is_resizing_area, drag_offset, click_ripple_anim
    global is_master_locked, is_typing_mode, is_scroll_active, scroll_step, last_scroll_time
    global is_heartbeat_active, heartbeat_start_time, last_heart_state, custom_message, scroll_cols

    comm = param
    mouse_pos = (x, y)

    ax = control_area['x']
    ay = control_area['y']
    aw = control_area['w']
    ah = control_area['h']
    resize_handle_rect = (ax + aw - 20, ay + ah - 20, 20, 20)

    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. Area Toolbar Buttons (below camera)
        if is_inside_rect(x, y, AREA_BUTTONS['LOCK_TOGGLE']):
            control_area['locked'] = not control_area['locked']
            print(f"[AREA] {'LOCKED' if control_area['locked'] else 'EDIT MODE'}", flush=True)
            return

        elif is_inside_rect(x, y, AREA_BUTTONS['CENTER']):
            control_area['x'] = CAM_X + 25
            control_area['y'] = CAM_Y + (CAM_H - control_area['h']) // 2
            print("[AREA] Centered on camera view", flush=True)
            return

        elif is_inside_rect(x, y, AREA_BUTTONS['CYCLE_SIZE']):
            sizes = [320, 380, 420]
            curr = control_area['w']
            next_size = sizes[(sizes.index(curr) + 1) % len(sizes)] if curr in sizes else 380
            control_area['w'] = next_size
            control_area['h'] = next_size
            # Clamp inside camera viewport leaving room for brightness bar
            max_x = CAM_X + CAM_W - next_size - BRIGHT_ZONE_W - BRIGHT_ZONE_GAP - 5
            control_area['x'] = max(CAM_X + 2, min(max_x, control_area['x']))
            control_area['y'] = max(CAM_Y + 2, min(CAM_Y + CAM_H - next_size - 2, control_area['y']))
            print(f"[AREA] Size: {next_size}x{next_size}", flush=True)
            return

        elif is_inside_rect(x, y, AREA_BUTTONS['RESET']):
            control_area['w'] = 380
            control_area['h'] = 380
            control_area['x'] = CAM_X + 25
            control_area['y'] = CAM_Y + 45
            control_area['locked'] = True
            print("[AREA] Reset to default and locked", flush=True)
            return

        # 2. In EDIT mode: check drag or resize on Control Box
        if not control_area['locked']:
            if is_inside_rect(x, y, resize_handle_rect):
                is_resizing_area = True
                return
            elif is_inside_rect(x, y, (ax, ay, aw, ah)):
                is_dragging_area = True
                drag_offset = (x - ax, y - ay)
                return

        # Typography Input & Scroll Buttons (above matrix)
        if is_inside_rect(x, y, TYPO_INPUT_RECT):
            if is_master_locked:
                print("[LOCK] System is LOCKED. Click [LOCKED] or press SPACE to unlock.", flush=True)
                return
            is_typing_mode = not is_typing_mode
            status = "ACTIVE - Type and press ENTER to scroll" if is_typing_mode else "CLOSED"
            print(f"[TYPOGRAPHY] Edit Mode: {status} (Current: '{custom_message}')", flush=True)
            return

        elif is_inside_rect(x, y, TYPO_SCROLL_RECT):
            if is_master_locked:
                return
            is_scroll_active = not is_scroll_active
            if is_scroll_active:
                is_heartbeat_active = False
                scroll_cols = build_scrolling_columns(custom_message)
                scroll_step = 0
                last_scroll_time = time.time()
                print(f"[TYPOGRAPHY] Scrolling started: '{custom_message}'", flush=True)
            else:
                grid_dots.fill(0)
                comm.send_frame(grid_dots)
                print("[TYPOGRAPHY] Scrolling stopped", flush=True)
            return

        # 3. Direct click on matrix dot
        dot = get_dot_at_xy(x, y)
        if dot is not None:
            if is_master_locked:
                print("[LOCK] System is LOCKED. Click [LOCKED] or press SPACE to unlock.", flush=True)
                return
            is_heartbeat_active = False
            is_scroll_active = False
            r, c = dot
            grid_dots[r, c] ^= 1
            comm.send_toggle_dot(r, c)
            cx = MATRIX_ORIGIN_X + c * DOT_SPACING
            cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
            click_ripple_anim = (cx, cy, time.time())
            print(f"[MOUSE] Dot ({r}, {c}) -> {'ON' if grid_dots[r,c] else 'OFF'}", flush=True)
            return

        # 4. Click on Brightness Slider
        slider_expand = (SLIDER_X - 10, SLIDER_Y - 12, SLIDER_W + 20, SLIDER_H + 24)
        if is_inside_rect(x, y, slider_expand):
            if is_master_locked:
                return
            is_dragging_brightness = True
            ratio = (x - SLIDER_X) / float(SLIDER_W)
            current_brightness = int(max(0, min(100, ratio * 100)))
            comm.send_brightness(current_brightness)
            print(f"[MOUSE] Brightness: {current_brightness}%", flush=True)
            return

        # 5. Click on Action Buttons
        for key, (bx, by, bw, bh, label) in BUTTONS.items():
            if is_inside_rect(x, y, (bx, by, bw, bh)):
                if key == 'LOCK':
                    is_master_locked = not is_master_locked
                    status_str = "LOCKED (No changes allowed)" if is_master_locked else "UNLOCKED"
                    print(f"[LOCK] Master Lock: {status_str}", flush=True)
                    return
                if is_master_locked:
                    print("[LOCK] System is LOCKED. Click [LOCKED] or press SPACE to unlock.", flush=True)
                    return
                if key == 'CLEAR':
                    is_heartbeat_active = False
                    is_scroll_active = False
                    grid_dots.fill(0)
                    comm.send_frame(grid_dots)
                    print("[MATRIX] Cleared", flush=True)
                elif key == 'HEART':
                    is_heartbeat_active = not is_heartbeat_active
                    is_scroll_active = False
                    if is_heartbeat_active:
                        heartbeat_start_time = time.time()
                        last_heart_state = None
                        print("[ANIMATION] Human Heartbeat mode ACTIVE (~70 BPM lub-dub)", flush=True)
                    else:
                        grid_dots.fill(0)
                        comm.send_frame(grid_dots)
                        print("[ANIMATION] Heartbeat stopped", flush=True)
                elif key == 'HELLO':
                    custom_message = "HELLO"
                    scroll_cols = build_scrolling_columns("HELLO")
                    scroll_step = 0
                    is_scroll_active = True
                    is_heartbeat_active = False
                    last_scroll_time = time.time()
                    print("[TYPOGRAPHY] Scrolling preset 'HELLO' right-to-left", flush=True)
                elif key == 'SMILE':
                    is_heartbeat_active = False
                    is_scroll_active = False
                    smile = [
                        [0,0,1,1,1,1,0,0],
                        [0,1,0,0,0,0,1,0],
                        [1,0,1,0,0,1,0,1],
                        [1,0,0,0,0,0,0,1],
                        [1,0,1,0,0,1,0,1],
                        [1,0,0,1,1,0,0,1],
                        [0,1,0,0,0,0,1,0],
                        [0,0,1,1,1,1,0,0]
                    ]
                    grid_dots[:] = smile
                    comm.send_frame(grid_dots)
                    print("[MATRIX] Smile pattern displayed", flush=True)
                return

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging_brightness and not is_master_locked:
            ratio = (x - SLIDER_X) / float(SLIDER_W)
            current_brightness = int(max(0, min(100, ratio * 100)))
            comm.send_brightness(current_brightness)

        elif is_dragging_area and not control_area['locked']:
            new_x = x - drag_offset[0]
            new_y = y - drag_offset[1]
            max_x = CAM_X + CAM_W - control_area['w'] - BRIGHT_ZONE_W - BRIGHT_ZONE_GAP - 5
            control_area['x'] = max(CAM_X + 2, min(max_x, new_x))
            control_area['y'] = max(CAM_Y + 2, min(CAM_Y + CAM_H - control_area['h'] - 2, new_y))

        elif is_resizing_area and not control_area['locked']:
            max_w = CAM_W - (control_area['x'] - CAM_X) - BRIGHT_ZONE_W - BRIGHT_ZONE_GAP - 5
            new_w = max(200, min(max_w, x - control_area['x']))
            new_h = max(200, min(CAM_H - (control_area['y'] - CAM_Y) - 2, y - control_area['y']))
            side = min(new_w, new_h)
            control_area['w'] = side
            control_area['h'] = side

    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging_brightness = False
        is_dragging_area = False
        is_resizing_area = False


# ==============================================================================
# 8. RENDERING ENGINE
# ==============================================================================
def render_ui(canvas, cam_cropped, tip_canvas, active_dot, dwell_progress, ip, port, gesture, brightness_active, transpose=True, invert=True):
    """Draws the clean widescreen UI: Camera View on Left, 8x8 Matrix on Right."""
    global click_ripple_anim

    # 1. TOP HEADER (Subtle, clean, monochrome)
    cv2.putText(canvas, "CAMERA VIEW  (AIR-PAD & BRIGHTNESS)", (CAM_X, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 148), 1, cv2.LINE_AA)

    cv2.putText(canvas, "8x8 LED MATRIX", (MATRIX_ORIGIN_X, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 225), 1, cv2.LINE_AA)

    if is_master_locked:
        cv2.rectangle(canvas, (MATRIX_ORIGIN_X + 160, 26), (MATRIX_ORIGIN_X + 270, 50), (20, 70, 160), -1)
        cv2.rectangle(canvas, (MATRIX_ORIGIN_X + 160, 26), (MATRIX_ORIGIN_X + 270, 50), (0, 160, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "LOCKED", (MATRIX_ORIGIN_X + 185, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 240, 255), 1, cv2.LINE_AA)

    # Status indicator (small clean green dot + IP)
    cv2.circle(canvas, (WINDOW_W - 190, 40), 4, (0, 220, 100), -1)
    cv2.putText(canvas, f"{ip}:{port}", (WINDOW_W - 176, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 135), 1, cv2.LINE_AA)

    # Subtle divider line
    cv2.line(canvas, (CAM_X, 60), (WINDOW_W - CAM_X, 60), (36, 36, 40), 1)

    # Typography Bar (Above 8x8 Matrix)
    in_hover = is_inside_rect(mouse_pos[0], mouse_pos[1], TYPO_INPUT_RECT)
    if is_typing_mode:
        in_bg = (30, 42, 58)
        in_border = (0, 200, 255)
        cursor = "_" if int(time.time() * 2.5) % 2 == 0 else " "
        in_text = f"Type: {custom_message}{cursor}"
        txt_col = (0, 230, 255)
    else:
        in_bg = (32, 32, 38) if in_hover else (24, 24, 28)
        in_border = (90, 90, 100) if in_hover else (45, 45, 52)
        in_text = f'Msg: "{custom_message}" [M to edit]'
        txt_col = (200, 200, 210)

    cv2.rectangle(canvas, (TYPO_INPUT_RECT[0], TYPO_INPUT_RECT[1]),
                  (TYPO_INPUT_RECT[0] + TYPO_INPUT_RECT[2], TYPO_INPUT_RECT[1] + TYPO_INPUT_RECT[3]), in_bg, -1)
    cv2.rectangle(canvas, (TYPO_INPUT_RECT[0], TYPO_INPUT_RECT[1]),
                  (TYPO_INPUT_RECT[0] + TYPO_INPUT_RECT[2], TYPO_INPUT_RECT[1] + TYPO_INPUT_RECT[3]), in_border, 1, cv2.LINE_AA)
    cv2.putText(canvas, in_text, (TYPO_INPUT_RECT[0] + 10, TYPO_INPUT_RECT[1] + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_col, 1, cv2.LINE_AA)

    # Scroll Toggle Button
    sc_hover = is_inside_rect(mouse_pos[0], mouse_pos[1], TYPO_SCROLL_RECT)
    if is_scroll_active:
        sc_bg = (18, 55, 65) if sc_hover else (12, 45, 55)
        sc_border = (0, 230, 200)
        sc_text = "SCROLL: ON"
        sc_col = (0, 255, 220)
    else:
        sc_bg = (34, 34, 40) if sc_hover else (24, 24, 28)
        sc_border = (90, 90, 100) if sc_hover else (45, 45, 52)
        sc_text = "Scroll: OFF"
        sc_col = (180, 180, 185)

    cv2.rectangle(canvas, (TYPO_SCROLL_RECT[0], TYPO_SCROLL_RECT[1]),
                  (TYPO_SCROLL_RECT[0] + TYPO_SCROLL_RECT[2], TYPO_SCROLL_RECT[1] + TYPO_SCROLL_RECT[3]), sc_bg, -1)
    cv2.rectangle(canvas, (TYPO_SCROLL_RECT[0], TYPO_SCROLL_RECT[1]),
                  (TYPO_SCROLL_RECT[0] + TYPO_SCROLL_RECT[2], TYPO_SCROLL_RECT[1] + TYPO_SCROLL_RECT[3]), sc_border, 1, cv2.LINE_AA)
    cv2.putText(canvas, sc_text, (TYPO_SCROLL_RECT[0] + 12, TYPO_SCROLL_RECT[1] + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, sc_col, 1, cv2.LINE_AA)

    # 2. LEFT PANEL: Camera Feed Viewport (100% natural, un-squeezed)
    canvas[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = cam_cropped
    cv2.rectangle(canvas, (CAM_X, CAM_Y), (CAM_X + CAM_W, CAM_Y + CAM_H), (45, 45, 52), 1)

    # --- Fixed 8x8 Control Box (Air-Pad) ---
    bx = control_area['x']
    by = control_area['y']
    bw = control_area['w']
    bh = control_area['h']
    is_locked = control_area['locked']

    box_color = (0, 230, 255) if is_locked else (0, 160, 255)

    # Subtle translucent dark fill inside the box
    overlay = canvas.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

    # Box outline
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), box_color, 2)

    # 8x8 subgrid guide lines
    cw = bw / 8.0
    ch = bh / 8.0
    for i in range(1, 8):
        gx = int(bx + i * cw)
        gy = int(by + i * ch)
        cv2.line(canvas, (gx, by), (gx, by + bh), (65, 75, 85), 1)
        cv2.line(canvas, (bx, gy), (bx + bw, gy), (65, 75, 85), 1)

    # Coordinate markers on the box
    for c in range(8):
        lbl_x = int(bx + c * cw + cw / 2 - 4)
        cv2.putText(canvas, str(c), (lbl_x, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 170), 1)
    for r in range(8):
        lbl_y = int(by + r * ch + ch / 2 + 4)
        cv2.putText(canvas, str(r), (bx - 14, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 170), 1)

    # Highlight active cell inside the control box
    if active_dot is not None:
        r, c = active_dot
        cx1 = int(bx + c * cw)
        cy1 = int(by + r * ch)
        cx2 = int(cx1 + cw)
        cy2 = int(cy1 + ch)
        cell_overlay = canvas.copy()
        cv2.rectangle(cell_overlay, (cx1, cy1), (cx2, cy2), (0, 230, 255), -1)
        cv2.addWeighted(cell_overlay, 0.30, canvas, 0.70, 0, canvas)
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (0, 240, 255), 2)

    # Fingertip crosshair & dwell ring inside control box
    if tip_canvas is not None and is_inside_rect(tip_canvas[0], tip_canvas[1], (bx, by, bw, bh)):
        tx, ty = tip_canvas
        cv2.circle(canvas, (tx, ty), 8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), 3, (0, 200, 255), -1)
        if dwell_progress > 0.0:
            end_angle = int(dwell_progress * 360)
            cv2.ellipse(canvas, (tx, ty), (14, 14), -90, 0, end_angle, (0, 255, 120), 2, cv2.LINE_AA)

    # Control Box Title Tag
    tag_txt = f"AIR-PAD [{ 'LOCKED' if is_locked else 'EDIT MODE' }]"
    cv2.rectangle(canvas, (bx, by - 22), (bx + 165, by), box_color, -1)
    cv2.putText(canvas, tag_txt, (bx + 8, by - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (15, 15, 20), 1, cv2.LINE_AA)

    # Resize handle (when in EDIT mode)
    if not is_locked:
        handle_x = bx + bw - 18
        handle_y = by + bh - 18
        cv2.rectangle(canvas, (handle_x, handle_y), (bx + bw, by + bh), (0, 160, 255), -1)
        cv2.putText(canvas, "+", (handle_x + 4, handle_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

    # --- DEDICATED HAND BRIGHTNESS CONTROL ZONE (ON CAMERA) ---
    br_x, br_y, br_w, br_h = get_bright_zone_rect()

    # Brightness zone background
    br_bg_overlay = canvas.copy()
    cv2.rectangle(br_bg_overlay, (br_x, br_y), (br_x + br_w, br_y + br_h), (18, 22, 32), -1)
    cv2.addWeighted(br_bg_overlay, 0.35, canvas, 0.65, 0, canvas)

    br_border_col = (0, 255, 255) if brightness_active else (70, 80, 95)
    cv2.rectangle(canvas, (br_x, br_y), (br_x + br_w, br_y + br_h), br_border_col, 2 if brightness_active else 1)

    # Inner vertical slider track
    track_pad_x = 14
    track_x = br_x + track_pad_x
    track_w = br_w - 2 * track_pad_x
    track_y = br_y + 35
    track_h = br_h - 70

    cv2.rectangle(canvas, (track_x, track_y), (track_x + track_w, track_y + track_h), (35, 38, 48), -1)
    cv2.rectangle(canvas, (track_x, track_y), (track_x + track_w, track_y + track_h), (75, 85, 100), 1)

    # Vertical fill (from bottom up)
    fill_h = int(track_h * (current_brightness / 100.0))
    fill_col = (0, 230, 255) if brightness_active else (0, 150, 220)
    cv2.rectangle(canvas, (track_x + 1, track_y + track_h - fill_h),
                  (track_x + track_w - 1, track_y + track_h), fill_col, -1)

    # Knob
    knob_y = track_y + track_h - fill_h
    cv2.circle(canvas, (track_x + track_w // 2, knob_y), 7, (255, 255, 255), -1)
    cv2.circle(canvas, (track_x + track_w // 2, knob_y), 7, (0, 180, 255), 2)

    # Labels on Brightness Zone
    cv2.putText(canvas, "BRIGHT", (br_x + 4, br_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{current_brightness}%", (br_x + 6, br_y + br_h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    # Active tag if hand is in zone
    if brightness_active:
        cv2.putText(canvas, "ADJUSTING", (br_x - 10, br_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 120), 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "MOVE HAND", (br_x - 12, br_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 150), 1, cv2.LINE_AA)

    # Area Control Toolbar (below camera)
    lock_txt = "Locked" if is_locked else "Edit Mode"
    for key, (abx, aby, abw, abh) in AREA_BUTTONS.items():
        is_hov = is_inside_rect(mouse_pos[0], mouse_pos[1], (abx, aby, abw, abh))
        bg = (40, 40, 46) if is_hov else (28, 28, 32)
        bdr = (120, 120, 130) if is_hov else (55, 55, 62)

        lbl = ""
        if key == 'LOCK_TOGGLE':
            lbl = f"[ {lock_txt} ]"
            if not is_locked:
                bdr = (0, 180, 255)
        elif key == 'CENTER':
            lbl = "Center"
        elif key == 'CYCLE_SIZE':
            lbl = f"Size: {bw}px"
        elif key == 'RESET':
            lbl = "Reset"

        cv2.rectangle(canvas, (abx, aby), (abx + abw, aby + abh), bg, -1)
        cv2.rectangle(canvas, (abx, aby), (abx + abw, aby + abh), bdr, 1, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (abx + (abw - len(lbl)*8)//2, aby + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (215, 215, 220), 1, cv2.LINE_AA)

    # 3. RIGHT PANEL: 8x8 LED Matrix Display
    # Column Guides (0..7)
    for c in range(8):
        cx = MATRIX_ORIGIN_X + c * DOT_SPACING
        cv2.putText(canvas, str(c), (cx - 4, MATRIX_ORIGIN_Y - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 100, 110), 1, cv2.LINE_AA)
    # Row Guides (0..7)
    for r in range(8):
        cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
        cv2.putText(canvas, str(r), (MATRIX_ORIGIN_X - 32, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 100, 110), 1, cv2.LINE_AA)

    # Render All 64 Circular LEDs
    for r in range(8):
        for c in range(8):
            cx = MATRIX_ORIGIN_X + c * DOT_SPACING
            cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
            is_lit = (grid_dots[r, c] == 1)
            is_hover = (active_dot == (r, c))

            if is_lit:
                # Realistic Glowing Red LED (matches screenshot)
                cv2.circle(canvas, (cx, cy), DOT_RADIUS + 7, (0, 0, 140), -1)       # Ambient halo
                cv2.circle(canvas, (cx, cy), DOT_RADIUS + 3, (20, 35, 220), -1)     # Mid glow
                cv2.circle(canvas, (cx, cy), DOT_RADIUS, (40, 70, 255), -1)         # Core
                cv2.circle(canvas, (cx, cy), DOT_RADIUS - 6, (120, 160, 255), -1)   # Hot center
                cv2.circle(canvas, (cx - 4, cy - 4), 3, (240, 240, 255), -1)       # Specular glint
            else:
                # Dark matte lens with clean subtle ring
                cv2.circle(canvas, (cx, cy), DOT_RADIUS, (26, 25, 29), -1)
                cv2.circle(canvas, (cx, cy), DOT_RADIUS, (45, 43, 50), 1, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 3, (38, 36, 44), -1)

            # Hover Ring (on the virtual matrix)
            if is_hover:
                cv2.circle(canvas, (cx, cy), DOT_RADIUS + 6, (220, 220, 230), 1, cv2.LINE_AA)
                if dwell_progress > 0.0:
                    end_angle = int(dwell_progress * 360)
                    cv2.ellipse(canvas, (cx, cy), (DOT_RADIUS + 9, DOT_RADIUS + 9),
                                -90, 0, end_angle, (0, 230, 255), 2, cv2.LINE_AA)

    # Click Ripple Animation
    if click_ripple_anim is not None:
        rx, ry, r_time = click_ripple_anim
        elapsed = time.time() - r_time
        if elapsed <= 0.30:
            radius = int(DOT_RADIUS + elapsed * 65)
            alpha = max(0, int(255 * (1.0 - elapsed / 0.30)))
            cv2.circle(canvas, (rx, ry), radius, (0, alpha, 255), 2, cv2.LINE_AA)
        else:
            click_ripple_anim = None

    # 4. Minimal Brightness Slider
    cv2.putText(canvas, "Brightness", (SLIDER_X, SLIDER_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 155), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{current_brightness}%", (SLIDER_X + SLIDER_W - 28, SLIDER_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 195), 1, cv2.LINE_AA)

    # Track groove
    cv2.rectangle(canvas, (SLIDER_X, SLIDER_Y), (SLIDER_X + SLIDER_W, SLIDER_Y + SLIDER_H), (36, 36, 42), -1)
    fill_w = int(SLIDER_W * (current_brightness / 100.0))
    cv2.rectangle(canvas, (SLIDER_X, SLIDER_Y), (SLIDER_X + fill_w, SLIDER_Y + SLIDER_H), (200, 210, 225), -1)
    # Knob
    cv2.circle(canvas, (SLIDER_X + fill_w, SLIDER_Y + SLIDER_H // 2), 6, (255, 255, 255), -1)

    # 5. Minimal Action Buttons (Clear, Heart, Hello, Smile, Lock)
    for key, (bx, by, bw, bh, label) in BUTTONS.items():
        is_hover = is_inside_rect(mouse_pos[0], mouse_pos[1], (bx, by, bw, bh))
        if key == 'LOCK':
            display_label = "LOCKED" if is_master_locked else "Lock"
            if is_master_locked:
                bg_col = (20, 70, 160) if is_hover else (15, 50, 120)
                border_col = (0, 160, 255)
                text_col = (240, 240, 255)
            else:
                bg_col = (38, 38, 44) if is_hover else (28, 28, 32)
                border_col = (110, 110, 120) if is_hover else (55, 55, 62)
                text_col = (220, 220, 225)
        elif key == 'HEART':
            display_label = "BEAT" if is_heartbeat_active else "Heart"
            if is_heartbeat_active:
                bg_col = (25, 25, 115) if is_hover else (20, 20, 95)
                border_col = (60, 90, 255)
                text_col = (180, 210, 255)
            else:
                bg_col = (38, 38, 44) if is_hover else (28, 28, 32)
                border_col = (110, 110, 120) if is_hover else (55, 55, 62)
                text_col = (220, 220, 225)
        elif key == 'HELLO':
            display_label = "Hello"
            if is_scroll_active and custom_message == "HELLO":
                bg_col = (18, 55, 65) if is_hover else (12, 45, 55)
                border_col = (0, 230, 200)
                text_col = (0, 255, 220)
            else:
                bg_col = (38, 38, 44) if is_hover else (28, 28, 32)
                border_col = (110, 110, 120) if is_hover else (55, 55, 62)
                text_col = (220, 220, 225)
        else:
            display_label = label
            bg_col = (38, 38, 44) if is_hover else (28, 28, 32)
            border_col = (110, 110, 120) if is_hover else (55, 55, 62)
            text_col = (220, 220, 225)

        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), bg_col, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_col, 1, cv2.LINE_AA)
        cv2.putText(canvas, display_label, (bx + (bw - len(display_label)*7)//2, by + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_col, 1, cv2.LINE_AA)

    # 6. Bottom Footer
    cv2.line(canvas, (CAM_X, 672), (WINDOW_W - CAM_X, 672), (32, 32, 36), 1)
    target_text = f"Target: Dot ({active_dot[0]}, {active_dot[1]})" if active_dot else "Waiting for hand"
    t_status = "ON" if transpose else "OFF"
    i_status = "ON" if invert else "OFF"
    lock_status = "LOCKED [SPACE]" if is_master_locked else "OFF [SPACE]"
    anim_status = "HEARTBEAT" if is_heartbeat_active else ("SCROLL" if is_scroll_active else "MANUAL")
    footer_text = f"Mode: {anim_status}  |  Msg: '{custom_message}'  |  Lock: {lock_status}  |  Hold: {DWELL_TRIGGER_TIME:.2f}s  |  {target_text}"
    cv2.putText(canvas, footer_text, (CAM_X, 696),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (115, 115, 122), 1, cv2.LINE_AA)


# ==============================================================================
# 9. MAIN LOOP
# ==============================================================================
def main():
    global dwell_dot, dwell_start_time, last_air_click_time, last_pinch_state
    global last_gesture_cmd_time, current_brightness, click_ripple_anim
    global dwell_lockout_dot, DWELL_TRIGGER_TIME, is_hand_adjusting_brightness
    global is_master_locked, is_typing_mode, is_scroll_active, scroll_step, last_scroll_time
    global is_heartbeat_active, heartbeat_start_time, last_heart_state, custom_message, scroll_cols

    parser = argparse.ArgumentParser(description="Minimal 8x8 LED Matrix Controller")
    parser.add_argument("--ip", type=str, default="10.150.46.102", help="ESP32 IP address")
    parser.add_argument("--port", type=int, default=8888, help="ESP32 UDP port")
    parser.add_argument("--serial", type=str, default=None, help="Optional Serial Port")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--demo", action="store_true", help="Run in simulation mode")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS")
    parser.add_argument("--dwell", type=float, default=0.75, help="Dwell delay in seconds")
    parser.add_argument("--no-transpose", action="store_true", help="Disable row/col transposition")
    parser.add_argument("--no-invert", action="store_true", help="Disable active-low polarity inversion")
    parser.add_argument("--msg", type=str, default="HELLO", help="Default scrolling message")
    args = parser.parse_args()

    TARGET_FPS = float(args.fps)
    FRAME_INTERVAL = 1.0 / TARGET_FPS
    DWELL_TRIGGER_TIME = float(args.dwell)
    transpose_init = not args.no_transpose
    invert_init = not args.no_invert
    custom_message = args.msg.upper()
    scroll_cols = build_scrolling_columns(custom_message)

    print("\n" + "=" * 60, flush=True)
    print("  8x8 LED MATRIX CONTROLLER (TYPOGRAPHY & HEARTBEAT ANIMATIONS)", flush=True)
    print(f"  Target ESP32:       {args.ip}:{args.port}", flush=True)
    print(f"  Message [M]:        '{custom_message}' (Right-to-Left Scroll)", flush=True)
    print("  Heartbeat [H]:      Human physiological rhythm (~70 BPM lub-dub)", flush=True)
    print(f"  Transpose [T]:      {'ON (row <-> col)' if transpose_init else 'OFF'}", flush=True)
    print(f"  Invert Polarity [I]:{'ON (active-low fixed)' if invert_init else 'OFF'}", flush=True)
    print("=" * 60 + "\n", flush=True)

    comm = MatrixCommunicator(udp_ip=args.ip, udp_port=args.port, serial_port=args.serial,
                              transpose=transpose_init, invert=invert_init)
    # Sync initial blank state to ESP32
    comm.send_frame(grid_dots)

    analyzer = HandGestureAnalyzer()
    jitter_filter = AntiJitterFilter(alpha=0.30)
    smoother = PointerSmoother(alpha=0.50)

    use_simulation = args.demo or not analyzer.available
    cap = None

    if not use_simulation:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[NOTE] Camera {args.camera} unavailable. Running in simulation mode.", flush=True)
            use_simulation = True

    # High-Definition 1280x720 capture for maximum tracking accuracy
    if not use_simulation:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    WINDOW_NAME = "8x8 LED Matrix Controller"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse_event, comm)

    sim_angle = 0.0

    try:
        while True:
            frame_start_time = time.perf_counter()

            # 1. Acquire Camera Frame
            if not use_simulation:
                ret, frame_raw = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                frame_cam = cv2.flip(frame_raw, 1)
                detected, gesture, tip_raw, is_pinching = analyzer.analyze(frame_cam)

                # Aspect-ratio preserving cover crop (eliminates squeezing completely)
                cam_cropped, cam_scale, crop_start_x, crop_start_y = fit_frame_cover(frame_cam, CAM_W, CAM_H)

                # Map fingertip from raw camera frame to canvas with exact aspect ratio
                tip_canvas = None
                if tip_raw is not None:
                    raw_mapped_x = int(CAM_X + tip_raw[0] * cam_scale - crop_start_x)
                    raw_mapped_y = int(CAM_Y + tip_raw[1] * cam_scale - crop_start_y)
                    tip_canvas = smoother.update(raw_mapped_x, raw_mapped_y)
                else:
                    smoother.reset()
            else:
                # Simulation mode
                cam_cropped = np.full((CAM_H, CAM_W, 3), 25, dtype=np.uint8)
                sim_angle += 0.03
                bx = control_area['x']
                by = control_area['y']
                bw = control_area['w']
                bh = control_area['h']
                sim_x = int(bx + bw * 0.5 + (bw * 0.35) * math.sin(sim_angle))
                sim_y = int(by + bh * 0.5 + (bh * 0.30) * math.cos(sim_angle * 0.7))
                tip_canvas = (sim_x, sim_y)
                is_pinching = (int(sim_angle * 2) % 8 == 0)
                detected = True
                gesture = "PINCH" if is_pinching else "POINT"

            # 2. Canvas Base
            canvas = np.full((WINDOW_H, WINDOW_W, 3), 18, dtype=np.uint8)

            # 3. HAND BRIGHTNESS CONTROL
            # Check if hand is in the dedicated Brightness Zone (on camera, to right of 8x8 box)
            brightness_active = False
            bright_rect = get_bright_zone_rect()

            if not is_master_locked and tip_canvas is not None and is_inside_rect(tip_canvas[0], tip_canvas[1], bright_rect):
                brightness_active = True
                br_x, br_y, br_w, br_h = bright_rect
                # Ratio: bottom = 0%, top = 100%
                b_ratio = (br_y + br_h - tip_canvas[1]) / float(br_h)
                raw_b = int(max(0.0, min(1.0, b_ratio)) * 100)
                current_brightness = int(jitter_filter.update(raw_b))
                comm.send_brightness(current_brightness)

            # 4. Map Hand from Fixed Control Box to Matrix Dot
            active_dot = None
            dwell_progress = 0.0

            if tip_canvas is not None and not brightness_active:
                active_dot = get_dot_from_control_box(tip_canvas[0], tip_canvas[1])

            # Mouse hover fallback (on matrix or control box)
            if active_dot is None:
                active_dot = get_dot_at_xy(mouse_pos[0], mouse_pos[1])
                if active_dot is None:
                    active_dot = get_dot_from_control_box(mouse_pos[0], mouse_pos[1])

            now = time.time()

            # PINCH-TO-CLICK (Instant toggle inside Fixed Box)
            if not is_master_locked and is_pinching and not last_pinch_state and (now - last_air_click_time >= 0.45):
                if active_dot is not None:
                    is_heartbeat_active = False
                    is_scroll_active = False
                    r, c = active_dot
                    grid_dots[r, c] ^= 1
                    comm.send_toggle_dot(r, c)
                    last_air_click_time = now
                    dwell_lockout_dot = active_dot
                    cx = MATRIX_ORIGIN_X + c * DOT_SPACING
                    cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
                    click_ripple_anim = (cx, cy, now)
                    print(f"[PINCH] Dot ({r}, {c}) -> {'ON' if grid_dots[r,c] else 'OFF'}", flush=True)

            # DWELL-TO-CLICK (Deliberate hold for DWELL_TRIGGER_TIME with single-fire lockout)
            if not is_master_locked and active_dot is not None and not is_pinching:
                if active_dot == dwell_dot:
                    if active_dot != dwell_lockout_dot:
                        dwell_time = now - dwell_start_time
                        dwell_progress = min(1.0, dwell_time / DWELL_TRIGGER_TIME)
                        if dwell_time >= DWELL_TRIGGER_TIME and (now - last_air_click_time >= 0.50):
                            is_heartbeat_active = False
                            is_scroll_active = False
                            r, c = active_dot
                            grid_dots[r, c] ^= 1
                            comm.send_toggle_dot(r, c)
                            last_air_click_time = now
                            dwell_lockout_dot = active_dot  # Single fire: locked until finger leaves dot
                            dwell_progress = 1.0
                            cx = MATRIX_ORIGIN_X + c * DOT_SPACING
                            cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
                            click_ripple_anim = (cx, cy, now)
                            print(f"[DWELL] Dot ({r}, {c}) -> {'ON' if grid_dots[r,c] else 'OFF'} (Hold: {DWELL_TRIGGER_TIME:.2f}s)", flush=True)
                    else:
                        dwell_progress = 0.0
                else:
                    dwell_dot = active_dot
                    dwell_start_time = now
                    dwell_lockout_dot = None
                    dwell_progress = 0.0
            else:
                dwell_dot = None
                dwell_start_time = now
                dwell_lockout_dot = None
                dwell_progress = 0.0

            last_pinch_state = is_pinching

            # Quick gestures when hand is outside the active dots
            if not is_master_locked and active_dot is None and not brightness_active and (now - last_gesture_cmd_time >= 2.5):
                if gesture == "PALM":
                    is_heartbeat_active = True
                    is_scroll_active = False
                    heartbeat_start_time = now
                    last_heart_state = None
                    last_gesture_cmd_time = now
                    print("[GESTURE] PALM detected -> Beating Heart activated (~70 BPM)", flush=True)

            # 4.5. Dynamic Animations (Heartbeat & Right-to-Left Typography Scrolling)
            if is_heartbeat_active and not is_master_locked:
                hb_frame, hb_state = get_heartbeat_frame(now, heartbeat_start_time)
                if hb_state != last_heart_state:
                    last_heart_state = hb_state
                    grid_dots[:] = hb_frame
                    comm.send_frame(grid_dots)

            elif is_scroll_active and not is_master_locked:
                if now - last_scroll_time >= SCROLL_SPEED:
                    last_scroll_time = now
                    max_steps = max(1, len(scroll_cols) - 7)
                    scroll_step = (scroll_step + 1) % max_steps
                    window = scroll_cols[scroll_step : scroll_step + 8]
                    for c_idx in range(8):
                        col_byte = window[c_idx] if c_idx < len(window) else 0x00
                        for r_idx in range(8):
                            grid_dots[r_idx, c_idx] = (col_byte >> r_idx) & 1
                    comm.send_frame(grid_dots)

            # 5. Render Minimal UI (Aspect-Correct Camera Left, Matrix Right)
            render_ui(canvas, cam_cropped, tip_canvas, active_dot, dwell_progress,
                      comm.udp_ip, comm.udp_port, gesture, brightness_active,
                      transpose=comm.transpose, invert=comm.invert)

            # 6. Display Window
            cv2.imshow(WINDOW_NAME, canvas)

            # 7. Keyboard Shortcuts & Typography Input
            raw_key = cv2.waitKey(1)
            key = raw_key & 0xFF if raw_key != -1 else -1

            if key != -1:
                if is_typing_mode:
                    if key in (13, 10):  # ENTER key - confirm and scroll
                        is_typing_mode = False
                        is_scroll_active = True
                        is_heartbeat_active = False
                        scroll_cols = build_scrolling_columns(custom_message)
                        scroll_step = 0
                        last_scroll_time = time.time()
                        print(f"[TYPOGRAPHY] Message updated & scrolling: '{custom_message}'", flush=True)
                    elif key == 27:  # ESC key - cancel typing
                        is_typing_mode = False
                        print("[TYPOGRAPHY] Typing mode closed.", flush=True)
                    elif key in (8, 127):  # Backspace
                        if len(custom_message) > 0:
                            custom_message = custom_message[:-1]
                            scroll_cols = build_scrolling_columns(custom_message)
                    elif 32 <= key <= 126:  # Printable ASCII
                        if len(custom_message) < 24:
                            custom_message += chr(key).upper()
                            scroll_cols = build_scrolling_columns(custom_message)
                else:
                    if key in (ord('q'), ord('Q'), 27):
                        break
                    elif key == ord(' '):
                        is_master_locked = not is_master_locked
                        status_str = "LOCKED (No changes allowed)" if is_master_locked else "UNLOCKED"
                        print(f"[LOCK] Master Lock: {status_str}", flush=True)
                    elif key in (ord('m'), ord('M')):
                        if is_master_locked:
                            print("[LOCK] System is LOCKED. Press SPACE or click [LOCKED] to unlock.", flush=True)
                        else:
                            is_typing_mode = True
                            print(f"[TYPOGRAPHY] Edit Mode: ACTIVE - Type message and press ENTER (Current: '{custom_message}')", flush=True)
                    elif key in (ord('h'), ord('H')):
                        if is_master_locked:
                            print("[LOCK] System is LOCKED. Press SPACE or click [LOCKED] to unlock.", flush=True)
                        else:
                            is_heartbeat_active = not is_heartbeat_active
                            is_scroll_active = False
                            if is_heartbeat_active:
                                heartbeat_start_time = time.time()
                                last_heart_state = None
                                print("[ANIMATION] Human Heartbeat mode ACTIVE (~70 BPM lub-dub)", flush=True)
                            else:
                                grid_dots.fill(0)
                                comm.send_frame(grid_dots)
                                print("[ANIMATION] Heartbeat stopped", flush=True)
                    elif key in (ord('c'), ord('C')):
                        if is_master_locked:
                            print("[LOCK] System is LOCKED. Press SPACE or click [LOCKED] to unlock.", flush=True)
                        else:
                            is_heartbeat_active = False
                            is_scroll_active = False
                            grid_dots.fill(0)
                            comm.send_frame(grid_dots)
                            print("[MATRIX] Cleared", flush=True)
                    elif key in (ord('s'), ord('S')):
                        if is_master_locked:
                            print("[LOCK] System is LOCKED. Press SPACE or click [LOCKED] to unlock.", flush=True)
                        else:
                            is_heartbeat_active = False
                            is_scroll_active = False
                            smile = [
                                [0,0,1,1,1,1,0,0],
                                [0,1,0,0,0,0,1,0],
                                [1,0,1,0,0,1,0,1],
                                [1,0,0,0,0,0,0,1],
                                [1,0,1,0,0,1,0,1],
                                [1,0,0,1,1,0,0,1],
                                [0,1,0,0,0,0,1,0],
                                [0,0,1,1,1,1,0,0]
                            ]
                            grid_dots[:] = smile
                            comm.send_frame(grid_dots)
                            print("[MATRIX] Smile pattern displayed", flush=True)
                    elif key in (ord('t'), ord('T')):
                        comm.transpose = not comm.transpose
                        print(f"[KEYBOARD] Transpose (row <-> col): {'ON' if comm.transpose else 'OFF'}", flush=True)
                        comm.send_frame(grid_dots)
                    elif key in (ord('i'), ord('I')):
                        comm.invert = not comm.invert
                        print(f"[KEYBOARD] Invert Polarity: {'ON' if comm.invert else 'OFF'}", flush=True)
                        comm.send_frame(grid_dots)
                    elif key in (ord('l'), ord('L')):
                        control_area['locked'] = not control_area['locked']
                        print(f"[KEYBOARD] Area: {'LOCKED' if control_area['locked'] else 'EDIT'}", flush=True)

            # 8. FPS Limiter
            proc_time = time.perf_counter() - frame_start_time
            sleep_needed = FRAME_INTERVAL - proc_time
            if sleep_needed > 0.0005:
                time.sleep(sleep_needed)

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Controller shut down cleanly.", flush=True)


if __name__ == "__main__":
    main()
