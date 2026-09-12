#!/usr/bin/env python3
"""
================================================================================
Project: Minimal 8x8 LED Matrix Controller
Script:  cv_matrix_controller.py

Layout:
  - Left Side:  Camera View with Fixed Interactive Control Box (Air-Pad)
  - Right Side: 8x8 LED Matrix Display with minimal Brightness & Action buttons
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
    def __init__(self, udp_ip="10.194.177.102", udp_port=8888, serial_port=None, baud_rate=115200):
        self.udp_ip = udp_ip
        self.udp_port = udp_port
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

    def send_toggle_dot(self, r, c):
        """Toggles dot at row r, col c on physical matrix."""
        self.send_command(f"TOGGLE:{r},{c}")


# ==============================================================================
# 2. HAND GESTURE & LANDMARK ANALYZER
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
    """Extracts hand landmarks using MediaPipe."""
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
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
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
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]

            wrist = pts[0]
            thumb_tip = pts[4]
            index_tip = pts[8]
            middle_tip = pts[12]
            ring_tip = pts[16]
            pinky_tip = pts[20]

            tip_raw_px = index_tip

            hand_scale = max(self.dist(pts[0], pts[9]), 20.0)
            pinch_dist_norm = self.dist(thumb_tip, index_tip) / hand_scale
            is_pinching = (pinch_dist_norm < 0.28)

            index_extended = self.dist(index_tip, wrist) > self.dist(pts[6], wrist) * 1.2
            middle_extended = self.dist(middle_tip, wrist) > self.dist(pts[10], wrist) * 1.2
            ring_extended = self.dist(ring_tip, wrist) > self.dist(pts[14], wrist) * 1.2
            pinky_extended = self.dist(pinky_tip, wrist) > self.dist(pts[18], wrist) * 1.2

            if is_pinching:
                gesture_name = "PINCH"
            elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
                gesture_name = "POINT"
            elif index_extended and middle_extended and not ring_extended and not pinky_extended:
                gesture_name = "PEACE"
            elif index_extended and middle_extended and ring_extended and pinky_extended:
                gesture_name = "PALM"
            elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                gesture_name = "FIST"
            else:
                gesture_name = "TRACK"

            # Subtle clean skeleton lines on camera
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame_bgr, pts[start_idx], pts[end_idx], (0, 200, 240), 1, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame_bgr, pt, 2, (0, 240, 255), -1)

            if is_pinching:
                pinch_mid = ((thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2)
                cv2.circle(frame_bgr, pinch_mid, 8, (0, 255, 255), -1)

        return hand_detected, gesture_name, tip_raw_px, is_pinching


# ==============================================================================
# 3. UI GEOMETRY & CONSTANTS
# ==============================================================================
WINDOW_W = 1240
WINDOW_H = 720

# LEFT SIDE: Camera Viewport
CAM_X = 40
CAM_Y = 80
CAM_W = 550
CAM_H = 460

# Fixed Interactive Control Box (Air-Pad) on Camera
control_area = {
    'x': CAM_X + (CAM_W - 380) // 2,  # Centered horizontally
    'y': CAM_Y + (CAM_H - 380) // 2,  # Centered vertically
    'w': 380,
    'h': 380,
    'locked': True
}

# Toolbar below Camera
AREA_BTN_Y = CAM_Y + CAM_H + 20
AREA_BTN_H = 32
AREA_BUTTONS = {
    'LOCK_TOGGLE': (CAM_X,       AREA_BTN_Y, 130, AREA_BTN_H),
    'CENTER':      (CAM_X + 145, AREA_BTN_Y, 90,  AREA_BTN_H),
    'CYCLE_SIZE':  (CAM_X + 250, AREA_BTN_Y, 110, AREA_BTN_H),
    'RESET':       (CAM_X + 375, AREA_BTN_Y, 90,  AREA_BTN_H)
}

# RIGHT SIDE: 8x8 Matrix Display (Exact minimal style from user's screenshot)
MATRIX_ORIGIN_X = 720
MATRIX_ORIGIN_Y = 135
DOT_SPACING = 54
DOT_RADIUS = 18

# 8x8 Dot Matrix State (1 = ON, 0 = OFF)
grid_dots = np.zeros((8, 8), dtype=np.uint8)

# Minimal Brightness Slider
SLIDER_X = 720
SLIDER_Y = 575
SLIDER_W = 378
SLIDER_H = 6
current_brightness = 75
is_dragging_brightness = False

# Minimal Action Buttons
BTN_W = 105
BTN_H = 34
BTN_GAP = 31
BTN_Y = 615
BUTTONS = {
    'CLEAR': (SLIDER_X,                   BTN_Y, BTN_W, BTN_H, "Clear"),
    'HEART': (SLIDER_X + BTN_W + BTN_GAP, BTN_Y, BTN_W, BTN_H, "Heart"),
    'SMILE': (SLIDER_X + (BTN_W + BTN_GAP)*2, BTN_Y, BTN_W, BTN_H, "Smile")
}

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
# 4. COORDINATE MAPPING & HIT TESTING
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


# ==============================================================================
# 5. MOUSE EVENT HANDLER
# ==============================================================================
def on_mouse_event(event, x, y, flags, param):
    """Handles mouse click & drag for dots, slider, buttons, and area control."""
    global mouse_pos, current_brightness, is_dragging_brightness
    global is_dragging_area, is_resizing_area, drag_offset, click_ripple_anim

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
            control_area['x'] = CAM_X + (CAM_W - control_area['w']) // 2
            control_area['y'] = CAM_Y + (CAM_H - control_area['h']) // 2
            print("[AREA] Centered on camera view", flush=True)
            return

        elif is_inside_rect(x, y, AREA_BUTTONS['CYCLE_SIZE']):
            sizes = [300, 380, 440]
            curr = control_area['w']
            next_size = sizes[(sizes.index(curr) + 1) % len(sizes)] if curr in sizes else 380
            control_area['w'] = next_size
            control_area['h'] = next_size
            control_area['x'] = max(CAM_X + 2, min(CAM_X + CAM_W - next_size - 2, control_area['x']))
            control_area['y'] = max(CAM_Y + 2, min(CAM_Y + CAM_H - next_size - 2, control_area['y']))
            print(f"[AREA] Size: {next_size}x{next_size}", flush=True)
            return

        elif is_inside_rect(x, y, AREA_BUTTONS['RESET']):
            control_area['w'] = 380
            control_area['h'] = 380
            control_area['x'] = CAM_X + (CAM_W - 380) // 2
            control_area['y'] = CAM_Y + (CAM_H - 380) // 2
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

        # 3. Direct click on matrix dot
        dot = get_dot_at_xy(x, y)
        if dot is not None:
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
            is_dragging_brightness = True
            ratio = (x - SLIDER_X) / float(SLIDER_W)
            current_brightness = int(max(0, min(100, ratio * 100)))
            comm.send_brightness(current_brightness)
            print(f"[MOUSE] Brightness: {current_brightness}%", flush=True)
            return

        # 5. Click on Action Buttons
        for key, (bx, by, bw, bh, label) in BUTTONS.items():
            if is_inside_rect(x, y, (bx, by, bw, bh)):
                if key == 'CLEAR':
                    grid_dots.fill(0)
                    comm.send_command("PATTERN:CLEAR")
                elif key == 'HEART':
                    heart = [
                        [0,1,1,0,0,1,1,0],
                        [1,1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1,1],
                        [0,1,1,1,1,1,1,0],
                        [0,0,1,1,1,1,0,0],
                        [0,0,0,1,1,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0]
                    ]
                    grid_dots[:] = heart
                    comm.send_command("PATTERN:HEART")
                elif key == 'SMILE':
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
                    comm.send_command("PATTERN:SMILE")
                return

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging_brightness:
            ratio = (x - SLIDER_X) / float(SLIDER_W)
            current_brightness = int(max(0, min(100, ratio * 100)))
            comm.send_brightness(current_brightness)

        elif is_dragging_area and not control_area['locked']:
            new_x = x - drag_offset[0]
            new_y = y - drag_offset[1]
            control_area['x'] = max(CAM_X + 2, min(CAM_X + CAM_W - control_area['w'] - 2, new_x))
            control_area['y'] = max(CAM_Y + 2, min(CAM_Y + CAM_H - control_area['h'] - 2, new_y))

        elif is_resizing_area and not control_area['locked']:
            new_w = max(200, min(CAM_W - (control_area['x'] - CAM_X) - 2, x - control_area['x']))
            new_h = max(200, min(CAM_H - (control_area['y'] - CAM_Y) - 2, y - control_area['y']))
            side = min(new_w, new_h)
            control_area['w'] = side
            control_area['h'] = side

    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging_brightness = False
        is_dragging_area = False
        is_resizing_area = False


# ==============================================================================
# 6. MINIMAL RENDERING ENGINE
# ==============================================================================
def render_ui(canvas, frame_cam, tip_canvas, active_dot, dwell_progress, ip, port, gesture):
    """Draws the clean widescreen UI: Camera View on Left, 8x8 Matrix on Right."""
    global click_ripple_anim

    # 1. TOP HEADER (Subtle, clean, monochrome)
    cv2.putText(canvas, "CAMERA VIEW  (CONTROL AREA)", (CAM_X, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 148), 1, cv2.LINE_AA)

    cv2.putText(canvas, "8x8 LED MATRIX", (MATRIX_ORIGIN_X, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 225), 1, cv2.LINE_AA)

    # Status indicator (small clean green dot + IP)
    cv2.circle(canvas, (WINDOW_W - 190, 40), 4, (0, 220, 100), -1)
    cv2.putText(canvas, f"{ip}:{port}", (WINDOW_W - 176, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 135), 1, cv2.LINE_AA)

    # Subtle divider line
    cv2.line(canvas, (CAM_X, 60), (WINDOW_W - CAM_X, 60), (36, 36, 40), 1)

    # 2. LEFT PANEL: Camera Feed Viewport
    cam_resized = cv2.resize(frame_cam, (CAM_W, CAM_H))
    canvas[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = cam_resized
    cv2.rectangle(canvas, (CAM_X, CAM_Y), (CAM_X + CAM_W, CAM_Y + CAM_H), (45, 45, 52), 1)

    # Fixed Interactive Control Box (Air-Pad)
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
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), box_color, 2 if is_locked else 2)

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
    cv2.rectangle(canvas, (bx, by - 22), (bx + 170, by), box_color, -1)
    cv2.putText(canvas, tag_txt, (bx + 8, by - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (15, 15, 20), 1, cv2.LINE_AA)

    # Resize handle (when in EDIT mode)
    if not is_locked:
        handle_x = bx + bw - 18
        handle_y = by + bh - 18
        cv2.rectangle(canvas, (handle_x, handle_y), (bx + bw, by + bh), (0, 160, 255), -1)
        cv2.putText(canvas, "+", (handle_x + 4, handle_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

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

    # 5. Minimal Action Buttons (Clear, Heart, Smile)
    for key, (bx, by, bw, bh, label) in BUTTONS.items():
        is_hover = is_inside_rect(mouse_pos[0], mouse_pos[1], (bx, by, bw, bh))
        bg_col = (38, 38, 44) if is_hover else (28, 28, 32)
        border_col = (110, 110, 120) if is_hover else (55, 55, 62)

        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), bg_col, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_col, 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (bx + (bw - len(label)*9)//2, by + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 225), 1, cv2.LINE_AA)

    # 6. Bottom Footer
    cv2.line(canvas, (CAM_X, 672), (WINDOW_W - CAM_X, 672), (32, 32, 36), 1)
    target_text = f"Target: Dot ({active_dot[0]}, {active_dot[1]})" if active_dot else "Waiting for hand"
    footer_text = f"Fixed Control Box Active   |   Hold Delay: {DWELL_TRIGGER_TIME:.2f}s   |   {target_text}"
    cv2.putText(canvas, footer_text, (CAM_X, 696),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (115, 115, 122), 1, cv2.LINE_AA)


# ==============================================================================
# 7. MAIN LOOP
# ==============================================================================
def main():
    global dwell_dot, dwell_start_time, last_air_click_time, last_pinch_state
    global last_gesture_cmd_time, current_brightness, click_ripple_anim
    global dwell_lockout_dot, DWELL_TRIGGER_TIME

    parser = argparse.ArgumentParser(description="Minimal 8x8 LED Matrix Controller")
    parser.add_argument("--ip", type=str, default="10.194.177.102", help="ESP32 IP address")
    parser.add_argument("--port", type=int, default=8888, help="ESP32 UDP port")
    parser.add_argument("--serial", type=str, default=None, help="Optional Serial Port")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--demo", action="store_true", help="Run in simulation mode")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS")
    parser.add_argument("--dwell", type=float, default=0.75, help="Dwell delay in seconds")
    args = parser.parse_args()

    TARGET_FPS = float(args.fps)
    FRAME_INTERVAL = 1.0 / TARGET_FPS
    DWELL_TRIGGER_TIME = float(args.dwell)

    print("\n" + "=" * 56, flush=True)
    print("  8x8 LED MATRIX CONTROLLER", flush=True)
    print(f"  Target ESP32:       {args.ip}:{args.port}", flush=True)
    print(f"  Hold Delay:         {DWELL_TRIGGER_TIME:.2f}s (single-fire anti-bounce)", flush=True)
    print("  Camera View (Left)  |  8x8 Matrix (Right)", flush=True)
    print("=" * 56 + "\n", flush=True)

    comm = MatrixCommunicator(udp_ip=args.ip, udp_port=args.port, serial_port=args.serial)
    analyzer = HandGestureAnalyzer()

    use_simulation = args.demo or not analyzer.available
    cap = None

    if not use_simulation:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[NOTE] Camera {args.camera} unavailable. Running in simulation mode.", flush=True)
            use_simulation = True

    if not use_simulation:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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

                # Map fingertip from raw camera frame to canvas camera viewport
                tip_canvas = None
                if tip_raw is not None:
                    raw_h, raw_w = frame_cam.shape[:2]
                    scale_x = CAM_W / float(raw_w)
                    scale_y = CAM_H / float(raw_h)
                    tip_canvas = (
                        int(CAM_X + tip_raw[0] * scale_x),
                        int(CAM_Y + tip_raw[1] * scale_y)
                    )
            else:
                # Simulation mode
                frame_cam = np.full((CAM_H, CAM_W, 3), 25, dtype=np.uint8)
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

            # 3. Map Hand from Fixed Control Box to Matrix Dot
            active_dot = None
            dwell_progress = 0.0

            if tip_canvas is not None:
                active_dot = get_dot_from_control_box(tip_canvas[0], tip_canvas[1])

            # Mouse hover fallback (on matrix or control box)
            if active_dot is None:
                active_dot = get_dot_at_xy(mouse_pos[0], mouse_pos[1])
                if active_dot is None:
                    active_dot = get_dot_from_control_box(mouse_pos[0], mouse_pos[1])

            now = time.time()

            # PINCH-TO-CLICK (Instant toggle inside Fixed Box)
            if is_pinching and not last_pinch_state and (now - last_air_click_time >= 0.45):
                if active_dot is not None:
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
            if active_dot is not None and not is_pinching:
                if active_dot == dwell_dot:
                    if active_dot != dwell_lockout_dot:
                        dwell_time = now - dwell_start_time
                        dwell_progress = min(1.0, dwell_time / DWELL_TRIGGER_TIME)
                        if dwell_time >= DWELL_TRIGGER_TIME and (now - last_air_click_time >= 0.50):
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
            if active_dot is None and (now - last_gesture_cmd_time >= 2.5):
                if gesture == "PALM":
                    heart = [
                        [0,1,1,0,0,1,1,0],
                        [1,1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1,1],
                        [0,1,1,1,1,1,1,0],
                        [0,0,1,1,1,1,0,0],
                        [0,0,0,1,1,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0]
                    ]
                    grid_dots[:] = heart
                    comm.send_command("PATTERN:HEART")
                    last_gesture_cmd_time = now

            # 4. Render Minimal UI (Camera Left, Matrix Right)
            render_ui(canvas, frame_cam, tip_canvas, active_dot, dwell_progress, comm.udp_ip, comm.udp_port, gesture)

            # 5. Display Window
            cv2.imshow(WINDOW_NAME, canvas)

            # 6. Keyboard Shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('c'), ord('C')):
                grid_dots.fill(0)
                comm.send_command("PATTERN:CLEAR")
            elif key in (ord('h'), ord('H')):
                comm.send_command("PATTERN:HEART")
            elif key in (ord('s'), ord('S')):
                comm.send_command("PATTERN:SMILE")
            elif key in (ord('l'), ord('L')):
                control_area['locked'] = not control_area['locked']
                print(f"[KEYBOARD] Area: {'LOCKED' if control_area['locked'] else 'EDIT'}", flush=True)

            # 7. FPS Limiter
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
