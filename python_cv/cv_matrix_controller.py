#!/usr/bin/env python3
"""
================================================================================
Project: Neural Matrix Controller & Hand Gesture Interface
Script:  cv_matrix_controller.py

Key Features:
  1. MODERN CYBER-TECH HUD UI:
     - Sleek dark theme with neon cyan, amber, and emerald accents.
     - Dedicated modular panels: Camera Viewport, Hardware Matrix Mirror,
       Control Dock, Brightness Slider, and Live Telemetry Bar.
     - Ultra-realistic 64-LED virtual board with multi-stage glow, bloom,
       and specular reflection effects.

  2. DEDICATED FIXED CONTROL AREA ("AIR-PAD / TOUCHPAD"):
     - Clearly marked, fixed interaction zone on the camera viewport.
     - 8x8 Alignment Subgrid overlay with Row (0-7) and Column (0-7) indices.
     - Live Fingertip Reticle snapping directly to grid cells with instant feedback.
     - Configurable Touch Point Delay (Dwell: 0.50s, 0.75s, 1.00s) with single-fire
       anti-bounce protection (won't re-toggle until finger leaves the dot).
     - Visual Dwell Progress ring and Pinch-to-Click ripple effect.
     - Strict boundary isolation: movements outside the area never toggle dots.
     - AREA CONTROL: Toggle between [LOCKED] and [EDIT] mode. In EDIT mode,
       click & drag or resize the control area using your mouse, or use
       quick presets ([CENTER], [RESET], [SIZE]).

  3. GESTURE & HARDWARE INTEGRATION:
     - Pointing & Pinch-to-click / Dwell-to-click for 64 individual dots.
     - Predefined gestures: Open Palm (Heart), Peace Sign (Hello), Fist (Greeting).
     - Dual communication: Wi-Fi UDP (port 8888) + USB Serial (115200).
     - Non-blocking frame engine locked to target FPS.
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
    print("[WARNING] MediaPipe not installed. Running in mouse/simulation mode.")

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
        print(f"[DOWNLOAD] Downloading MediaPipe Hand Landmarker model to {MODEL_PATH}...", flush=True)
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[DOWNLOAD] Model downloaded successfully.", flush=True)
        except Exception as e:
            print(f"[DOWNLOAD ERROR] Could not download model: {e}", flush=True)


# ==============================================================================
# 1. COMMUNICATION CLIENT (UDP & SERIAL)
# ==============================================================================
class MatrixCommunicator:
    """Manages low-latency UDP packet transmission and USB Serial fallback."""
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
                print(f"[SERIAL] Connected to {serial_port} at {baud_rate} baud.", flush=True)
            except Exception as e:
                print(f"[SERIAL WARNING] Could not open {serial_port}: {e}", flush=True)

        self.last_brightness_send_time = 0
        self.last_sent_brightness = -1
        self.last_sent_cmd = "IDLE"
        self.last_cmd_status = "READY"
        self.tx_count = 0

    def send_command(self, cmd_str):
        """Sends an ASCII command string to the ESP32-C3."""
        cmd_str = cmd_str.strip()
        if not cmd_str:
            return

        payload = (cmd_str + "\n").encode('utf-8')

        try:
            self.sock.sendto(payload, (self.udp_ip, self.udp_port))
            self.last_cmd_status = "TX OK (UDP)"
        except Exception as e:
            self.last_cmd_status = f"UDP ERR: {e}"

        if self.ser and self.ser.is_open:
            try:
                self.ser.write(payload)
                self.last_cmd_status += " + SERIAL"
            except Exception:
                pass

        self.last_sent_cmd = cmd_str
        self.tx_count += 1
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
    """Extracts landmarks using MediaPipe Tasks HandLandmarker."""
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
            print(f"[WARNING] MediaPipe HandLandmarker init error: {e}", flush=True)

    def dist(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def analyze(self, frame_bgr):
        if not self.available:
            return False, "NONE", None, None, False, None

        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.detector.detect(mp_image)

        hand_detected = False
        gesture_name = "NONE"
        index_tip_px = None
        thumb_tip_px = None
        is_pinching = False
        norm_pointer = None

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand_detected = True
            lm_list = result.hand_landmarks[0]

            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
            norm_pts = [(lm.x, lm.y) for lm in lm_list]

            wrist = pts[0]
            thumb_tip = pts[4]
            index_tip = pts[8]
            middle_tip = pts[12]
            ring_tip = pts[16]
            pinky_tip = pts[20]

            index_tip_px = index_tip
            thumb_tip_px = thumb_tip

            hand_scale = max(self.dist(pts[0], pts[9]), 20.0)

            index_extended = self.dist(index_tip, wrist) > self.dist(pts[6], wrist) * 1.2
            middle_extended = self.dist(middle_tip, wrist) > self.dist(pts[10], wrist) * 1.2
            ring_extended = self.dist(ring_tip, wrist) > self.dist(pts[14], wrist) * 1.2
            pinky_extended = self.dist(pinky_tip, wrist) > self.dist(pts[18], wrist) * 1.2

            pinch_dist_norm = self.dist(thumb_tip, index_tip) / hand_scale
            is_pinching = (pinch_dist_norm < 0.28)

            if is_pinching:
                gesture_name = "PINCH CLICK"
            elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
                gesture_name = "POINTING"
            elif index_extended and middle_extended and not ring_extended and not pinky_extended:
                gesture_name = "PEACE SIGN"
            elif index_extended and middle_extended and ring_extended and pinky_extended:
                gesture_name = "OPEN PALM"
            elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                gesture_name = "FIST"
            else:
                gesture_name = "TRACKING"

            norm_pointer = (norm_pts[8][0], norm_pts[8][1])

            # Draw sleek cyber-skeleton on camera
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame_bgr, pts[start_idx], pts[end_idx], (0, 210, 255), 2)
            for i, pt in enumerate(pts):
                color = (0, 255, 180) if i in (4, 8) else (255, 120, 50)
                cv2.circle(frame_bgr, pt, 3, color, -1)

            if is_pinching:
                pinch_mid = ((thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2)
                cv2.circle(frame_bgr, pinch_mid, 12, (0, 255, 255), -1)
                cv2.circle(frame_bgr, pinch_mid, 18, (0, 200, 255), 2)

        return hand_detected, gesture_name, index_tip_px, thumb_tip_px, is_pinching, norm_pointer


# ==============================================================================
# 3. UI LAYOUT CONSTANTS & GEOMETRY
# ==============================================================================
CANVAS_W = 1280
CANVAS_H = 720

# Viewport for Camera Feed (Left Panel)
CAM_VIEW_X = 25
CAM_VIEW_Y = 65
CAM_VIEW_W = 600
CAM_VIEW_H = 450

# Dedicated Fixed Control Area ("Air-Pad") inside Camera View
# Default: centered 360x360 box inside camera view
control_area = {
    'x': CAM_VIEW_X + (CAM_VIEW_W - 360) // 2,  # 145
    'y': CAM_VIEW_Y + (CAM_VIEW_H - 360) // 2,  # 110
    'w': 360,
    'h': 360,
    'locked': True
}

# 8x8 Dot Matrix State (1 = ON, 0 = OFF)
grid_dots = np.zeros((8, 8), dtype=np.uint8)

# Right Panel (Virtual Matrix & Controls)
RIGHT_PANEL_X = 645
RIGHT_PANEL_Y = 65
RIGHT_PANEL_W = 610
RIGHT_PANEL_H = 580

# 8x8 Virtual Matrix Geometry (Centered inside right PCB card)
MATRIX_ORIGIN_X = 812
MATRIX_ORIGIN_Y = 135
DOT_SPACING = 40
DOT_RADIUS = 14

# Brightness Slider Geometry (Horizontal)
SLIDER_X = 725
SLIDER_Y = 478
SLIDER_W = 440
SLIDER_H = 20
current_brightness = 75
is_dragging_brightness = False

# Quick Preset Buttons (Two rows of 3 buttons)
BTN_W = 145
BTN_H = 34
BTN_GAP_X = 16
BTN_START_X = 720
BTN_ROW1_Y = 524
BTN_ROW2_Y = 570

BUTTONS = {
    'CLEAR':  (BTN_START_X,                            BTN_ROW1_Y, BTN_W, BTN_H, "CLEAR",      (55, 55, 70), (220, 220, 240)),
    'HEART':  (BTN_START_X + BTN_W + BTN_GAP_X,       BTN_ROW1_Y, BTN_W, BTN_H, "HEART",      (80, 25, 50), (255, 110, 180)),
    'SMILE':  (BTN_START_X + (BTN_W + BTN_GAP_X)*2,   BTN_ROW1_Y, BTN_W, BTN_H, "SMILE",      (30, 70, 50), (80, 240, 160)),
    'PULSE':  (BTN_START_X,                            BTN_ROW2_Y, BTN_W, BTN_H, "PULSE",      (75, 40, 20), (255, 170, 60)),
    'HELLO':  (BTN_START_X + BTN_W + BTN_GAP_X,       BTN_ROW2_Y, BTN_W, BTN_H, "MSG: HELLO", (25, 60, 80), (70, 200, 255)),
    'GREET':  (BTN_START_X + (BTN_W + BTN_GAP_X)*2,   BTN_ROW2_Y, BTN_W, BTN_H, "MSG: GREET", (55, 30, 75), (210, 120, 255))
}

# Area Customization & Touch Delay Buttons (below Camera View)
AREA_BTN_Y = CAM_VIEW_Y + CAM_VIEW_H + 15  # 530
AREA_BTN_H = 32
AREA_BUTTONS = {
    'LOCK_TOGGLE': (CAM_VIEW_X + 5,   AREA_BTN_Y, 115, AREA_BTN_H),
    'DELAY_CYCLE': (CAM_VIEW_X + 128, AREA_BTN_Y, 125, AREA_BTN_H),
    'CYCLE_SIZE':  (CAM_VIEW_X + 261, AREA_BTN_Y, 115, AREA_BTN_H),
    'CENTER':      (CAM_VIEW_X + 384, AREA_BTN_Y, 95,  AREA_BTN_H),
    'RESET':       (CAM_VIEW_X + 487, AREA_BTN_Y, 85,  AREA_BTN_H)
}

# Touch Point Delay & Anti-Bounce Settings
DWELL_PRESETS = [0.50, 0.75, 1.00]
dwell_preset_idx = 1  # 0.75s default (comfortable, deliberate touch)
dwell_trigger_time = DWELL_PRESETS[dwell_preset_idx]
dwell_lockout_dot = None  # Prevents re-triggering the same dot until finger leaves

# Mouse & Drag State
mouse_pos = (-1, -1)
is_dragging_area = False
is_resizing_area = False
drag_offset = (0, 0)

# Interaction Timers
dwell_dot = None
dwell_start_time = 0
last_air_click_time = 0
last_pinch_state = False
last_gesture_cmd_time = 0
click_ripple_anim = None  # (cx, cy, start_time)


# ==============================================================================
# 4. COORDINATE MAPPING & HIT TESTING
# ==============================================================================
def is_inside_rect(px, py, rect):
    rx, ry, rw, rh = rect
    return (rx <= px <= rx + rw) and (ry <= py <= ry + rh)

def get_dot_at_xy(px, py):
    """Returns (r, c) if coordinate (px, py) falls on any virtual 8x8 dot."""
    for r in range(8):
        for c in range(8):
            cx = MATRIX_ORIGIN_X + c * DOT_SPACING
            cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
            if math.hypot(px - cx, py - cy) <= DOT_RADIUS + 6:
                return (r, c)
    return None

def get_cell_from_control_area(px, py):
    """
    Maps pixel position (px, py) inside the Fixed Control Area directly to (r, c).
    Returns None if (px, py) is outside the Fixed Control Area.
    """
    ax = control_area['x']
    ay = control_area['y']
    aw = control_area['w']
    ah = control_area['h']

    if ax <= px <= ax + aw and ay <= py <= ay + ah:
        c = int(((px - ax) / float(aw)) * 8)
        r = int(((py - ay) / float(ah)) * 8)
        return (max(0, min(7, r)), max(0, min(7, c)))
    return None


# ==============================================================================
# 5. MOUSE EVENT HANDLER
# ==============================================================================
def on_mouse_event(event, x, y, flags, param):
    """Handles mouse interaction for virtual dots, sliders, presets, and area customization."""
    global mouse_pos, current_brightness, is_dragging_brightness
    global is_dragging_area, is_resizing_area, drag_offset, click_ripple_anim
    global dwell_preset_idx, dwell_trigger_time

    comm = param
    mouse_pos = (x, y)

    ax = control_area['x']
    ay = control_area['y']
    aw = control_area['w']
    ah = control_area['h']
    resize_handle_rect = (ax + aw - 24, ay + ah - 24, 24, 24)

    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. Click on Area Toolbar Buttons
        # Lock / Edit Toggle
        if is_inside_rect(x, y, AREA_BUTTONS['LOCK_TOGGLE']):
            control_area['locked'] = not control_area['locked']
            print(f"[CONTROL AREA] Mode toggled: {'LOCKED' if control_area['locked'] else 'EDIT/CUSTOMIZE'}", flush=True)
            return

        # Delay Cycle Button (0.50s -> 0.75s -> 1.00s)
        elif is_inside_rect(x, y, AREA_BUTTONS['DELAY_CYCLE']):
            dwell_preset_idx = (dwell_preset_idx + 1) % len(DWELL_PRESETS)
            dwell_trigger_time = DWELL_PRESETS[dwell_preset_idx]
            print(f"[TOUCH DELAY] Touch point dwell delay set to {dwell_trigger_time:.2f}s", flush=True)
            return

        # Center Area
        elif is_inside_rect(x, y, AREA_BUTTONS['CENTER']):
            control_area['x'] = CAM_VIEW_X + (CAM_VIEW_W - control_area['w']) // 2
            control_area['y'] = CAM_VIEW_Y + (CAM_VIEW_H - control_area['h']) // 2
            print("[CONTROL AREA] Centered inside camera viewport.", flush=True)
            return

        # Cycle Size (300 -> 360 -> 420)
        elif is_inside_rect(x, y, AREA_BUTTONS['CYCLE_SIZE']):
            sizes = [300, 360, 420]
            curr = control_area['w']
            next_size = sizes[(sizes.index(curr) + 1) % len(sizes)] if curr in sizes else 360
            control_area['w'] = next_size
            control_area['h'] = next_size
            # Clamp inside viewport
            control_area['x'] = max(CAM_VIEW_X + 5, min(CAM_VIEW_X + CAM_VIEW_W - next_size - 5, control_area['x']))
            control_area['y'] = max(CAM_VIEW_Y + 5, min(CAM_VIEW_Y + CAM_VIEW_H - next_size - 5, control_area['y']))
            print(f"[CONTROL AREA] Size changed to {next_size}x{next_size}", flush=True)
            return

        # Reset Area
        elif is_inside_rect(x, y, AREA_BUTTONS['RESET']):
            control_area['w'] = 360
            control_area['h'] = 360
            control_area['x'] = CAM_VIEW_X + (CAM_VIEW_W - 360) // 2
            control_area['y'] = CAM_VIEW_Y + (CAM_VIEW_H - 360) // 2
            control_area['locked'] = True
            print("[CONTROL AREA] Reset to factory default and locked.", flush=True)
            return

        # 2. If Area is in EDIT mode: check for drag or resize handles
        if not control_area['locked']:
            if is_inside_rect(x, y, resize_handle_rect):
                is_resizing_area = True
                return
            elif is_inside_rect(x, y, (ax, ay, aw, ah)):
                is_dragging_area = True
                drag_offset = (x - ax, y - ay)
                return

        # 3. Direct Click on Virtual Matrix 64 Dots
        dot = get_dot_at_xy(x, y)
        if dot is not None:
            r, c = dot
            grid_dots[r, c] ^= 1
            comm.send_toggle_dot(r, c)
            cx = MATRIX_ORIGIN_X + c * DOT_SPACING
            cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
            click_ripple_anim = (cx, cy, time.time())
            print(f"[MOUSE] Toggled Matrix Dot ({r}, {c}) -> {'ON' if grid_dots[r,c] else 'OFF'}", flush=True)
            return

        # 4. Brightness Slider Click
        slider_expand = (SLIDER_X - 10, SLIDER_Y - 10, SLIDER_W + 20, SLIDER_H + 20)
        if is_inside_rect(x, y, slider_expand):
            is_dragging_brightness = True
            ratio = (x - SLIDER_X) / float(SLIDER_W)
            current_brightness = int(max(0, min(100, ratio * 100)))
            comm.send_brightness(current_brightness)
            print(f"[MOUSE] Brightness set to {current_brightness}%", flush=True)
            return

        # 5. Preset Action Buttons
        for key, btn in BUTTONS.items():
            rect = (btn[0], btn[1], btn[2], btn[3])
            if is_inside_rect(x, y, rect):
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
                elif key == 'PULSE':
                    comm.send_command("ANIMATION:PULSE")
                elif key == 'HELLO':
                    comm.send_command("MESSAGE:HELLO")
                elif key == 'GREET':
                    comm.send_command("MESSAGE:HOW YOU DOING?")
                return

    elif event == cv2.EVENT_MOUSEMOVE:
        # Dragging Brightness
        if is_dragging_brightness:
            ratio = (x - SLIDER_X) / float(SLIDER_W)
            current_brightness = int(max(0, min(100, ratio * 100)))
            comm.send_brightness(current_brightness)

        # Dragging Control Area (when in EDIT mode)
        elif is_dragging_area and not control_area['locked']:
            new_x = x - drag_offset[0]
            new_y = y - drag_offset[1]
            # Clamp inside camera viewport
            control_area['x'] = max(CAM_VIEW_X + 4, min(CAM_VIEW_X + CAM_VIEW_W - control_area['w'] - 4, new_x))
            control_area['y'] = max(CAM_VIEW_Y + 4, min(CAM_VIEW_Y + CAM_VIEW_H - control_area['h'] - 4, new_y))

        # Resizing Control Area (when in EDIT mode)
        elif is_resizing_area and not control_area['locked']:
            new_w = max(200, min(CAM_VIEW_W - (control_area['x'] - CAM_VIEW_X) - 4, x - control_area['x']))
            new_h = max(200, min(CAM_VIEW_H - (control_area['y'] - CAM_VIEW_Y) - 4, y - control_area['y']))
            # Keep square aspect
            side = min(new_w, new_h)
            control_area['w'] = side
            control_area['h'] = side

    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging_brightness = False
        is_dragging_area = False
        is_resizing_area = False


# ==============================================================================
# 6. MODERN UI RENDERING ENGINE
# ==============================================================================
def draw_header_bar(canvas, ip, port, fps, gesture, serial_active):
    """Draws the futuristic top HUD header."""
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, 52), (20, 22, 28), -1)
    cv2.line(canvas, (0, 52), (CANVAS_W, 52), (0, 220, 255), 2)
    cv2.line(canvas, (0, 53), (CANVAS_W, 53), (0, 70, 90), 1)

    # Logo / Title
    cv2.circle(canvas, (24, 26), 7, (0, 255, 200), -1)
    cv2.circle(canvas, (24, 26), 11, (0, 200, 160), 1)
    cv2.putText(canvas, "NEURAL MATRIX CONTROLLER", (42, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "| 8x8 GESTURE INTERFACE", (370, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

    # ESP32 Status Pill
    badge_x = 590
    cv2.rectangle(canvas, (badge_x, 12), (badge_x + 230, 40), (30, 34, 44), -1)
    cv2.rectangle(canvas, (badge_x, 12), (badge_x + 230, 40), (70, 80, 100), 1)
    cv2.circle(canvas, (badge_x + 16, 26), 5, (0, 255, 100), -1)
    cv2.putText(canvas, f"ESP32: {ip}:{port}", (badge_x + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 230, 240), 1, cv2.LINE_AA)

    # FPS Badge
    fps_x = 835
    cv2.rectangle(canvas, (fps_x, 12), (fps_x + 105, 40), (30, 34, 44), -1)
    cv2.rectangle(canvas, (fps_x, 12), (fps_x + 105, 40), (70, 80, 100), 1)
    fps_color = (0, 255, 120) if fps >= 50 else (0, 220, 255)
    cv2.putText(canvas, f"FPS: {fps:.1f}", (fps_x + 14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, fps_color, 1, cv2.LINE_AA)

    # Gesture Indicator Pill
    g_colors = {
        'PINCH CLICK': (0, 230, 255),
        'POINTING':    (255, 200, 0),
        'OPEN PALM':   (0, 255, 120),
        'PEACE SIGN':  (255, 120, 180),
        'FIST':        (180, 120, 255),
        'TRACKING':    (200, 200, 200),
        'NONE':        (120, 120, 130)
    }
    g_col = g_colors.get(gesture, (150, 150, 150))
    pill_x = 955
    cv2.rectangle(canvas, (pill_x, 12), (pill_x + 305, 40), (28, 30, 38), -1)
    cv2.rectangle(canvas, (pill_x, 12), (pill_x + 305, 40), g_col, 2)
    cv2.circle(canvas, (pill_x + 16, 26), 6, g_col, -1)
    cv2.putText(canvas, f"GESTURE: {gesture}", (pill_x + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def draw_camera_viewport(canvas, frame_cam):
    """Embeds and frames the camera feed on the left panel."""
    # Frame Container Card
    cv2.rectangle(canvas, (CAM_VIEW_X - 6, CAM_VIEW_Y - 6),
                  (CAM_VIEW_X + CAM_VIEW_W + 6, CAM_VIEW_Y + CAM_VIEW_H + 6), (28, 32, 42), -1)
    cv2.rectangle(canvas, (CAM_VIEW_X - 6, CAM_VIEW_Y - 6),
                  (CAM_VIEW_X + CAM_VIEW_W + 6, CAM_VIEW_Y + CAM_VIEW_H + 6), (50, 60, 80), 1)

    # Resize and place camera frame
    cam_resized = cv2.resize(frame_cam, (CAM_VIEW_W, CAM_VIEW_H))
    canvas[CAM_VIEW_Y:CAM_VIEW_Y + CAM_VIEW_H, CAM_VIEW_X:CAM_VIEW_X + CAM_VIEW_W] = cam_resized

    # Corner brackets (Sci-Fi reticle look)
    bracket_len = 18
    corners = [
        (CAM_VIEW_X, CAM_VIEW_Y),
        (CAM_VIEW_X + CAM_VIEW_W, CAM_VIEW_Y),
        (CAM_VIEW_X, CAM_VIEW_Y + CAM_VIEW_H),
        (CAM_VIEW_X + CAM_VIEW_W, CAM_VIEW_Y + CAM_VIEW_H)
    ]
    for cx, cy in corners:
        dx = 1 if cx == CAM_VIEW_X else -1
        dy = 1 if cy == CAM_VIEW_Y else -1
        cv2.line(canvas, (cx, cy), (cx + dx * bracket_len, cy), (0, 240, 255), 2)
        cv2.line(canvas, (cx, cy), (cx, cy + dy * bracket_len), (0, 240, 255), 2)


def draw_fixed_control_area(canvas, active_cell, dwell_progress=0.0, tip_px=None):
    """
    Renders the dedicated Fixed Control Area (Air-Pad) with:
      - 8x8 translucent subgrid overlay with row & col indices
      - Active hover cell illumination
      - Fingertip reticle & dwell circle
      - Resizing / moving handles when in EDIT mode
    """
    ax = control_area['x']
    ay = control_area['y']
    aw = control_area['w']
    ah = control_area['h']
    is_locked = control_area['locked']

    # 1. Base Border & Background Tint
    border_color = (0, 230, 255) if is_locked else (0, 165, 255)  # Cyan if locked, Orange if edit
    title_text = f"FIXED AIR-PAD [LOCKED | {dwell_trigger_time:.2f}s]" if is_locked else "AIR-PAD [EDIT MODE: DRAG/RESIZE]"

    # Subtle translucent dark fill
    overlay = canvas.copy()
    cv2.rectangle(overlay, (ax, ay), (ax + aw, ay + ah), (15, 20, 30), -1)
    cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, canvas)

    # Outer border
    cv2.rectangle(canvas, (ax, ay), (ax + aw, ay + ah), border_color, 2)

    # 2. 8x8 Alignment Subgrid
    cw = aw / 8.0
    ch = ah / 8.0

    for i in range(1, 8):
        # Vertical grid line
        gx = int(ax + i * cw)
        cv2.line(canvas, (gx, ay), (gx, ay + ah), (50, 75, 100), 1)
        # Horizontal grid line
        gy = int(ay + i * ch)
        cv2.line(canvas, (ax, gy), (ax + aw, gy), (50, 75, 100), 1)

    # Grid Index Labels along top and left
    for c in range(8):
        lbl_x = int(ax + c * cw + cw / 2 - 4)
        cv2.putText(canvas, str(c), (lbl_x, ay - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
    for r in range(8):
        lbl_y = int(ay + r * ch + ch / 2 + 4)
        cv2.putText(canvas, str(r), (ax - 14, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

    # 3. Active Target Cell Highlight (if fingertip or mouse is hovering)
    if active_cell is not None:
        r, c = active_cell
        cell_x1 = int(ax + c * cw)
        cell_y1 = int(ay + r * ch)
        cell_x2 = int(cell_x1 + cw)
        cell_y2 = int(cell_y1 + ch)

        # Highlight cell fill
        cell_overlay = canvas.copy()
        cv2.rectangle(cell_overlay, (cell_x1, cell_y1), (cell_x2, cell_y2), (0, 240, 255), -1)
        cv2.addWeighted(cell_overlay, 0.35, canvas, 0.65, 0, canvas)

        # Glowing cell border
        cv2.rectangle(canvas, (cell_x1, cell_y1), (cell_x2, cell_y2), (0, 255, 255), 2)
        # Cell coordinate tag
        cv2.putText(canvas, f"({r},{c})", (cell_x1 + 3, cell_y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Fingertip Crosshair Reticle & Dwell Indicator
    if tip_px is not None and is_inside_rect(tip_px[0], tip_px[1], (ax, ay, aw, ah)):
        tx, ty = tip_px
        cv2.circle(canvas, (tx, ty), 12, (0, 255, 255), 2)
        cv2.circle(canvas, (tx, ty), 4, (0, 200, 255), -1)
        cv2.line(canvas, (tx - 16, ty), (tx + 16, ty), (0, 255, 255), 1)
        cv2.line(canvas, (tx, ty - 16), (tx, ty + 16), (0, 255, 255), 1)

        # Dwell progress arc (Smoothly fills over dwell_trigger_time)
        if dwell_progress > 0.0:
            end_angle = int(dwell_progress * 360)
            arc_color = (0, 255, 120) if dwell_progress < 1.0 else (0, 255, 255)
            cv2.ellipse(canvas, (tx, ty), (18, 18), -90, 0, end_angle, arc_color, 3)

    # 5. Top Banner on Control Area
    banner_w = 230 if is_locked else 240
    cv2.rectangle(canvas, (ax, ay - 24), (ax + banner_w, ay), border_color, -1)
    cv2.putText(canvas, title_text, (ax + 6, ay - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (15, 20, 30), 1, cv2.LINE_AA)

    # 6. Resize Handle (shown when in EDIT mode)
    if not is_locked:
        handle_x = ax + aw - 20
        handle_y = ay + ah - 20
        cv2.rectangle(canvas, (handle_x, handle_y), (ax + aw, ay + ah), (0, 165, 255), -1)
        cv2.putText(canvas, "+", (handle_x + 5, handle_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)


def draw_area_control_toolbar(canvas):
    """Draws the area configuration toolbar below the camera feed."""
    is_locked = control_area['locked']

    # 1. Lock/Edit Toggle Button
    bx, by, bw, bh = AREA_BUTTONS['LOCK_TOGGLE']
    lock_bg = (35, 75, 45) if is_locked else (80, 50, 20)
    lock_border = (0, 240, 120) if is_locked else (0, 180, 255)
    lock_txt = "[ LOCKED ]" if is_locked else "[ EDIT AREA ]"
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), lock_bg, -1)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), lock_border, 2)
    cv2.putText(canvas, lock_txt, (bx + 12, by + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Touch Delay Cycle Button
    bx, by, bw, bh = AREA_BUTTONS['DELAY_CYCLE']
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (30, 48, 65), -1)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (0, 200, 255), 1)
    cv2.putText(canvas, f"HOLD: {dwell_trigger_time:.2f}s", (bx + 14, by + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (0, 240, 255), 1, cv2.LINE_AA)

    # 3. Cycle Size Button
    bx, by, bw, bh = AREA_BUTTONS['CYCLE_SIZE']
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (36, 42, 54), -1)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (80, 95, 120), 1)
    cv2.putText(canvas, f"SIZE: {control_area['w']}px", (bx + 12, by + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 220, 240), 1, cv2.LINE_AA)

    # 4. Center Button
    bx, by, bw, bh = AREA_BUTTONS['CENTER']
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (36, 42, 54), -1)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (80, 95, 120), 1)
    cv2.putText(canvas, "CENTER", (bx + 16, by + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 240), 1, cv2.LINE_AA)

    # 5. Reset Button
    bx, by, bw, bh = AREA_BUTTONS['RESET']
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (50, 32, 38), -1)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (120, 70, 80), 1)
    cv2.putText(canvas, "RESET", (bx + 16, by + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 180, 190), 1, cv2.LINE_AA)

    # Usage Note below toolbar
    cv2.putText(canvas, f"Touch Point Delay: Hold finger for {dwell_trigger_time:.2f}s to toggle. Single-fire lock prevents repeat toggling.",
                (CAM_VIEW_X + 5, by + bh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 155, 175), 1, cv2.LINE_AA)


def draw_virtual_matrix(canvas, active_cell=None):
    """
    Renders the hardware mirror: a realistic 8x8 LED matrix PCB
    with glowing red/amber LEDs, multi-stage bloom, and coordinate labels.
    """
    # PCB Panel Card
    card_x1 = RIGHT_PANEL_X + 20
    card_y1 = RIGHT_PANEL_Y + 10
    card_x2 = RIGHT_PANEL_X + RIGHT_PANEL_W - 20
    card_y2 = RIGHT_PANEL_Y + 395  # 460

    cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), (18, 20, 26), -1)
    cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), (55, 65, 85), 2)

    # Board Header
    cv2.putText(canvas, "64-DOT HARDWARE MATRIX (8x8 MIRROR)", (card_x1 + 18, card_y1 + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 240, 255), 1, cv2.LINE_AA)

    # Active LED count pill
    lit_count = int(np.sum(grid_dots))
    pill_text = f"{lit_count} / 64 LIT"
    cv2.rectangle(canvas, (card_x2 - 125, card_y1 + 12), (card_x2 - 15, card_y1 + 34), (32, 38, 48), -1)
    cv2.rectangle(canvas, (card_x2 - 125, card_y1 + 12), (card_x2 - 15, card_y1 + 34), (0, 200, 255), 1)
    cv2.putText(canvas, pill_text, (card_x2 - 114, card_y1 + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 200), 1, cv2.LINE_AA)

    # Column Numbers (0-7)
    for c in range(8):
        cx = MATRIX_ORIGIN_X + c * DOT_SPACING
        cv2.putText(canvas, f"C{c}", (cx - 8, MATRIX_ORIGIN_Y - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 140, 165), 1, cv2.LINE_AA)

    # Row Numbers (0-7)
    for r in range(8):
        cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
        cv2.putText(canvas, f"R{r}", (MATRIX_ORIGIN_X - 32, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 140, 165), 1, cv2.LINE_AA)

    # Render All 64 Circular LEDs
    for r in range(8):
        for c in range(8):
            cx = MATRIX_ORIGIN_X + c * DOT_SPACING
            cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
            is_lit = (grid_dots[r, c] == 1)
            is_hovered = (active_cell == (r, c))

            if is_lit:
                # Multi-stage glow effect (ambient halo -> core -> hot specular glint)
                cv2.circle(canvas, (cx, cy), DOT_RADIUS + 7, (0, 0, 160), -1)        # Outer red halo
                cv2.circle(canvas, (cx, cy), DOT_RADIUS + 3, (20, 40, 230), -1)      # Mid glow
                cv2.circle(canvas, (cx, cy), DOT_RADIUS, (40, 80, 255), -1)          # Bright core
                cv2.circle(canvas, (cx, cy), DOT_RADIUS - 4, (120, 170, 255), -1)    # Hot center
                cv2.circle(canvas, (cx - 3, cy - 3), 3, (230, 240, 255), -1)        # Specular glint
            else:
                # Dark matte lens with subtle cathode ring
                cv2.circle(canvas, (cx, cy), DOT_RADIUS, (30, 26, 36), -1)
                cv2.circle(canvas, (cx, cy), DOT_RADIUS, (65, 55, 75), 1)
                cv2.circle(canvas, (cx, cy), 3, (48, 42, 56), -1)

            # Hover Targeting Ring (from hand or mouse)
            if is_hovered:
                cv2.circle(canvas, (cx, cy), DOT_RADIUS + 8, (0, 255, 255), 2)
                cv2.circle(canvas, (cx, cy), DOT_RADIUS + 11, (0, 180, 255), 1)

    # Click Ripple Animation
    global click_ripple_anim
    if click_ripple_anim is not None:
        rx, ry, r_time = click_ripple_anim
        elapsed = time.time() - r_time
        if elapsed <= 0.35:
            radius = int(DOT_RADIUS + elapsed * 70)
            alpha = max(0, int(255 * (1.0 - elapsed / 0.35)))
            cv2.circle(canvas, (rx, ry), radius, (0, alpha, 255), 2)
        else:
            click_ripple_anim = None


def draw_brightness_slider(canvas, brightness, is_dragging):
    """Draws the sleek horizontal brightness control slider."""
    # Label & Value
    cv2.putText(canvas, "BRIGHTNESS", (SLIDER_X, SLIDER_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{brightness}%", (SLIDER_X + SLIDER_W + 15, SLIDER_Y + 15),
                cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    # Background Track
    cv2.rectangle(canvas, (SLIDER_X, SLIDER_Y), (SLIDER_X + SLIDER_W, SLIDER_Y + SLIDER_H), (32, 36, 46), -1)
    track_border = (0, 240, 255) if is_dragging else (65, 75, 95)
    cv2.rectangle(canvas, (SLIDER_X, SLIDER_Y), (SLIDER_X + SLIDER_W, SLIDER_Y + SLIDER_H), track_border, 1)

    # Filled Bar (Glowing gradient look)
    fill_w = int(SLIDER_W * (brightness / 100.0))
    fill_color = (0, 220, 255) if is_dragging else (0, 160, 220)
    cv2.rectangle(canvas, (SLIDER_X + 2, SLIDER_Y + 2), (SLIDER_X + fill_w, SLIDER_Y + SLIDER_H - 2), fill_color, -1)

    # Thumb Knob
    knob_x = SLIDER_X + fill_w
    knob_y = SLIDER_Y + SLIDER_H // 2
    cv2.circle(canvas, (knob_x, knob_y), 9, (255, 255, 255), -1)
    cv2.circle(canvas, (knob_x, knob_y), 9, (0, 180, 255), 2)


def draw_preset_buttons(canvas):
    """Draws quick hardware pattern action buttons."""
    for key, (bx, by, bw, bh, label, bg_col, text_col) in BUTTONS.items():
        is_hover = is_inside_rect(mouse_pos[0], mouse_pos[1], (bx, by, bw, bh))
        current_bg = tuple(min(255, c + 25) for c in bg_col) if is_hover else bg_col
        border_col = (0, 240, 255) if is_hover else tuple(min(255, c + 40) for c in bg_col)

        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), current_bg, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_col, 2 if is_hover else 1)
        cv2.putText(canvas, label, (bx + 18, by + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_col, 1, cv2.LINE_AA)


def draw_telemetry_bar(canvas, active_cell, last_cmd, cmd_status):
    """Draws the bottom telemetry status card."""
    bar_y = 660
    bar_h = 50
    cv2.rectangle(canvas, (0, bar_y), (CANVAS_W, bar_y + bar_h), (18, 20, 26), -1)
    cv2.line(canvas, (0, bar_y), (CANVAS_W, bar_y), (45, 55, 75), 1)

    # Area status & touch delay
    area_status = f"AREA: {control_area['w']}x{control_area['h']} [{'LOCKED' if control_area['locked'] else 'EDIT'}] | DELAY: {dwell_trigger_time:.2f}s"
    cv2.putText(canvas, area_status, (25, bar_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

    # Active Target
    target_str = f"TARGET: Row {active_cell[0]}, Col {active_cell[1]} [Dot #{active_cell[0]*8 + active_cell[1] + 1}]" if active_cell else "TARGET: None (Out of area)"
    target_col = (0, 255, 120) if active_cell else (130, 140, 160)
    cv2.putText(canvas, target_str, (370, bar_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, target_col, 1, cv2.LINE_AA)

    # Last Command & Comm Status
    comm_str = f"LAST TX: {last_cmd} | {cmd_status}"
    cv2.putText(canvas, comm_str, (710, bar_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 60), 1, cv2.LINE_AA)

    # Shortcuts Tip
    cv2.putText(canvas, "[Q] Quit  [C] Clear  [H] Heart  [D] Delay", (1030, bar_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 160, 180), 1, cv2.LINE_AA)


# ==============================================================================
# 7. MAIN EXECUTION LOOP
# ==============================================================================
def main():
    global dwell_dot, dwell_start_time, last_air_click_time, last_pinch_state
    global last_gesture_cmd_time, current_brightness, click_ripple_anim
    global dwell_lockout_dot, dwell_preset_idx, dwell_trigger_time

    parser = argparse.ArgumentParser(description="Neural Matrix Controller: 8x8 Hand Gesture Interface")
    parser.add_argument("--ip", type=str, default="10.194.177.102", help="ESP32-C3 Wi-Fi IP address")
    parser.add_argument("--port", type=int, default=8888, help="ESP32-C3 UDP port (default: 8888)")
    parser.add_argument("--serial", type=str, default=None, help="Optional Serial Port")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--demo", action="store_true", help="Run in simulation mode")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS limit (default: 60)")
    parser.add_argument("--dwell", type=float, default=0.75, help="Touch point dwell delay in seconds (default: 0.75)")
    args = parser.parse_args()

    TARGET_FPS = float(args.fps)
    FRAME_INTERVAL = 1.0 / TARGET_FPS
    dwell_trigger_time = float(args.dwell)

    print("\n" + "=" * 64, flush=True)
    print("  NEURAL MATRIX CONTROLLER: 8x8 HAND GESTURE INTERFACE", flush=True)
    print(f"  Target ESP32-C3 IP:   {args.ip}:{args.port}", flush=True)
    if args.serial:
        print(f"  Serial Fallback:      {args.serial}", flush=True)
    print(f"  Target Frame Rate:    {int(TARGET_FPS)} FPS", flush=True)
    print(f"  Touch Point Delay:    {dwell_trigger_time:.2f}s (single-fire anti-bounce active)", flush=True)
    print("  Dedicated Fixed Control Area: Active (isolated from accidental input)", flush=True)
    print("=" * 64 + "\n", flush=True)

    comm = MatrixCommunicator(udp_ip=args.ip, udp_port=args.port, serial_port=args.serial)
    analyzer = HandGestureAnalyzer()

    use_simulation = args.demo or not analyzer.available
    cap = None

    if not use_simulation:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[NOTE] Camera {args.camera} unavailable. Running in SIMULATION mode.", flush=True)
            use_simulation = True

    if not use_simulation:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    WINDOW_NAME = "Neural Matrix Controller"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse_event, comm)

    fps = TARGET_FPS
    sim_angle = 0.0

    print("[SYSTEM] Controller ready. Move hand inside the Fixed Control Area to interact.", flush=True)

    try:
        while True:
            frame_start_time = time.perf_counter()

            # 1. Acquire Frame (Camera or Simulation)
            if not use_simulation:
                ret, frame_raw = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                frame_cam = cv2.flip(frame_raw, 1)
                detected, gesture, tip_px_raw, thumb_px_raw, is_pinching, norm_pointer = analyzer.analyze(frame_cam)

                # Scale fingertip coordinates from raw camera resolution to canvas camera viewport
                tip_canvas = None
                if tip_px_raw is not None:
                    raw_h, raw_w = frame_cam.shape[:2]
                    scale_x = CAM_VIEW_W / float(raw_w)
                    scale_y = CAM_VIEW_H / float(raw_h)
                    tip_canvas = (
                        int(CAM_VIEW_X + tip_px_raw[0] * scale_x),
                        int(CAM_VIEW_Y + tip_px_raw[1] * scale_y)
                    )
            else:
                # Demo simulation mode: animated hand pointer
                frame_cam = np.full((CAM_VIEW_H, CAM_VIEW_W, 3), 26, dtype=np.uint8)
                sim_angle += 0.03
                ax = control_area['x']
                ay = control_area['y']
                aw = control_area['w']
                ah = control_area['h']
                sim_x = int(ax + aw * 0.5 + (aw * 0.35) * math.sin(sim_angle))
                sim_y = int(ay + ah * 0.5 + (ah * 0.30) * math.cos(sim_angle * 0.7))
                tip_canvas = (sim_x, sim_y)
                is_pinching = (int(sim_angle * 2) % 8 == 0)
                detected = True
                gesture = "PINCH CLICK" if is_pinching else "POINTING"

                # Draw pointer on simulated camera
                cam_sim_x = sim_x - CAM_VIEW_X
                cam_sim_y = sim_y - CAM_VIEW_Y
                if 0 <= cam_sim_x < CAM_VIEW_W and 0 <= cam_sim_y < CAM_VIEW_H:
                    cv2.circle(frame_cam, (cam_sim_x, cam_sim_y), 10, (0, 255, 255), -1)

            # 2. Main Canvas Assembly
            canvas = np.full((CANVAS_H, CANVAS_W, 3), 16, dtype=np.uint8)

            # 3. Hand Interaction Inside Fixed Control Area
            active_cell = None
            dwell_progress = 0.0

            if tip_canvas is not None:
                active_cell = get_cell_from_control_area(tip_canvas[0], tip_canvas[1])

            # Mouse hover fallback (on virtual matrix or control area)
            if active_cell is None:
                active_cell = get_dot_at_xy(mouse_pos[0], mouse_pos[1])
                if active_cell is None:
                    active_cell = get_cell_from_control_area(mouse_pos[0], mouse_pos[1])

            now = time.time()

            # PINCH-TO-CLICK (Instant toggle inside Fixed Control Area)
            if is_pinching and not last_pinch_state and (now - last_air_click_time >= 0.45):
                if active_cell is not None:
                    r, c = active_cell
                    grid_dots[r, c] ^= 1
                    comm.send_toggle_dot(r, c)
                    last_air_click_time = now
                    dwell_lockout_dot = active_cell  # Also lock out dwell on this dot
                    cx = MATRIX_ORIGIN_X + c * DOT_SPACING
                    cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
                    click_ripple_anim = (cx, cy, now)
                    print(f"[HAND PINCH] Toggled Dot ({r}, {c}) -> {'ON' if grid_dots[r,c] else 'OFF'}", flush=True)

            # DWELL-TO-CLICK (Hold steady for dwell_trigger_time inside Fixed Control Area)
            if active_cell is not None and not is_pinching:
                if active_cell == dwell_dot:
                    if active_cell != dwell_lockout_dot:
                        dwell_time = now - dwell_start_time
                        dwell_progress = min(1.0, dwell_time / dwell_trigger_time)
                        if dwell_time >= dwell_trigger_time and (now - last_air_click_time >= 0.50):
                            r, c = active_cell
                            grid_dots[r, c] ^= 1
                            comm.send_toggle_dot(r, c)
                            last_air_click_time = now
                            dwell_lockout_dot = active_cell  # Single-fire: won't re-toggle until finger leaves dot!
                            dwell_progress = 1.0
                            cx = MATRIX_ORIGIN_X + c * DOT_SPACING
                            cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
                            click_ripple_anim = (cx, cy, now)
                            print(f"[HAND DWELL] Toggled Dot ({r}, {c}) -> {'ON' if grid_dots[r,c] else 'OFF'} (Delay: {dwell_trigger_time:.2f}s)", flush=True)
                    else:
                        # Already toggled once while on this dot; waiting for user to leave
                        dwell_progress = 0.0
                else:
                    # Moved to a new cell: reset dwell timer and clear lockout
                    dwell_dot = active_cell
                    dwell_start_time = now
                    dwell_lockout_dot = None
                    dwell_progress = 0.0
            else:
                dwell_dot = None
                dwell_start_time = now
                dwell_lockout_dot = None
                dwell_progress = 0.0

            last_pinch_state = is_pinching

            # Predefined Gestures (Triggered outside the control area)
            if active_cell is None and (now - last_gesture_cmd_time >= 2.5):
                if gesture == "OPEN PALM":
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
                elif gesture == "PEACE SIGN":
                    comm.send_command("MESSAGE:HELLO")
                    last_gesture_cmd_time = now
                elif gesture == "FIST":
                    comm.send_command("MESSAGE:HOW YOU DOING?")
                    last_gesture_cmd_time = now

            # 4. Render UI Components
            draw_camera_viewport(canvas, frame_cam)
            draw_fixed_control_area(canvas, active_cell=active_cell, dwell_progress=dwell_progress, tip_px=tip_canvas)
            draw_area_control_toolbar(canvas)
            draw_virtual_matrix(canvas, active_cell=active_cell)
            draw_brightness_slider(canvas, current_brightness, is_dragging_brightness)
            draw_preset_buttons(canvas)
            draw_telemetry_bar(canvas, active_cell, comm.last_sent_cmd, comm.last_cmd_status)
            draw_header_bar(canvas, comm.udp_ip, comm.udp_port, fps, gesture, comm.serial_connected)

            # 5. Display Window
            cv2.imshow(WINDOW_NAME, canvas)

            # 6. Keyboard Shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("[SYSTEM] Exit requested by user.", flush=True)
                break
            elif key in (ord('c'), ord('C')):
                grid_dots.fill(0)
                comm.send_command("PATTERN:CLEAR")
            elif key in (ord('h'), ord('H')):
                comm.send_command("PATTERN:HEART")
            elif key in (ord('s'), ord('S')):
                comm.send_command("PATTERN:SMILE")
            elif key in (ord('p'), ord('P')):
                comm.send_command("ANIMATION:PULSE")
            elif key in (ord('l'), ord('L')):
                control_area['locked'] = not control_area['locked']
                print(f"[KEYBOARD] Area mode: {'LOCKED' if control_area['locked'] else 'EDIT'}", flush=True)
            elif key in (ord('d'), ord('D')):
                dwell_preset_idx = (dwell_preset_idx + 1) % len(DWELL_PRESETS)
                dwell_trigger_time = DWELL_PRESETS[dwell_preset_idx]
                print(f"[KEYBOARD] Touch delay set to: {dwell_trigger_time:.2f}s", flush=True)

            # 7. Frame Rate Limiter
            proc_time = time.perf_counter() - frame_start_time
            sleep_needed = FRAME_INTERVAL - proc_time
            if sleep_needed > 0.0005:
                time.sleep(sleep_needed)

            total_frame_dur = time.perf_counter() - frame_start_time
            instant_fps = 1.0 / total_frame_dur if total_frame_dur > 0 else TARGET_FPS
            fps = 0.90 * fps + 0.10 * min(TARGET_FPS, instant_fps)

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Controller shut down cleanly.", flush=True)


if __name__ == "__main__":
    main()
