#!/usr/bin/env python3
"""
================================================================================
Project: ESHAN × MATRIX 8x8 // Cyber-Neural Interface
Script:  cv_matrix_controller.py

Design Language:
  - BOLD, HIGH-OCTANE CYBER-NEURAL AESTHETIC.
  - Deep obsidian void with electric cyan, hot magenta, and solar amber neon accents.
  - Supercharged multi-layer Gaussian LED bloom & holographic reticles.
  - Tactical Air-Pad with glowing alignment grid, charge-up dwell rings, and shockwave ripples.
  - High-impact typography and punchy tactile action controls.
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
        print(f"[DOWNLOAD] Downloading Hand Landmarker model to {MODEL_PATH}...", flush=True)
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
        self.last_cmd_status = "ONLINE"
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
            self.last_cmd_status = f"UDP ERR"

        if self.ser and self.ser.is_open:
            try:
                self.ser.write(payload)
                self.last_cmd_status += " + SER"
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
                gesture_name = "PINCH"
            elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
                gesture_name = "POINTING"
            elif index_extended and middle_extended and not ring_extended and not pinky_extended:
                gesture_name = "PEACE"
            elif index_extended and middle_extended and ring_extended and pinky_extended:
                gesture_name = "PALM"
            elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                gesture_name = "FIST"
            else:
                gesture_name = "TRACKING"

            norm_pointer = (norm_pts[8][0], norm_pts[8][1])

            # Sexy Glowing Cyber-Skeleton
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame_bgr, pts[start_idx], pts[end_idx], (255, 200, 0), 2, cv2.LINE_AA)
            for i, pt in enumerate(pts):
                if i in (4, 8):
                    cv2.circle(frame_bgr, pt, 5, (255, 50, 200), -1, cv2.LINE_AA)
                    cv2.circle(frame_bgr, pt, 8, (255, 230, 0), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame_bgr, pt, 3, (0, 240, 255), -1, cv2.LINE_AA)

            if is_pinching:
                pinch_mid = ((thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2)
                cv2.circle(frame_bgr, pinch_mid, 14, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, pinch_mid, 22, (255, 0, 200), 2, cv2.LINE_AA)

        return hand_detected, gesture_name, index_tip_px, thumb_tip_px, is_pinching, norm_pointer


# ==============================================================================
# 3. HIGH-OCTANE CYBER UI CONSTANTS & GEOMETRY
# ==============================================================================
CANVAS_W = 1280
CANVAS_H = 720

# Dual Panel Layout
PANEL_LEFT_X = 35
PANEL_LEFT_Y = 60
PANEL_LEFT_W = 580
PANEL_LEFT_H = 585

PANEL_RIGHT_X = 645
PANEL_RIGHT_Y = 60
PANEL_RIGHT_W = 600
PANEL_RIGHT_H = 585

# Camera Viewport inside Left Panel
CAM_VIEW_X = PANEL_LEFT_X + 16
CAM_VIEW_Y = PANEL_LEFT_Y + 46
CAM_VIEW_W = PANEL_LEFT_W - 32  # 548
CAM_VIEW_H = 435

# Dedicated Fixed Holographic Air-Pad
control_area = {
    'x': CAM_VIEW_X + (CAM_VIEW_W - 360) // 2,  # centered
    'y': CAM_VIEW_Y + (CAM_VIEW_H - 360) // 2,
    'w': 360,
    'h': 360,
    'locked': True
}

# 8x8 Dot Matrix State (1 = ON, 0 = OFF)
grid_dots = np.zeros((8, 8), dtype=np.uint8)

# 8x8 Virtual Matrix Geometry (Centered inside right console)
MATRIX_ORIGIN_X = PANEL_RIGHT_X + (PANEL_RIGHT_W - 7 * 42) // 2  # 797
MATRIX_ORIGIN_Y = PANEL_RIGHT_Y + 128
DOT_SPACING = 42
DOT_RADIUS = 14

# Glowing Neon Linear Brightness Slider
SLIDER_X = PANEL_RIGHT_X + 45
SLIDER_Y = PANEL_RIGHT_Y + 462
SLIDER_W = PANEL_RIGHT_W - 90  # 510
SLIDER_H = 10
current_brightness = 75
is_dragging_brightness = False

# Bold High-Voltage Action Buttons (4 punchy glowing controls)
BTN_W = 124
BTN_H = 36
BTN_GAP = 14
BTN_Y = PANEL_RIGHT_Y + 518
BTN_START_X = PANEL_RIGHT_X + (PANEL_RIGHT_W - (4 * BTN_W + 3 * BTN_GAP)) // 2  # centered

ACTION_BUTTONS = {
    'CLEAR':  (BTN_START_X,                         BTN_Y, BTN_W, BTN_H, "CLEAR",  (45, 45, 60),   (255, 220, 0)),
    'HEART':  (BTN_START_X + (BTN_W + BTN_GAP),     BTN_Y, BTN_W, BTN_H, "HEART",  (80, 20, 60),   (255, 40, 220)),
    'INVERT': (BTN_START_X + (BTN_W + BTN_GAP)*2,   BTN_Y, BTN_W, BTN_H, "INVERT", (20, 60, 80),   (0, 240, 255)),
    'PULSE':  (BTN_START_X + (BTN_W + BTN_GAP)*3,   BTN_Y, BTN_W, BTN_H, "PULSE",  (70, 40, 20),   (0, 180, 255))
}

# Area Customization & Touch Delay Buttons (below Camera View)
AREA_BTN_Y = CAM_VIEW_Y + CAM_VIEW_H + 18
AREA_BTN_H = 32
AREA_BUTTONS = {
    'LOCK_TOGGLE': (CAM_VIEW_X + 5,   AREA_BTN_Y, 115, AREA_BTN_H),
    'DELAY_CYCLE': (CAM_VIEW_X + 130, AREA_BTN_Y, 130, AREA_BTN_H),
    'CYCLE_SIZE':  (CAM_VIEW_X + 270, AREA_BTN_Y, 120, AREA_BTN_H),
    'RESET':       (CAM_VIEW_X + 400, AREA_BTN_Y, 95,  AREA_BTN_H)
}

# Touch Point Delay & Anti-Bounce Settings
DWELL_PRESETS = [0.50, 0.75, 1.00]
dwell_preset_idx = 1  # 0.75s default
dwell_trigger_time = DWELL_PRESETS[dwell_preset_idx]
dwell_lockout_dot = None

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
            if math.hypot(px - cx, py - cy) <= DOT_RADIUS + 8:
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
        if is_inside_rect(x, y, AREA_BUTTONS['LOCK_TOGGLE']):
            control_area['locked'] = not control_area['locked']
            print(f"[CONTROL AREA] {'LOCKED' if control_area['locked'] else 'EDIT MODE'}", flush=True)
            return

        elif is_inside_rect(x, y, AREA_BUTTONS['DELAY_CYCLE']):
            dwell_preset_idx = (dwell_preset_idx + 1) % len(DWELL_PRESETS)
            dwell_trigger_time = DWELL_PRESETS[dwell_preset_idx]
            print(f"[HOLD DELAY] Set to {dwell_trigger_time:.2f}s", flush=True)
            return

        elif is_inside_rect(x, y, AREA_BUTTONS['CYCLE_SIZE']):
            sizes = [300, 360, 420]
            curr = control_area['w']
            next_size = sizes[(sizes.index(curr) + 1) % len(sizes)] if curr in sizes else 360
            control_area['w'] = next_size
            control_area['h'] = next_size
            control_area['x'] = max(CAM_VIEW_X + 5, min(CAM_VIEW_X + CAM_VIEW_W - next_size - 5, control_area['x']))
            control_area['y'] = max(CAM_VIEW_Y + 5, min(CAM_VIEW_Y + CAM_VIEW_H - next_size - 5, control_area['y']))
            print(f"[CONTROL AREA] Size: {next_size}px", flush=True)
            return

        elif is_inside_rect(x, y, AREA_BUTTONS['RESET']):
            control_area['w'] = 360
            control_area['h'] = 360
            control_area['x'] = CAM_VIEW_X + (CAM_VIEW_W - 360) // 2
            control_area['y'] = CAM_VIEW_Y + (CAM_VIEW_H - 360) // 2
            control_area['locked'] = True
            print("[CONTROL AREA] Reset to default.", flush=True)
            return

        # 2. Area edit drag / resize
        if not control_area['locked']:
            if is_inside_rect(x, y, resize_handle_rect):
                is_resizing_area = True
                return
            elif is_inside_rect(x, y, (ax, ay, aw, ah)):
                is_dragging_area = True
                drag_offset = (x - ax, y - ay)
                return

        # 3. Click directly on 64 virtual dots
        dot = get_dot_at_xy(x, y)
        if dot is not None:
            r, c = dot
            grid_dots[r, c] ^= 1
            comm.send_toggle_dot(r, c)
            cx = MATRIX_ORIGIN_X + c * DOT_SPACING
            cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
            click_ripple_anim = (cx, cy, time.time())
            print(f"[CLICK] Dot ({r}, {c}) -> {'ON' if grid_dots[r,c] else 'OFF'}", flush=True)
            return

        # 4. Brightness Slider Click
        slider_expand = (SLIDER_X - 10, SLIDER_Y - 14, SLIDER_W + 20, SLIDER_H + 28)
        if is_inside_rect(x, y, slider_expand):
            is_dragging_brightness = True
            ratio = (x - SLIDER_X) / float(SLIDER_W)
            current_brightness = int(max(0, min(100, ratio * 100)))
            comm.send_brightness(current_brightness)
            return

        # 5. High-Voltage Action Buttons
        for key, btn in ACTION_BUTTONS.items():
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
                elif key == 'INVERT':
                    grid_dots[:] = 1 - grid_dots
                    comm.send_command("PATTERN:INVERT")
                elif key == 'PULSE':
                    comm.send_command("ANIMATION:PULSE")
                return

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging_brightness:
            ratio = (x - SLIDER_X) / float(SLIDER_W)
            current_brightness = int(max(0, min(100, ratio * 100)))
            comm.send_brightness(current_brightness)

        elif is_dragging_area and not control_area['locked']:
            new_x = x - drag_offset[0]
            new_y = y - drag_offset[1]
            control_area['x'] = max(CAM_VIEW_X + 4, min(CAM_VIEW_X + CAM_VIEW_W - control_area['w'] - 4, new_x))
            control_area['y'] = max(CAM_VIEW_Y + 4, min(CAM_VIEW_Y + CAM_VIEW_H - control_area['h'] - 4, new_y))

        elif is_resizing_area and not control_area['locked']:
            new_w = max(200, min(CAM_VIEW_W - (control_area['x'] - CAM_VIEW_X) - 4, x - control_area['x']))
            new_h = max(200, min(CAM_VIEW_H - (control_area['y'] - CAM_VIEW_Y) - 4, y - control_area['y']))
            side = min(new_w, new_h)
            control_area['w'] = side
            control_area['h'] = side

    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging_brightness = False
        is_dragging_area = False
        is_resizing_area = False


# ==============================================================================
# 6. BOLD CYBER-NEURAL RENDERING SYSTEM
# ==============================================================================
def draw_bold_header(canvas, ip, port, fps, gesture):
    """Draws an electric cyber-gradient top header."""
    # Top Bar Solid Floor
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, 50), (14, 12, 22), -1)

    # Electric Neon Dual-Tone Accent Line (Cyan -> Magenta)
    half_w = CANVAS_W // 2
    cv2.line(canvas, (0, 50), (half_w, 50), (255, 230, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, (half_w, 50), (CANVAS_W, 50), (255, 40, 220), 2, cv2.LINE_AA)

    # Title with Glowing Cyber Badge
    cv2.circle(canvas, (24, 25), 6, (255, 230, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, (24, 25), 10, (255, 50, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "ESHAN MATRIX", (42, 33), cv2.FONT_HERSHEY_DUPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "// NEURAL PRO 8x8", (215, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 255), 1, cv2.LINE_AA)

    # Gesture Dynamic Badge (Center)
    if gesture and gesture != "NONE":
        g_col = (0, 255, 255) if gesture == "PINCH" else (255, 50, 200)
        cv2.rectangle(canvas, (540, 10), (740, 40), (28, 20, 38), -1)
        cv2.rectangle(canvas, (540, 10), (740, 40), g_col, 2, cv2.LINE_AA)
        cv2.putText(canvas, f">> {gesture}", (560, 30), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (540, 10), (740, 40), (20, 18, 28), -1)
        cv2.rectangle(canvas, (540, 10), (740, 40), (60, 50, 75), 1, cv2.LINE_AA)
        cv2.putText(canvas, ">> STANDBY", (580, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 110, 135), 1, cv2.LINE_AA)

    # Telemetry Status (Right)
    cv2.putText(canvas, f"ESP32: {ip}:{port}", (870, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{fps:.0f} FPS", (1120, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, "[Q] EXIT", (1200, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 150, 175), 1, cv2.LINE_AA)


def draw_cyber_chassis(canvas, x, y, w, h, title=""):
    """Draws a deep obsidian card with glowing corner accents."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (18, 16, 26), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (48, 42, 65), 1)

    # Glowing Corner Brackets
    c_len = 16
    for cx, cy, dx, dy in [(x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)]:
        cv2.line(canvas, (cx, cy), (cx + dx * c_len, cy), (255, 200, 0), 2, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy), (cx, cy + dy * c_len), (255, 200, 0), 2, cv2.LINE_AA)

    if title:
        cv2.putText(canvas, title, (x + 20, y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 240, 120), 1, cv2.LINE_AA)


def draw_camera_feed(canvas, frame_cam):
    """Embeds the camera feed inside the left chassis with cyber-reticle corners."""
    cam_resized = cv2.resize(frame_cam, (CAM_VIEW_W, CAM_VIEW_H))
    canvas[CAM_VIEW_Y:CAM_VIEW_Y + CAM_VIEW_H, CAM_VIEW_X:CAM_VIEW_X + CAM_VIEW_W] = cam_resized

    cv2.rectangle(canvas, (CAM_VIEW_X, CAM_VIEW_Y),
                  (CAM_VIEW_X + CAM_VIEW_W, CAM_VIEW_Y + CAM_VIEW_H), (60, 50, 80), 1)


def draw_holographic_air_pad(canvas, active_cell, dwell_progress=0.0, tip_px=None):
    """
    Renders an electric holographic Air-Pad with:
      - Neon Cyan / Magenta cyber-grid overlay
      - Illuminated target cell with coordinates
      - High-voltage charge-up reticle (Cyan -> Emerald -> Gold)
      - Shockwave ripple animations
    """
    ax = control_area['x']
    ay = control_area['y']
    aw = control_area['w']
    ah = control_area['h']
    is_locked = control_area['locked']

    # Translucent cyber-tint
    overlay = canvas.copy()
    cv2.rectangle(overlay, (ax, ay), (ax + aw, ay + ah), (25, 15, 35), -1)
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

    # High-Voltage Neon Border
    pad_border = (255, 230, 0) if is_locked else (0, 160, 255)
    cv2.rectangle(canvas, (ax, ay), (ax + aw, ay + ah), pad_border, 2)

    # 8x8 Grid with Coordinates
    cw = aw / 8.0
    ch = ah / 8.0
    for i in range(1, 8):
        gx = int(ax + i * cw)
        gy = int(ay + i * ch)
        cv2.line(canvas, (gx, ay), (gx, ay + ah), (55, 45, 75), 1)
        cv2.line(canvas, (ax, gy), (ax + aw, gy), (55, 45, 75), 1)

    # Micro row/column coordinates on perimeter
    for c in range(8):
        lbl_x = int(ax + c * cw + cw / 2 - 4)
        cv2.putText(canvas, str(c), (lbl_x, ay - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 220, 0), 1, cv2.LINE_AA)
    for r in range(8):
        lbl_y = int(ay + r * ch + ch / 2 + 4)
        cv2.putText(canvas, str(r), (ax - 14, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 220, 0), 1, cv2.LINE_AA)

    # Active Target Cell Highlight
    if active_cell is not None:
        r, c = active_cell
        cx1 = int(ax + c * cw)
        cy1 = int(ay + r * ch)
        cx2 = int(cx1 + cw)
        cy2 = int(cy1 + ch)

        cell_glow = canvas.copy()
        cv2.rectangle(cell_glow, (cx1, cy1), (cx2, cy2), (255, 230, 0), -1)
        cv2.addWeighted(cell_glow, 0.38, canvas, 0.62, 0, canvas)

        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (255, 255, 255), 2)
        coord_lbl = f"[{r}:{c}]"
        cv2.putText(canvas, coord_lbl, (cx1 + 4, cy1 + 14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    # High-Voltage Charging Reticle
    if tip_px is not None and is_inside_rect(tip_px[0], tip_px[1], (ax, ay, aw, ah)):
        tx, ty = tip_px
        cv2.circle(canvas, (tx, ty), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), 12, (255, 230, 0), 2, cv2.LINE_AA)
        cv2.line(canvas, (tx - 16, ty), (tx + 16, ty), (255, 230, 0), 1, cv2.LINE_AA)
        cv2.line(canvas, (tx, ty - 16), (tx, ty + 16), (255, 230, 0), 1, cv2.LINE_AA)

        # Dynamic Multi-Stage Dwell Charge Arc
        if dwell_progress > 0.0:
            end_ang = int(dwell_progress * 360)
            if dwell_progress < 0.6:
                arc_col = (255, 230, 0)       # Electric Cyan
            elif dwell_progress < 0.95:
                arc_col = (0, 255, 120)       # Hot Emerald
            else:
                arc_col = (0, 255, 255)       # Solar Gold
            cv2.ellipse(canvas, (tx, ty), (18, 18), -90, 0, end_ang, arc_col, 3, cv2.LINE_AA)

    # Tag Banner
    tag = "AIR-PAD // LOCKED" if is_locked else "AIR-PAD // DRAG / RESIZE"
    cv2.rectangle(canvas, (ax, ay - 24), (ax + 175, ay), pad_border, -1)
    cv2.putText(canvas, tag, (ax + 8, ay - 7), cv2.FONT_HERSHEY_DUPLEX, 0.36, (15, 12, 22), 1, cv2.LINE_AA)

    if not is_locked:
        hx = ax + aw - 22
        hy = ay + ah - 22
        cv2.rectangle(canvas, (hx, hy), (ax + aw, ay + ah), (0, 180, 255), -1)
        cv2.putText(canvas, "+", (hx + 5, hy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2)


def draw_cyber_button(canvas, rect, label, is_hovered=False, active_color=(0, 240, 255), is_active=False):
    """Draws a bold, punchy cyber button with glowing border and hover intensity."""
    bx, by, bw, bh = rect
    bg = (38, 32, 52) if is_hovered else (24, 20, 34)
    if is_active:
        bg = (50, 40, 70)
    border = active_color if is_hovered else tuple(int(c * 0.7) for c in active_color)

    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), bg, -1)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border, 2 if is_hovered else 1)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
    tx = bx + (bw - tw) // 2
    ty = by + (bh + th) // 2
    txt_col = (255, 255, 255) if is_hovered else (220, 225, 235)
    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.44, txt_col, 1, cv2.LINE_AA)


def draw_virtual_matrix(canvas, active_cell=None):
    """
    Renders the supercharged 8x8 hardware matrix
    with rich cinematic Gaussian bloom and brilliant core illumination.
    """
    lit_count = int(np.sum(grid_dots))
    badge_str = f"{lit_count} / 64 LIT"
    cv2.rectangle(canvas, (PANEL_RIGHT_X + PANEL_RIGHT_W - 130, PANEL_RIGHT_Y + 12),
                  (PANEL_RIGHT_X + PANEL_RIGHT_W - 20, PANEL_RIGHT_Y + 36), (36, 24, 48), -1)
    cv2.rectangle(canvas, (PANEL_RIGHT_X + PANEL_RIGHT_W - 130, PANEL_RIGHT_Y + 12),
                  (PANEL_RIGHT_X + PANEL_RIGHT_W - 20, PANEL_RIGHT_Y + 36), (255, 40, 220), 1)
    cv2.putText(canvas, badge_str, (PANEL_RIGHT_X + PANEL_RIGHT_W - 118, PANEL_RIGHT_Y + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 120, 230), 1, cv2.LINE_AA)

    # Column Numbers
    for c in range(8):
        cx = MATRIX_ORIGIN_X + c * DOT_SPACING
        cv2.putText(canvas, f"C{c}", (cx - 8, MATRIX_ORIGIN_Y - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 130, 160), 1, cv2.LINE_AA)

    # Row Numbers
    for r in range(8):
        cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
        cv2.putText(canvas, f"R{r}", (MATRIX_ORIGIN_X - 32, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 130, 160), 1, cv2.LINE_AA)

    # Render All 64 Glowing LEDs
    for r in range(8):
        for c in range(8):
            cx = MATRIX_ORIGIN_X + c * DOT_SPACING
            cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
            is_lit = (grid_dots[r, c] == 1)
            is_hovered = (active_cell == (r, c))

            if is_lit:
                # Supercharged multi-layer Gaussian bloom (Crimson -> Fiery Red -> Hot Gold -> White Core)
                cv2.circle(canvas, (cx, cy), DOT_RADIUS + 10, (15, 20, 180), -1, cv2.LINE_AA)    # Outer atmospheric aura
                cv2.circle(canvas, (cx, cy), DOT_RADIUS + 5, (20, 60, 255), -1, cv2.LINE_AA)     # Radiant bloom
                cv2.circle(canvas, (cx, cy), DOT_RADIUS, (40, 140, 255), -1, cv2.LINE_AA)        # Vibrant fiery body
                cv2.circle(canvas, (cx, cy), DOT_RADIUS - 5, (160, 220, 255), -1, cv2.LINE_AA)   # Hot solar core
                cv2.circle(canvas, (cx - 3, cy - 3), 3, (255, 255, 255), -1, cv2.LINE_AA)        # Diamond reflection
            else:
                # Sleek glossy dark lens
                cv2.circle(canvas, (cx, cy), DOT_RADIUS, (30, 26, 38), -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), DOT_RADIUS, (70, 60, 90), 1, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 3, (50, 42, 62), -1, cv2.LINE_AA)

            # High-Impact Hover Reticle
            if is_hovered:
                cv2.circle(canvas, (cx, cy), DOT_RADIUS + 8, (255, 230, 0), 2, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), DOT_RADIUS + 12, (255, 40, 220), 1, cv2.LINE_AA)

    # Shockwave Ripple Animation
    global click_ripple_anim
    if click_ripple_anim is not None:
        rx, ry, r_time = click_ripple_anim
        elapsed = time.time() - r_time
        if elapsed <= 0.35:
            radius = int(DOT_RADIUS + elapsed * 75)
            alpha = max(0, int(255 * (1.0 - elapsed / 0.35)))
            cv2.circle(canvas, (rx, ry), radius, (alpha, alpha, 255), 2, cv2.LINE_AA)
        else:
            click_ripple_anim = None


def draw_neon_brightness(canvas, brightness, is_dragging):
    """Draws a glowing neon linear slider with dynamic gradient track."""
    cv2.putText(canvas, "BRIGHTNESS INTENSITY", (SLIDER_X, SLIDER_Y - 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 240, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{brightness}%", (SLIDER_X + SLIDER_W - 40, SLIDER_Y - 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # Base Track
    cv2.rectangle(canvas, (SLIDER_X, SLIDER_Y), (SLIDER_X + SLIDER_W, SLIDER_Y + SLIDER_H), (34, 28, 46), -1)
    cv2.rectangle(canvas, (SLIDER_X, SLIDER_Y), (SLIDER_X + SLIDER_W, SLIDER_Y + SLIDER_H), (70, 60, 95), 1)

    # Glowing Gradient Fill
    fill_w = int(SLIDER_W * (brightness / 100.0))
    fill_col = (255, 230, 0) if is_dragging else (255, 180, 0)
    cv2.rectangle(canvas, (SLIDER_X, SLIDER_Y), (SLIDER_X + fill_w, SLIDER_Y + SLIDER_H), fill_col, -1)

    # Radiant Knob
    knob_x = SLIDER_X + fill_w
    knob_y = SLIDER_Y + SLIDER_H // 2
    cv2.circle(canvas, (knob_x, knob_y), 9, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (knob_x, knob_y), 13, (255, 50, 200), 2, cv2.LINE_AA)


def draw_bottom_telemetry(canvas, active_cell, last_cmd, cmd_status):
    """Draws the high-contrast bottom status HUD."""
    bar_y = 665
    cv2.rectangle(canvas, (0, bar_y), (CANVAS_W, CANVAS_H), (14, 12, 22), -1)
    cv2.line(canvas, (0, bar_y), (CANVAS_W, bar_y), (48, 40, 65), 1)

    # Target
    if active_cell:
        r, c = active_cell
        target_str = f"TARGET >> ROW {r} : COL {c} [DOT #{r*8 + c + 1:02d}]"
        t_col = (0, 255, 120)
    else:
        target_str = "TARGET >> AIR-PAD READY"
        t_col = (130, 120, 150)
    cv2.putText(canvas, target_str, (40, bar_y + 32), cv2.FONT_HERSHEY_DUPLEX, 0.44, t_col, 1, cv2.LINE_AA)

    # Last Command Status
    disp_str = f"DISPATCH: {cmd_status} // {last_cmd}"
    cv2.putText(canvas, disp_str, (520, bar_y + 32), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 200, 0), 1, cv2.LINE_AA)

    # Shortcut Hints
    keys_str = "[C] CLEAR  [H] HEART  [I] INVERT  [P] PULSE  [L] LOCK"
    cv2.putText(canvas, keys_str, (880, bar_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 140, 170), 1, cv2.LINE_AA)


# ==============================================================================
# 7. MAIN EXECUTION LOOP
# ==============================================================================
def main():
    global dwell_dot, dwell_start_time, last_air_click_time, last_pinch_state
    global current_brightness, click_ripple_anim
    global dwell_lockout_dot, dwell_preset_idx, dwell_trigger_time

    parser = argparse.ArgumentParser(description="ESHAN MATRIX // Cyber-Neural Interface")
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
    print("  ESHAN MATRIX 8x8 // BOLD CYBER-NEURAL CONTROLLER", flush=True)
    print(f"  Target ESP32-C3 IP:   {args.ip}:{args.port}", flush=True)
    if args.serial:
        print(f"  Serial Fallback:      {args.serial}", flush=True)
    print(f"  Touch Point Delay:    {dwell_trigger_time:.2f}s", flush=True)
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

    WINDOW_NAME = "ESHAN MATRIX // Cyber-Neural Interface"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse_event, comm)

    fps = TARGET_FPS
    sim_angle = 0.0

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
                frame_cam = np.full((CAM_VIEW_H, CAM_VIEW_W, 3), 22, dtype=np.uint8)
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
                gesture = "PINCH" if is_pinching else "POINTING"

                cam_sim_x = sim_x - CAM_VIEW_X
                cam_sim_y = sim_y - CAM_VIEW_Y
                if 0 <= cam_sim_x < CAM_VIEW_W and 0 <= cam_sim_y < CAM_VIEW_H:
                    cv2.circle(frame_cam, (cam_sim_x, cam_sim_y), 8, (255, 230, 0), -1, cv2.LINE_AA)

            # 2. Main Canvas (Deep Midnight Void)
            canvas = np.full((CANVAS_H, CANVAS_W, 3), 12, dtype=np.uint8)

            # 3. Hand Interaction Inside Fixed Control Area
            active_cell = None
            dwell_progress = 0.0

            if tip_canvas is not None:
                active_cell = get_cell_from_control_area(tip_canvas[0], tip_canvas[1])

            # Mouse hover fallback
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
                    dwell_lockout_dot = active_cell
                    cx = MATRIX_ORIGIN_X + c * DOT_SPACING
                    cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
                    click_ripple_anim = (cx, cy, now)
                    print(f"[HAND PINCH] Dot ({r}, {c}) -> {'ON' if grid_dots[r,c] else 'OFF'}", flush=True)

            # DWELL-TO-CLICK (Hold steady for dwell_trigger_time with single-fire anti-bounce)
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
                            dwell_lockout_dot = active_cell
                            dwell_progress = 1.0
                            cx = MATRIX_ORIGIN_X + c * DOT_SPACING
                            cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
                            click_ripple_anim = (cx, cy, now)
                            print(f"[HAND DWELL] Dot ({r}, {c}) -> {'ON' if grid_dots[r,c] else 'OFF'}", flush=True)
                    else:
                        dwell_progress = 0.0
                else:
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

            # 4. Render UI Components
            draw_cyber_chassis(canvas, PANEL_LEFT_X, PANEL_LEFT_Y, PANEL_LEFT_W, PANEL_LEFT_H, "HOLOGRAPHIC AIR-PAD")
            draw_cyber_chassis(canvas, PANEL_RIGHT_X, PANEL_RIGHT_Y, PANEL_RIGHT_W, PANEL_RIGHT_H, "8x8 HARDWARE MATRIX")

            # Left Viewport & Holographic Pad
            draw_camera_feed(canvas, frame_cam)
            draw_holographic_air_pad(canvas, active_cell=active_cell, dwell_progress=dwell_progress, tip_px=tip_canvas)

            # Left Toolbar Cyber Buttons
            is_locked = control_area['locked']
            lock_txt = "[ LOCKED ]" if is_locked else "[ EDITING ]"
            lock_col = (0, 255, 120) if is_locked else (0, 160, 255)
            draw_cyber_button(canvas, AREA_BUTTONS['LOCK_TOGGLE'], lock_txt,
                              is_hovered=is_inside_rect(mouse_pos[0], mouse_pos[1], AREA_BUTTONS['LOCK_TOGGLE']),
                              active_color=lock_col, is_active=is_locked)
            draw_cyber_button(canvas, AREA_BUTTONS['DELAY_CYCLE'], f"HOLD {dwell_trigger_time:.2f}s",
                              is_hovered=is_inside_rect(mouse_pos[0], mouse_pos[1], AREA_BUTTONS['DELAY_CYCLE']),
                              active_color=(255, 230, 0))
            draw_cyber_button(canvas, AREA_BUTTONS['CYCLE_SIZE'], f"{control_area['w']}px",
                              is_hovered=is_inside_rect(mouse_pos[0], mouse_pos[1], AREA_BUTTONS['CYCLE_SIZE']),
                              active_color=(255, 40, 220))
            draw_cyber_button(canvas, AREA_BUTTONS['RESET'], "RESET",
                              is_hovered=is_inside_rect(mouse_pos[0], mouse_pos[1], AREA_BUTTONS['RESET']),
                              active_color=(0, 200, 255))

            # Right Panel: Glowing 8x8 Matrix, Neon Slider, Punchy Action Buttons
            draw_virtual_matrix(canvas, active_cell=active_cell)
            draw_neon_brightness(canvas, current_brightness, is_dragging_brightness)

            for key, btn in ACTION_BUTTONS.items():
                rect = (btn[0], btn[1], btn[2], btn[3])
                draw_cyber_button(canvas, rect, btn[4],
                                  is_hovered=is_inside_rect(mouse_pos[0], mouse_pos[1], rect),
                                  active_color=btn[6])

            # Bold Header & Telemetry
            draw_bold_header(canvas, comm.udp_ip, comm.udp_port, fps, gesture)
            draw_bottom_telemetry(canvas, active_cell, comm.last_sent_cmd, comm.last_cmd_status)

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
            elif key in (ord('i'), ord('I')):
                grid_dots[:] = 1 - grid_dots
                comm.send_command("PATTERN:INVERT")
            elif key in (ord('p'), ord('P')):
                comm.send_command("ANIMATION:PULSE")
            elif key in (ord('l'), ord('L')):
                control_area['locked'] = not control_area['locked']
            elif key in (ord('d'), ord('D')):
                dwell_preset_idx = (dwell_preset_idx + 1) % len(DWELL_PRESETS)
                dwell_trigger_time = DWELL_PRESETS[dwell_preset_idx]

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
