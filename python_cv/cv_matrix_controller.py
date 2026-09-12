#!/usr/bin/env python3
"""
================================================================================
Project: Studio Matrix Controller (8x8 Hardware Interface)
Script:  cv_matrix_controller.py

Design Language:
  - High-end minimalist studio hardware aesthetic (Teenage Engineering / Dieter Rams).
  - Deep matte obsidian & graphite palette with warm Gaussian LED illumination.
  - Clean typographic hierarchy and hairline layout grid.
  - Dedicated Fixed Air-Pad with 8x8 micro-guide points and single-fire touch delay.
  - Pure, focused interaction: Zero clutter, zero unnecessary text buttons.
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
            self.last_cmd_status = "TX OK"
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

            # Subtle minimalist bone drawing (hairline muted bone structure)
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame_bgr, pts[start_idx], pts[end_idx], (70, 75, 85), 1, cv2.LINE_AA)
            for i, pt in enumerate(pts):
                if i in (4, 8):
                    cv2.circle(frame_bgr, pt, 3, (240, 240, 245), -1, cv2.LINE_AA)
                else:
                    cv2.circle(frame_bgr, pt, 2, (100, 105, 115), -1, cv2.LINE_AA)

            if is_pinching:
                pinch_mid = ((thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2)
                cv2.circle(frame_bgr, pinch_mid, 8, (255, 255, 255), -1, cv2.LINE_AA)

        return hand_detected, gesture_name, index_tip_px, thumb_tip_px, is_pinching, norm_pointer


# ==============================================================================
# 3. MINIMALIST UI CONSTANTS & GEOMETRY
# ==============================================================================
CANVAS_W = 1280
CANVAS_H = 720

# Dual Panel Layout
PANEL_LEFT_X = 40
PANEL_LEFT_Y = 54
PANEL_LEFT_W = 575
PANEL_LEFT_H = 585

PANEL_RIGHT_X = 645
PANEL_RIGHT_Y = 54
PANEL_RIGHT_W = 595
PANEL_RIGHT_H = 585

# Camera Viewport inside Left Panel
CAM_VIEW_X = PANEL_LEFT_X + 18
CAM_VIEW_Y = PANEL_LEFT_Y + 48
CAM_VIEW_W = PANEL_LEFT_W - 36  # 539
CAM_VIEW_H = 430

# Dedicated Fixed Control Area ("Air-Pad")
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
MATRIX_ORIGIN_Y = PANEL_RIGHT_Y + 130
DOT_SPACING = 42
DOT_RADIUS = 13

# Minimal Linear Brightness Slider
SLIDER_X = PANEL_RIGHT_X + 50
SLIDER_Y = PANEL_RIGHT_Y + 468
SLIDER_W = PANEL_RIGHT_W - 100  # 495
SLIDER_H = 6
current_brightness = 75
is_dragging_brightness = False

# Minimalist Core Actions (Only 3 essential buttons, no clutter!)
BTN_W = 145
BTN_H = 34
BTN_Y = PANEL_RIGHT_Y + 518
BTN_START_X = PANEL_RIGHT_X + (PANEL_RIGHT_W - (3 * BTN_W + 2 * 20)) // 2  # centered

ACTION_BUTTONS = {
    'CLEAR':  (BTN_START_X,                BTN_Y, BTN_W, BTN_H, "CLEAR"),
    'HEART':  (BTN_START_X + BTN_W + 20,   BTN_Y, BTN_W, BTN_H, "HEART"),
    'INVERT': (BTN_START_X + (BTN_W + 20)*2, BTN_Y, BTN_W, BTN_H, "INVERT")
}

# Area Customization & Touch Delay Buttons (below Camera View)
AREA_BTN_Y = CAM_VIEW_Y + CAM_VIEW_H + 20  # 498
AREA_BTN_H = 30
AREA_BUTTONS = {
    'LOCK_TOGGLE': (CAM_VIEW_X + 5,   AREA_BTN_Y, 110, AREA_BTN_H),
    'DELAY_CYCLE': (CAM_VIEW_X + 125, AREA_BTN_Y, 125, AREA_BTN_H),
    'CYCLE_SIZE':  (CAM_VIEW_X + 260, AREA_BTN_Y, 120, AREA_BTN_H),
    'RESET':       (CAM_VIEW_X + 390, AREA_BTN_Y, 95,  AREA_BTN_H)
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
            if math.hypot(px - cx, py - cy) <= DOT_RADIUS + 7:
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
    resize_handle_rect = (ax + aw - 20, ay + ah - 20, 20, 20)

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

        # 5. Minimal Action Buttons
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
# 6. MINIMALIST RENDERING SYSTEM
# ==============================================================================
def draw_minimal_topbar(canvas, ip, port, fps, gesture):
    """Draws a clean, refined header without heavy blocks or garish borders."""
    # Brand
    cv2.putText(canvas, "MATRIX 8x8", (42, 33), cv2.FONT_HERSHEY_DUPLEX, 0.58, (240, 240, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, "STUDIO CONTROLLER", (168, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (110, 115, 125), 1, cv2.LINE_AA)

    # Active Gesture Status (Center-aligned subtle text)
    if gesture and gesture != "NONE":
        g_text = f"●  {gesture}"
        g_color = (255, 255, 255) if gesture == "PINCH" else (160, 210, 255)
    else:
        g_text = "○  IDLE"
        g_color = (90, 95, 105)
    cv2.putText(canvas, g_text, (580, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.42, g_color, 1, cv2.LINE_AA)

    # Telemetry Status (Right-aligned, whisper light)
    status_str = f"ESP32 · {ip}:{port}   |   {fps:.0f} FPS   |   [Q] EXIT"
    cv2.putText(canvas, status_str, (880, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 115, 125), 1, cv2.LINE_AA)


def draw_studio_card(canvas, x, y, w, h, title=""):
    """Draws a sleek matte dark chassis with subtle hairline border."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (18, 18, 22), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (34, 34, 40), 1)

    if title:
        cv2.putText(canvas, title, (x + 18, y + 26), cv2.FONT_HERSHEY_DUPLEX, 0.42, (180, 185, 195), 1, cv2.LINE_AA)


def draw_camera_feed(canvas, frame_cam):
    """Subtly desaturates and embeds the video feed inside the left chassis."""
    # Darken and desaturate slightly for high-end cinematic blend
    cam_resized = cv2.resize(frame_cam, (CAM_VIEW_W, CAM_VIEW_H))
    cam_dim = cv2.addWeighted(cam_resized, 0.85, np.zeros_like(cam_resized), 0.15, 0)
    canvas[CAM_VIEW_Y:CAM_VIEW_Y + CAM_VIEW_H, CAM_VIEW_X:CAM_VIEW_X + CAM_VIEW_W] = cam_dim

    # Hairline frame
    cv2.rectangle(canvas, (CAM_VIEW_X, CAM_VIEW_Y),
                  (CAM_VIEW_X + CAM_VIEW_W, CAM_VIEW_Y + CAM_VIEW_H), (34, 34, 40), 1)

    # Subtle corner cross marks
    m = 10
    corners = [
        (CAM_VIEW_X, CAM_VIEW_Y),
        (CAM_VIEW_X + CAM_VIEW_W, CAM_VIEW_Y),
        (CAM_VIEW_X, CAM_VIEW_Y + CAM_VIEW_H),
        (CAM_VIEW_X + CAM_VIEW_W, CAM_VIEW_Y + CAM_VIEW_H)
    ]
    for cx, cy in corners:
        dx = 1 if cx == CAM_VIEW_X else -1
        dy = 1 if cy == CAM_VIEW_Y else -1
        cv2.line(canvas, (cx, cy), (cx + dx * m, cy), (70, 75, 85), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy), (cx, cy + dy * m), (70, 75, 85), 1, cv2.LINE_AA)


def draw_fixed_air_pad(canvas, active_cell, dwell_progress=0.0, tip_px=None):
    """
    Renders an architectural-grade Air-Pad with:
      - Translucent matte dark glass surface
      - Subtle 8x8 micro-dot guide grid
      - Smooth target focus and clean progress gauge
    """
    ax = control_area['x']
    ay = control_area['y']
    aw = control_area['w']
    ah = control_area['h']
    is_locked = control_area['locked']

    # Translucent glass fill
    glass = canvas.copy()
    cv2.rectangle(glass, (ax, ay), (ax + aw, ay + ah), (14, 15, 18), -1)
    cv2.addWeighted(glass, 0.40, canvas, 0.60, 0, canvas)

    # Subtle boundary
    pad_border = (60, 65, 75) if is_locked else (200, 140, 60)
    cv2.rectangle(canvas, (ax, ay), (ax + aw, ay + ah), pad_border, 1)

    # Micro-dot 8x8 grid alignment
    cw = aw / 8.0
    ch = ah / 8.0
    for r in range(8):
        for c in range(8):
            px = int(ax + (c + 0.5) * cw)
            py = int(ay + (r + 0.5) * ch)
            cv2.circle(canvas, (px, py), 1, (50, 55, 65), -1)

    # Active hover cell highlight
    if active_cell is not None:
        r, c = active_cell
        cx1 = int(ax + c * cw)
        cy1 = int(ay + r * ch)
        cx2 = int(cx1 + cw)
        cy2 = int(cy1 + ch)

        hover_overlay = canvas.copy()
        cv2.rectangle(hover_overlay, (cx1, cy1), (cx2, cy2), (255, 255, 255), -1)
        cv2.addWeighted(hover_overlay, 0.12, canvas, 0.88, 0, canvas)
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (180, 185, 195), 1)

        # Coordinate label (minimalistic monospace)
        coord_lbl = f"{r:02d} · {c:02d}"
        cv2.putText(canvas, coord_lbl, (cx1 + 4, cy1 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 205, 215), 1, cv2.LINE_AA)

    # Fingertip Cursor & Dwell Progress Ring
    if tip_px is not None and is_inside_rect(tip_px[0], tip_px[1], (ax, ay, aw, ah)):
        tx, ty = tip_px
        # Center precision dot
        cv2.circle(canvas, (tx, ty), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), 8, (160, 165, 175), 1, cv2.LINE_AA)

        # Whisper-thin dwell progress arc
        if dwell_progress > 0.0:
            end_ang = int(dwell_progress * 360)
            cv2.ellipse(canvas, (tx, ty), (15, 15), -90, 0, end_ang, (255, 255, 255), 2, cv2.LINE_AA)

    # Pad Label
    tag = "AIR-PAD" if is_locked else "AIR-PAD (DRAG / RESIZE)"
    cv2.putText(canvas, tag, (ax + 6, ay - 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.36, (120, 125, 135), 1, cv2.LINE_AA)


def draw_minimal_button(canvas, rect, label, is_hovered=False, is_active=False):
    """Draws an ultra-clean, studio-grade button."""
    bx, by, bw, bh = rect
    bg = (30, 30, 36) if is_hovered else (22, 22, 26)
    if is_active:
        bg = (40, 42, 50)
    border = (90, 95, 105) if is_hovered else (45, 45, 52)
    txt_col = (255, 255, 255) if is_hovered else (180, 185, 195)

    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), bg, -1)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border, 1)

    # Center label
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    tx = bx + (bw - tw) // 2
    ty = by + (bh + th) // 2
    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_col, 1, cv2.LINE_AA)


def draw_virtual_matrix(canvas, active_cell=None):
    """
    Renders a stunning, physical-hardware-inspired 8x8 matrix
    with authentic Gaussian bloom and tactile micro-lenses.
    """
    # Active LED count
    lit_count = int(np.sum(grid_dots))
    count_str = f"{lit_count} / 64 ACTIVE"
    cv2.putText(canvas, count_str, (PANEL_RIGHT_X + PANEL_RIGHT_W - 125, PANEL_RIGHT_Y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130, 135, 145), 1, cv2.LINE_AA)

    # Render All 64 Circular Apertures
    for r in range(8):
        for c in range(8):
            cx = MATRIX_ORIGIN_X + c * DOT_SPACING
            cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
            is_lit = (grid_dots[r, c] == 1)
            is_hovered = (active_cell == (r, c))

            if is_lit:
                # Atmospheric multi-stage warm amber-red glow
                cv2.circle(canvas, (cx, cy), DOT_RADIUS + 9, (8, 20, 120), -1, cv2.LINE_AA)     # Ambient falloff
                cv2.circle(canvas, (cx, cy), DOT_RADIUS + 4, (15, 45, 210), -1, cv2.LINE_AA)    # Medium bloom
                cv2.circle(canvas, (cx, cy), DOT_RADIUS, (30, 85, 255), -1, cv2.LINE_AA)        # Vibrant core
                cv2.circle(canvas, (cx, cy), DOT_RADIUS - 5, (130, 180, 255), -1, cv2.LINE_AA)  # Hot center
                cv2.circle(canvas, (cx - 3, cy - 3), 2, (255, 255, 255), -1, cv2.LINE_AA)       # Specular pinhole
            else:
                # Deep recessed matte lens
                cv2.circle(canvas, (cx, cy), DOT_RADIUS, (26, 26, 30), -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), DOT_RADIUS, (42, 42, 48), 1, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 2, (38, 38, 44), -1, cv2.LINE_AA)

            # Quiet hover focus ring
            if is_hovered:
                cv2.circle(canvas, (cx, cy), DOT_RADIUS + 5, (220, 225, 235), 1, cv2.LINE_AA)

    # Click Ripple Animation
    global click_ripple_anim
    if click_ripple_anim is not None:
        rx, ry, r_time = click_ripple_anim
        elapsed = time.time() - r_time
        if elapsed <= 0.30:
            radius = int(DOT_RADIUS + elapsed * 55)
            alpha = max(0, int(220 * (1.0 - elapsed / 0.30)))
            cv2.circle(canvas, (rx, ry), radius, (alpha, alpha, alpha), 1, cv2.LINE_AA)
        else:
            click_ripple_anim = None


def draw_linear_brightness(canvas, brightness, is_dragging):
    """Draws a minimalist linear slider track."""
    # Label & Value
    cv2.putText(canvas, "INTENSITY", (SLIDER_X, SLIDER_Y - 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.36, (130, 135, 145), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{brightness}%", (SLIDER_X + SLIDER_W - 32, SLIDER_Y - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 225, 235), 1, cv2.LINE_AA)

    # 4px Rail Track
    cv2.rectangle(canvas, (SLIDER_X, SLIDER_Y), (SLIDER_X + SLIDER_W, SLIDER_Y + SLIDER_H), (30, 30, 36), -1)

    # Active Fill
    fill_w = int(SLIDER_W * (brightness / 100.0))
    cv2.rectangle(canvas, (SLIDER_X, SLIDER_Y), (SLIDER_X + fill_w, SLIDER_Y + SLIDER_H), (200, 205, 215), -1)

    # Slider Knob
    knob_x = SLIDER_X + fill_w
    knob_y = SLIDER_Y + SLIDER_H // 2
    cv2.circle(canvas, (knob_x, knob_y), 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (knob_x, knob_y), 7, (40, 40, 48), 1, cv2.LINE_AA)


def draw_bottom_bar(canvas, active_cell, last_cmd, cmd_status):
    """Draws a clean, quiet bottom telemetry footer."""
    bar_y = 668
    cv2.line(canvas, (40, bar_y), (CANVAS_W - 40, bar_y), (28, 28, 34), 1)

    # Left: Target Coordinate
    if active_cell:
        r, c = active_cell
        target_str = f"TARGET · ROW {r:02d}  COL {c:02d}  [#{r*8 + c + 1:02d}]"
        t_col = (240, 240, 245)
    else:
        target_str = "TARGET · NONE"
        t_col = (100, 105, 115)
    cv2.putText(canvas, target_str, (45, bar_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, t_col, 1, cv2.LINE_AA)

    # Center: Last Dispatch
    disp_str = f"STATUS · {cmd_status} ({last_cmd})"
    cv2.putText(canvas, disp_str, (480, bar_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 135, 145), 1, cv2.LINE_AA)

    # Right: Minimal Keyboard Shortcuts
    keys_str = "[C] CLEAR   [H] HEART   [I] INVERT   [L] LOCK PAD"
    cv2.putText(canvas, keys_str, (850, bar_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 105, 115), 1, cv2.LINE_AA)


# ==============================================================================
# 7. MAIN EXECUTION LOOP
# ==============================================================================
def main():
    global dwell_dot, dwell_start_time, last_air_click_time, last_pinch_state
    global current_brightness, click_ripple_anim
    global dwell_lockout_dot, dwell_preset_idx, dwell_trigger_time

    parser = argparse.ArgumentParser(description="Studio Matrix Controller")
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

    print("\n" + "=" * 60, flush=True)
    print("  STUDIO MATRIX CONTROLLER · 8x8 HARDWARE INTERFACE", flush=True)
    print(f"  Target ESP32-C3 IP:   {args.ip}:{args.port}", flush=True)
    if args.serial:
        print(f"  Serial Fallback:      {args.serial}", flush=True)
    print(f"  Touch Point Delay:    {dwell_trigger_time:.2f}s", flush=True)
    print("=" * 60 + "\n", flush=True)

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

    WINDOW_NAME = "Studio Matrix Controller"
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
                frame_cam = np.full((CAM_VIEW_H, CAM_VIEW_W, 3), 20, dtype=np.uint8)
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
                    cv2.circle(frame_cam, (cam_sim_x, cam_sim_y), 6, (240, 240, 245), -1, cv2.LINE_AA)

            # 2. Main Canvas (Deep Void Obsidian)
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
            # Left & Right Chassis Panels
            draw_studio_card(canvas, PANEL_LEFT_X, PANEL_LEFT_Y, PANEL_LEFT_W, PANEL_LEFT_H, "AIR-PAD SURFACE")
            draw_studio_card(canvas, PANEL_RIGHT_X, PANEL_RIGHT_Y, PANEL_RIGHT_W, PANEL_RIGHT_H, "TACTILE 8x8 ARRAY")

            # Camera Viewport & Air-Pad
            draw_camera_feed(canvas, frame_cam)
            draw_fixed_air_pad(canvas, active_cell=active_cell, dwell_progress=dwell_progress, tip_px=tip_canvas)

            # Left Toolbar Buttons
            is_locked = control_area['locked']
            lock_label = "LOCKED" if is_locked else "EDITING"
            draw_minimal_button(canvas, AREA_BUTTONS['LOCK_TOGGLE'], lock_label,
                                is_hovered=is_inside_rect(mouse_pos[0], mouse_pos[1], AREA_BUTTONS['LOCK_TOGGLE']),
                                is_active=is_locked)
            draw_minimal_button(canvas, AREA_BUTTONS['DELAY_CYCLE'], f"HOLD: {dwell_trigger_time:.2f}s",
                                is_hovered=is_inside_rect(mouse_pos[0], mouse_pos[1], AREA_BUTTONS['DELAY_CYCLE']))
            draw_minimal_button(canvas, AREA_BUTTONS['CYCLE_SIZE'], f"SIZE: {control_area['w']}px",
                                is_hovered=is_inside_rect(mouse_pos[0], mouse_pos[1], AREA_BUTTONS['CYCLE_SIZE']))
            draw_minimal_button(canvas, AREA_BUTTONS['RESET'], "RESET",
                                is_hovered=is_inside_rect(mouse_pos[0], mouse_pos[1], AREA_BUTTONS['RESET']))

            # Right Panel: 8x8 Matrix, Linear Brightness Slider, Actions
            draw_virtual_matrix(canvas, active_cell=active_cell)
            draw_linear_brightness(canvas, current_brightness, is_dragging_brightness)

            for key, btn in ACTION_BUTTONS.items():
                rect = (btn[0], btn[1], btn[2], btn[3])
                draw_minimal_button(canvas, rect, btn[4],
                                    is_hovered=is_inside_rect(mouse_pos[0], mouse_pos[1], rect))

            # Header & Footer
            draw_minimal_topbar(canvas, comm.udp_ip, comm.udp_port, fps, gesture)
            draw_bottom_bar(canvas, active_cell, comm.last_sent_cmd, comm.last_cmd_status)

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
