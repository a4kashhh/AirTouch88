#!/usr/bin/env python3
"""
================================================================================
Project: Minimal 8x8 LED Matrix Controller & Hand Gesture Interface
Script:  cv_matrix_controller.py

Features:
  - Minimalist, distraction-free UI (pure dark canvas, no camera video bloat)
  - Centered high-contrast 8x8 LED Matrix display with realistic LED glow
  - Fixed Hand Control Area: mid-air tracking zone maps 1:1 to the 64 dots
  - Touch Point Dwell Delay: deliberate 0.75s hold with single-fire anti-bounce
  - Pinch-to-click and direct mouse click support
  - Minimal controls: Brightness slider, Clear, Heart, Smile
  - UDP communication to ESP32-C3
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
# 2. HAND GESTURE & LANDMARK ANALYZER (RUNS IN BACKGROUND)
# ==============================================================================
class HandGestureAnalyzer:
    """Extracts landmarks silently in the background without drawing video feed."""
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
            return False, "NONE", False, None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)

        hand_detected = False
        gesture_name = "NONE"
        is_pinching = False
        norm_pointer = None

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand_detected = True
            lm_list = result.hand_landmarks[0]
            norm_pts = [(lm.x, lm.y) for lm in lm_list]

            wrist = norm_pts[0]
            thumb_tip = norm_pts[4]
            index_tip = norm_pts[8]
            middle_tip = norm_pts[12]
            ring_tip = norm_pts[16]
            pinky_tip = norm_pts[20]

            hand_scale = max(self.dist(wrist, norm_pts[9]), 0.05)

            index_extended = self.dist(index_tip, wrist) > self.dist(norm_pts[6], wrist) * 1.2
            middle_extended = self.dist(middle_tip, wrist) > self.dist(norm_pts[10], wrist) * 1.2
            ring_extended = self.dist(ring_tip, wrist) > self.dist(norm_pts[14], wrist) * 1.2
            pinky_extended = self.dist(pinky_tip, wrist) > self.dist(norm_pts[18], wrist) * 1.2

            pinch_dist_norm = self.dist(thumb_tip, index_tip) / hand_scale
            is_pinching = (pinch_dist_norm < 0.28)

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

            norm_pointer = (index_tip[0], index_tip[1])

        return hand_detected, gesture_name, is_pinching, norm_pointer


# ==============================================================================
# 3. MINIMAL UI GEOMETRY & CONSTANTS
# ==============================================================================
WINDOW_W = 640
WINDOW_H = 720

# Centered 8x8 Matrix Geometry
MATRIX_ORIGIN_X = 124
MATRIX_ORIGIN_Y = 125
DOT_SPACING = 56
DOT_RADIUS = 18

# Fixed Control Area (Normalized tracking zone in front of webcam: X: 0.15-0.85, Y: 0.15-0.85)
FIXED_ZONE_X_MIN = 0.15
FIXED_ZONE_X_MAX = 0.85
FIXED_ZONE_Y_MIN = 0.15
FIXED_ZONE_Y_MAX = 0.85

# 8x8 Dot Matrix State (1 = ON, 0 = OFF)
grid_dots = np.zeros((8, 8), dtype=np.uint8)

# Minimal Brightness Slider
SLIDER_X = 124
SLIDER_Y = 572
SLIDER_W = 392
SLIDER_H = 6
current_brightness = 75
is_dragging_brightness = False

# Minimal Action Buttons
BTN_W = 110
BTN_H = 34
BTN_GAP = 31
BTN_Y = 612
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
mouse_pos = (-1, -1)


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

def get_dot_from_fixed_area(norm_x, norm_y):
    """
    Maps normalized mid-air hand position inside the Fixed Control Area
    directly to (r, c) on the 8x8 matrix. Returns None if outside.
    """
    if FIXED_ZONE_X_MIN <= norm_x <= FIXED_ZONE_X_MAX and FIXED_ZONE_Y_MIN <= norm_y <= FIXED_ZONE_Y_MAX:
        rel_x = (norm_x - FIXED_ZONE_X_MIN) / (FIXED_ZONE_X_MAX - FIXED_ZONE_X_MIN)
        rel_y = (norm_y - FIXED_ZONE_Y_MIN) / (FIXED_ZONE_Y_MAX - FIXED_ZONE_Y_MIN)
        c = int(rel_x * 8)
        r = int(rel_y * 8)
        return (max(0, min(7, r)), max(0, min(7, c)))
    return None


# ==============================================================================
# 5. MOUSE INTERACTION
# ==============================================================================
def on_mouse_event(event, x, y, flags, param):
    """Handles mouse click & drag for dots, slider, and buttons."""
    global mouse_pos, current_brightness, is_dragging_brightness, click_ripple_anim

    comm = param
    mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. Click on matrix dot directly
        dot = get_dot_at_xy(x, y)
        if dot is not None:
            r, c = dot
            grid_dots[r, c] ^= 1
            comm.send_toggle_dot(r, c)
            cx = MATRIX_ORIGIN_X + c * DOT_SPACING
            cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
            click_ripple_anim = (cx, cy, time.time())
            print(f"[MOUSE] Toggled Dot ({r}, {c}) -> {'ON' if grid_dots[r,c] else 'OFF'}", flush=True)
            return

        # 2. Click on Brightness Slider
        slider_expand = (SLIDER_X - 10, SLIDER_Y - 12, SLIDER_W + 20, SLIDER_H + 24)
        if is_inside_rect(x, y, slider_expand):
            is_dragging_brightness = True
            ratio = (x - SLIDER_X) / float(SLIDER_W)
            current_brightness = int(max(0, min(100, ratio * 100)))
            comm.send_brightness(current_brightness)
            print(f"[MOUSE] Brightness set to {current_brightness}%", flush=True)
            return

        # 3. Click on Action Buttons
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

    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging_brightness = False


# ==============================================================================
# 6. MINIMAL RENDERING ENGINE
# ==============================================================================
def render_ui(canvas, active_dot, dwell_progress, ip, port, gesture):
    """Draws a clean, minimal UI focused purely on the 8x8 matrix and essentials."""
    global click_ripple_anim

    # 1. Subtle Header
    cv2.putText(canvas, "8x8 LED MATRIX", (36, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (215, 215, 220), 1, cv2.LINE_AA)

    # Status indicator (small clean green dot + IP)
    cv2.circle(canvas, (WINDOW_W - 170, 36), 4, (0, 220, 100), -1)
    cv2.putText(canvas, f"{ip}:{port}", (WINDOW_W - 156, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 135), 1, cv2.LINE_AA)

    # Subtle divider
    cv2.line(canvas, (36, 56), (WINDOW_W - 36, 56), (36, 36, 40), 1)

    # 2. Column & Row Guides (Minimal text labels)
    for c in range(8):
        cx = MATRIX_ORIGIN_X + c * DOT_SPACING
        cv2.putText(canvas, str(c), (cx - 4, MATRIX_ORIGIN_Y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (95, 95, 105), 1, cv2.LINE_AA)
    for r in range(8):
        cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
        cv2.putText(canvas, str(r), (MATRIX_ORIGIN_X - 34, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (95, 95, 105), 1, cv2.LINE_AA)

    # 3. Render 64 Circular Dots
    for r in range(8):
        for c in range(8):
            cx = MATRIX_ORIGIN_X + c * DOT_SPACING
            cy = MATRIX_ORIGIN_Y + r * DOT_SPACING
            is_lit = (grid_dots[r, c] == 1)
            is_hover = (active_dot == (r, c))

            if is_lit:
                # Beautiful realistic red LED
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

            # Hover Cursor
            if is_hover:
                cv2.circle(canvas, (cx, cy), DOT_RADIUS + 6, (220, 220, 230), 1, cv2.LINE_AA)
                # Dwell progress arc
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

    # 5. Minimal Action Buttons (Clean, monochrome text buttons)
    for key, (bx, by, bw, bh, label) in BUTTONS.items():
        is_hover = is_inside_rect(mouse_pos[0], mouse_pos[1], (bx, by, bw, bh))
        bg_col = (38, 38, 44) if is_hover else (28, 28, 32)
        border_col = (110, 110, 120) if is_hover else (55, 55, 62)

        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), bg_col, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_col, 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (bx + (bw - len(label)*9)//2, by + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 225), 1, cv2.LINE_AA)

    # 6. Minimal Footer
    cv2.line(canvas, (36, 665), (WINDOW_W - 36, 665), (32, 32, 36), 1)
    target_text = f"Dot ({active_dot[0]}, {active_dot[1]})" if active_dot else "Waiting for hand"
    status_line = f"Fixed Area Active   |   Hold Delay: {DWELL_TRIGGER_TIME:.2f}s   |   {target_text}"
    cv2.putText(canvas, status_line, (36, 692),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110, 110, 118), 1, cv2.LINE_AA)


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

    print("\n" + "=" * 54, flush=True)
    print("  MINIMAL 8x8 LED MATRIX CONTROLLER", flush=True)
    print(f"  Target ESP32:       {args.ip}:{args.port}", flush=True)
    print(f"  Touch Point Delay:  {DWELL_TRIGGER_TIME:.2f}s (single-fire anti-bounce)", flush=True)
    print(f"  Fixed Control Area: Normalized [0.15 - 0.85]", flush=True)
    print("=" * 54 + "\n", flush=True)

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

            # 1. Process Hand Landmarks in Background (no webcam image rendered to screen)
            if not use_simulation:
                ret, frame_cam = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                frame_cam = cv2.flip(frame_cam, 1)
                detected, gesture, is_pinching, norm_pointer = analyzer.analyze(frame_cam)
            else:
                sim_angle += 0.03
                sim_x = 0.50 + 0.28 * math.sin(sim_angle)
                sim_y = 0.50 + 0.24 * math.cos(sim_angle * 0.7)
                norm_pointer = (sim_x, sim_y)
                is_pinching = (int(sim_angle * 2) % 8 == 0)
                detected = True
                gesture = "PINCH" if is_pinching else "POINT"

            # 2. Pure Dark Canvas Base
            canvas = np.full((WINDOW_H, WINDOW_W, 3), 18, dtype=np.uint8)

            # 3. Map Hand from Fixed Area to Matrix Dot
            active_dot = None
            dwell_progress = 0.0

            if detected and norm_pointer is not None:
                active_dot = get_dot_from_fixed_area(norm_pointer[0], norm_pointer[1])

            # Mouse hover fallback
            if active_dot is None:
                active_dot = get_dot_at_xy(mouse_pos[0], mouse_pos[1])

            now = time.time()

            # PINCH-TO-CLICK (Instant toggle inside Fixed Area)
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

            # 4. Render Minimal UI
            render_ui(canvas, active_dot, dwell_progress, comm.udp_ip, comm.udp_port, gesture)

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
