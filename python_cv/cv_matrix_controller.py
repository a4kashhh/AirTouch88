#!/usr/bin/env python3
"""
================================================================================
Project: AirTouch-88: Hand Gesture Controlled 8x8 LED Matrix (64 Dots)
Script:  cv_matrix_controller.py

Key Hand Control Features:
  - 8x8 Matrix (64 Dots) controlled entirely by HAND GESTURES:
      1. Point in Air: Index finger points to any of the 64 dots (Row 0-7, Col 0-7)
      2. Pinch-to-Click: Pinch thumb & index finger together to TOGGLE any dot
      3. Dwell-to-Click: Hold finger over a dot for 0.35s to activate it
      4. Mouse Backup: You can also click any dot directly with your mouse
  - Top bar with numbers removed for a clean, immersive interface
  - Large, high-visibility 64-dot visual grid with active LED glow effects
  - BMW-style vertical height brightness control
  - Predefined gesture triggers: Open Palm (Heart), Peace Sign ("HELLO"), Fist ("HOW YOU DOING?")
  - Dual communication: Wi-Fi UDP (port 8888) + USB Serial (115200)
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

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("[ERROR] MediaPipe is not installed. Run: pip install mediapipe opencv-python numpy pyserial")
    sys.exit(1)

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
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[DOWNLOAD] Model downloaded successfully.", flush=True)


# ==============================================================================
# 1. COMMUNICATION CLIENT (UDP & SERIAL)
# ==============================================================================
class MatrixCommunicator:
    """Manages low-latency UDP packet transmission and optional USB Serial fallback."""
    def __init__(self, udp_ip="192.168.1.100", udp_port=8888, serial_port=None, baud_rate=115200):
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

        self.ser = None
        if serial_port and SERIAL_AVAILABLE:
            try:
                self.ser = serial.Serial(serial_port, baud_rate, timeout=0.1)
                print(f"[SERIAL] Connected to {serial_port} at {baud_rate} baud.", flush=True)
            except Exception as e:
                print(f"[SERIAL WARNING] Could not open {serial_port}: {e}", flush=True)

        self.last_brightness_send_time = 0
        self.last_sent_brightness = -1
        self.last_sent_cmd = ""

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
# 2. ANTI-JITTER & HYSTERESIS FILTER
# ==============================================================================
class AntiJitterFilter:
    """Provides EMA continuous filtering."""
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.filtered_val = None

    def update_continuous(self, new_val):
        if self.filtered_val is None:
            self.filtered_val = new_val
        else:
            self.filtered_val = self.alpha * new_val + (1.0 - self.alpha) * self.filtered_val
        return self.filtered_val


# ==============================================================================
# 3. HAND GESTURE & LANDMARK ANALYZER (MEDIAPIPE TASKS)
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
        ensure_model_asset()
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

    def dist(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def analyze(self, frame_bgr):
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        result = self.detector.detect(mp_image)

        hand_detected = False
        gesture_name = "NONE"
        raw_brightness = None
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

            # Check Pinch gesture (Thumb tip to Index tip distance)
            pinch_dist_norm = self.dist(thumb_tip, index_tip) / hand_scale
            is_pinching = (pinch_dist_norm < 0.26)

            # Gesture classification
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

            # BMW Brightness (Vertical index height)
            norm_y = norm_pts[8][1]
            b_ratio = (0.85 - norm_y) / (0.85 - 0.15)
            raw_brightness = int(max(0.0, min(1.0, b_ratio)) * 100)

            # Normalized pointer coordinate for 8x8 spatial air tracking
            norm_pointer = (norm_pts[8][0], norm_pts[8][1])

            # Draw Hand Skeleton
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame_bgr, pts[start_idx], pts[end_idx], (0, 220, 255), 2)
            for pt in pts:
                cv2.circle(frame_bgr, pt, 4, (0, 0, 255), -1)

            # If pinching, highlight pinch midpoint with a bright cyan burst
            if is_pinching:
                pinch_mid = ((thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2)
                cv2.circle(frame_bgr, pinch_mid, 14, (255, 255, 0), -1)
                cv2.circle(frame_bgr, pinch_mid, 20, (0, 255, 255), 2)

        return hand_detected, gesture_name, raw_brightness, index_tip_px, thumb_tip_px, is_pinching, norm_pointer


# ==============================================================================
# 4. LARGE INTERACTIVE 64-DOT 8x8 MATRIX UI (PROMINENT RIGHT-CENTER)
# ==============================================================================
GRID_PANEL_X = 760
GRID_PANEL_Y = 40
GRID_PANEL_W = 480
GRID_PANEL_H = 550
DOT_SPACING = 52
DOT_RADIUS = 16
GRID_ORIGIN_X = GRID_PANEL_X + 58
GRID_ORIGIN_Y = GRID_PANEL_Y + 75

# 8x8 Dot Matrix State (1 = ON, 0 = OFF)
grid_dots = np.zeros((8, 8), dtype=np.uint8)

# Buttons
BTN_CLEAR_RECT = (GRID_PANEL_X + 40, GRID_PANEL_Y + 490, 180, 36)
BTN_HEART_RECT = (GRID_PANEL_X + 260, GRID_PANEL_Y + 490, 180, 36)

# Mouse hover tracking
mouse_hover_pos = (-1, -1)

# Hand Dwell / Pinch state
dwell_dot = None
dwell_start_time = 0
last_air_click_time = 0
last_pinch_state = False

def get_dot_at_xy(px, py):
    """Returns (r, c) if coordinate (px, py) falls inside any of the 64 dots."""
    for r in range(8):
        for c in range(8):
            cx = GRID_ORIGIN_X + c * DOT_SPACING
            cy = GRID_ORIGIN_Y + r * DOT_SPACING
            if math.hypot(px - cx, py - cy) <= DOT_RADIUS + 8:
                return (r, c)
    return None

def get_dot_from_spatial_air(norm_x, norm_y):
    """
    Maps mid-air finger position in the left half of the screen
    (x: 0.10 -> 0.55, y: 0.20 -> 0.80) directly to (r, c) in the 8x8 matrix!
    """
    x_min, x_max = 0.10, 0.55
    y_min, y_max = 0.20, 0.80

    if x_min <= norm_x <= x_max and y_min <= norm_y <= y_max:
        c = int(((norm_x - x_min) / (x_max - x_min)) * 8)
        r = int(((norm_y - y_min) / (y_max - y_min)) * 8)
        return (max(0, min(7, r)), max(0, min(7, c)))
    return None

def is_inside_rect(px, py, rect):
    rx, ry, rw, rh = rect
    return (rx <= px <= rx + rw) and (ry <= py <= ry + rh)


def on_mouse_event(event, x, y, flags, param):
    """Mouse click backup to toggle any dot directly."""
    global mouse_hover_pos
    comm = param
    mouse_hover_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        dot = get_dot_at_xy(x, y)
        if dot is not None:
            r, c = dot
            grid_dots[r, c] ^= 1
            comm.send_toggle_dot(r, c)
            dot_num = r * 8 + c + 1
            print(f"[MOUSE CLICK] Dot at ({r},{c}) [Dot #{dot_num}] -> State: {'ON' if grid_dots[r,c] else 'OFF'}", flush=True)

        elif is_inside_rect(x, y, BTN_CLEAR_RECT):
            grid_dots.fill(0)
            comm.send_command("PATTERN:CLEAR")
            print("[MOUSE CLICK] CLEAR ALL dots on 8x8 matrix", flush=True)

        elif is_inside_rect(x, y, BTN_HEART_RECT):
            heart_map = [
                [0,1,1,0,0,1,1,0],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [0,1,1,1,1,1,1,0],
                [0,0,1,1,1,1,0,0],
                [0,0,0,1,1,0,0,0],
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0]
            ]
            for r in range(8):
                for c in range(8):
                    grid_dots[r, c] = heart_map[r][c]
            comm.send_command("PATTERN:HEART")
            print("[MOUSE CLICK] HEART pattern applied", flush=True)


def draw_64_dot_matrix(frame, active_hover_dot=None, dwell_progress=0.0):
    """Renders the large 8x8 grid with 64 interactive dots."""
    # 1. Main Matrix Frame Card
    cv2.rectangle(frame, (GRID_PANEL_X, GRID_PANEL_Y), 
                  (GRID_PANEL_X + GRID_PANEL_W, GRID_PANEL_Y + GRID_PANEL_H), 
                  (18, 18, 22), -1)
    cv2.rectangle(frame, (GRID_PANEL_X, GRID_PANEL_Y), 
                  (GRID_PANEL_X + GRID_PANEL_W, GRID_PANEL_Y + GRID_PANEL_H), 
                  (0, 220, 255), 2)

    # Title Banner
    cv2.putText(frame, "64-DOT INTERACTIVE MATRIX (8x8)", (GRID_PANEL_X + 22, GRID_PANEL_Y + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 240, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Point at any dot & PINCH (or dwell) to turn ON/OFF", (GRID_PANEL_X + 22, GRID_PANEL_Y + 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # 2. Render all 64 Circular Dots
    for r in range(8):
        for c in range(8):
            cx = GRID_ORIGIN_X + c * DOT_SPACING
            cy = GRID_ORIGIN_Y + r * DOT_SPACING
            is_lit = (grid_dots[r, c] == 1)
            is_hovered = (active_hover_dot == (r, c))

            if is_lit:
                # Active glowing red LED
                cv2.circle(frame, (cx, cy), DOT_RADIUS + 6, (0, 0, 180), -1)       # Ambient glow
                cv2.circle(frame, (cx, cy), DOT_RADIUS, (20, 30, 255), -1)         # Core
                cv2.circle(frame, (cx - 3, cy - 3), 4, (200, 220, 255), -1)        # Specular glint
                cv2.circle(frame, (cx, cy), DOT_RADIUS + 1, (100, 180, 255), 1)
            else:
                # Unlit dark dot
                cv2.circle(frame, (cx, cy), DOT_RADIUS, (38, 30, 46), -1)          # Dark base
                cv2.circle(frame, (cx, cy), DOT_RADIUS, (85, 70, 105), 1)          # Border
                cv2.circle(frame, (cx, cy), 3, (60, 50, 75), -1)

            # If hovered by hand or mouse
            if is_hovered:
                # Target crosshair / golden halo ring
                cv2.circle(frame, (cx, cy), DOT_RADIUS + 8, (0, 255, 255), 2)
                # Dwell progress arc
                if dwell_progress > 0.0:
                    end_angle = int(dwell_progress * 360)
                    cv2.ellipse(frame, (cx, cy), (DOT_RADIUS + 10, DOT_RADIUS + 10), -90, 0, end_angle, (0, 255, 0), 3)
                # Coordinate badge
                cv2.putText(frame, f"({r},{c})", (cx - 18, cy - DOT_RADIUS - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

    # 3. Quick Action Buttons
    # [CLEAR ALL]
    bx, by, bw, bh = BTN_CLEAR_RECT
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (45, 45, 55), -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (130, 130, 150), 1)
    cv2.putText(frame, "CLEAR MATRIX", (bx + 26, by + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # [HEART SHAPE]
    bx, by, bw, bh = BTN_HEART_RECT
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (70, 20, 45), -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (220, 60, 130), 1)
    cv2.putText(frame, "HEART PATTERN", (bx + 20, by + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 190), 1)


# ==============================================================================
# 5. SPATIAL AIR-TRACKING BOUNDING BOX & HUD
# ==============================================================================
def draw_hud(frame, brightness, gesture, fps, current_pattern, air_dot_target=None):
    h, w, _ = frame.shape

    # 1. Mid-Air Pointing Interaction Box on Left Camera Feed
    box_x1 = int(w * 0.08)
    box_x2 = int(w * 0.52)
    box_y1 = int(h * 0.18)
    box_y2 = int(h * 0.82)

    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (60, 60, 80), 1)
    cv2.putText(frame, "MID-AIR 8x8 GESTURE ZONE", (box_x1 + 10, box_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Move finger here to target dots | Pinch to click", (box_x1 + 10, box_y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)

    # 2. Vertical Brightness Gauge
    bar_x = int(w * 0.56)
    bar_y_top = 100
    bar_y_bottom = h - 140
    bar_h = bar_y_bottom - bar_y_top
    fill_h = int(bar_h * (brightness / 100.0))

    cv2.rectangle(frame, (bar_x, bar_y_top), (bar_x + 24, bar_y_bottom), (35, 35, 35), -1)
    cv2.rectangle(frame, (bar_x, bar_y_top), (bar_x + 24, bar_y_bottom), (120, 120, 120), 2)
    cv2.rectangle(frame, (bar_x + 2, bar_y_bottom - fill_h), (bar_x + 22, bar_y_bottom - 2), (0, 165, 255), -1)

    cv2.putText(frame, f"{brightness}%", (bar_x - 48, bar_y_bottom - fill_h + 5),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 220, 255), 1)
    cv2.putText(frame, "BRIGHTNESS", (bar_x - 35, bar_y_top - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # 3. Bottom-Left Dashboard Card
    card_w, card_h = 320, 150
    card_x, card_y = 20, h - card_h - 20
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 80, 80), 2)

    cv2.putText(frame, "AirTouch-88 STATUS", (card_x + 12, card_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
    cv2.putText(frame, f"Gesture:    {gesture}", (card_x + 12, card_y + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    target_str = f"Row {air_dot_target[0]}, Col {air_dot_target[1]}" if air_dot_target else "--"
    cv2.putText(frame, f"Target Dot: {target_str}", (card_x + 12, card_y + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)
    cv2.putText(frame, f"Brightness: {brightness}%", (card_x + 12, card_y + 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1)
    cv2.putText(frame, f"Display:    {current_pattern}", (card_x + 12, card_y + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 50), 1)

    # 4. FPS Counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


# ==============================================================================
# 6. MAIN EXECUTION LOOP
# ==============================================================================
def main():
    global dwell_dot, dwell_start_time, last_air_click_time, last_pinch_state

    parser = argparse.ArgumentParser(description="AirTouch-88: Hand Gesture Controlled 8x8 LED Matrix")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="ESP32-C3 Wi-Fi IP address")
    parser.add_argument("--port", type=int, default=8888, help="ESP32-C3 UDP port (default: 8888)")
    parser.add_argument("--serial", type=str, default=None, help="Optional Serial Port")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--demo", action="store_true", help="Run in simulation mode")
    args = parser.parse_args()

    print("\n============================================================", flush=True)
    print("  AIRTOUCH-88: HAND GESTURE CONTROLLED 8x8 MATRIX (64 DOTS)", flush=True)
    print(f"  Target ESP32-C3 IP:   {args.ip}:{args.port}", flush=True)
    if args.serial:
        print(f"  Serial Fallback:      {args.serial}", flush=True)
    print("  CONTROL DOTS WITH HANDS: Point & Pinch to Toggle Any Dot!", flush=True)
    print("============================================================\n", flush=True)

    comm = MatrixCommunicator(udp_ip=args.ip, udp_port=args.port, serial_port=args.serial)
    analyzer = HandGestureAnalyzer()
    jitter_filter = AntiJitterFilter(alpha=0.25)

    use_simulation = args.demo
    cap = None

    if not use_simulation:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[NOTE] Camera {args.camera} unavailable. Running in DEMO mode.", flush=True)
            use_simulation = True

    if not use_simulation:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_pattern_str = "INTERACTIVE 64 DOTS"
    last_gesture_cmd_time = 0

    fps = 30.0
    frame_count = 0
    start_time = time.time()

    WINDOW_NAME = "AirTouch-88: Hand Controlled 8x8 Matrix"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse_event, comm)

    sim_angle = 0.0

    print("[SYSTEM] Camera active. You can now control the 64 dots with your hands.", flush=True)

    try:
        while True:
            if not use_simulation:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                frame = cv2.flip(frame, 1)
                detected, gesture, raw_b, tip_px, thumb_px, is_pinching, norm_pointer = analyzer.analyze(frame)
            else:
                frame = np.full((720, 1280, 3), 28, dtype=np.uint8)
                sim_angle += 0.05
                sim_x = int(350 + 200 * math.sin(sim_angle))
                sim_y = int(360 + 160 * math.cos(sim_angle * 0.8))
                tip_px = (sim_x, sim_y)
                thumb_px = (sim_x - 15, sim_y + 15)
                is_pinching = (int(sim_angle) % 3 == 0)
                norm_pointer = (sim_x / 1280.0, sim_y / 720.0)
                raw_b = int(max(0, min(100, int((720 - sim_y) / 720 * 100))))
                detected = True
                gesture = "PINCH CLICK" if is_pinching else "POINTING"
                cv2.circle(frame, (sim_x, sim_y), 16, (0, 220, 255), -1)

            # Continuous brightness smoothing
            if raw_b is not None:
                smoothed_b = int(jitter_filter.update_continuous(raw_b))
                comm.send_brightness(smoothed_b)
            else:
                smoothed_b = int(jitter_filter.filtered_val) if jitter_filter.filtered_val is not None else 75

            # --- HAND INTERACTION WITH THE 64 DOTS ---
            active_hover_dot = None
            dwell_progress = 0.0

            if detected and norm_pointer is not None:
                # Mode 1: Direct screen reach into the 8x8 grid panel
                direct_screen_dot = None
                if tip_px is not None:
                    direct_screen_dot = get_dot_at_xy(tip_px[0], tip_px[1])

                # Mode 2: Mid-air gesture zone mapping
                spatial_air_dot = get_dot_from_spatial_air(norm_pointer[0], norm_pointer[1])

                # Prefer direct screen dot if touching panel, else use spatial air mapping
                active_hover_dot = direct_screen_dot if direct_screen_dot else spatial_air_dot

                # ACTION A: Pinch-to-Click (Immediate trigger on pinch)
                now = time.time()
                if is_pinching and not last_pinch_state and (now - last_air_click_time >= 0.45):
                    if active_hover_dot is not None:
                        r, c = active_hover_dot
                        grid_dots[r, c] ^= 1
                        comm.send_toggle_dot(r, c)
                        last_air_click_time = now
                        dot_num = r * 8 + c + 1
                        print(f"[HAND PINCH CLICK] Toggled Dot at ({r},{c}) [Dot #{dot_num}] -> {'ON' if grid_dots[r,c] else 'OFF'}", flush=True)

                # ACTION B: Dwell-to-Click (Hover steady for 0.35s)
                if active_hover_dot is not None:
                    if active_hover_dot == dwell_dot:
                        dwell_time = now - dwell_start_time
                        dwell_progress = min(1.0, dwell_time / 0.35)
                        if dwell_time >= 0.35 and (now - last_air_click_time >= 0.60):
                            r, c = active_hover_dot
                            grid_dots[r, c] ^= 1
                            comm.send_toggle_dot(r, c)
                            last_air_click_time = now
                            dwell_start_time = now + 0.5  # debounce
                            print(f"[HAND DWELL CLICK] Toggled Dot at ({r},{c})", flush=True)
                    else:
                        dwell_dot = active_hover_dot
                        dwell_start_time = now
                        dwell_progress = 0.0
                else:
                    dwell_dot = None
                    dwell_progress = 0.0

                last_pinch_state = is_pinching

            # Check if mouse is hovering over any dot (if no hand hover)
            if active_hover_dot is None:
                active_hover_dot = get_dot_at_xy(mouse_hover_pos[0], mouse_hover_pos[1])

            # Predefined Gestures (Heart, Hello, HowYouDoing) with cooldown
            now = time.time()
            if now - last_gesture_cmd_time >= 2.5:
                if gesture == "OPEN PALM":
                    comm.send_command("PATTERN:HEART")
                    current_pattern_str = "HEART PATTERN"
                    last_gesture_cmd_time = now
                elif gesture == "PEACE SIGN":
                    comm.send_command("MESSAGE:HELLO")
                    current_pattern_str = "SCROLL: HELLO"
                    last_gesture_cmd_time = now
                elif gesture == "FIST":
                    comm.send_command("MESSAGE:HOW YOU DOING?")
                    current_pattern_str = "SCROLL: HOW YOU DOING?"
                    last_gesture_cmd_time = now

            frame_count += 1
            if frame_count % 15 == 0:
                elapsed = time.time() - start_time
                fps = 15.0 / elapsed if elapsed > 0 else 30.0
                start_time = time.time()

            # Render Clean HUD (No top numbers bar)
            draw_hud(frame, smoothed_b, gesture, fps, current_pattern_str, air_dot_target=active_hover_dot)

            # Render Large Interactive 8x8 (64 Dots) Matrix
            draw_64_dot_matrix(frame, active_hover_dot=active_hover_dot, dwell_progress=dwell_progress)

            # Fingertip visual pointer
            if tip_px is not None:
                cv2.circle(frame, tip_px, 12, (0, 255, 255), -1)
                cv2.circle(frame, tip_px, 16, (0, 180, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[SYSTEM] Exit requested.", flush=True)
                break
            elif key == ord('h') or key == ord('H'):
                comm.send_command("PATTERN:HEART")
                current_pattern_str = "HEART"
            elif key == ord('c') or key == ord('C'):
                grid_dots.fill(0)
                comm.send_command("PATTERN:CLEAR")
                current_pattern_str = "CLEARED"
            elif key == ord('m') or key == ord('M'):
                comm.send_command("MESSAGE:HELLO")
                current_pattern_str = "SCROLL: HELLO"
            elif key == ord('d') or key == ord('D'):
                comm.send_command("MESSAGE:HOW YOU DOING?")
                current_pattern_str = "SCROLL: HOW YOU DOING?"

            time.sleep(0.02)

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Controller shut down cleanly.", flush=True)


if __name__ == "__main__":
    main()
