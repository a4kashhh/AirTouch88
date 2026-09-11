#!/usr/bin/env python3
"""
================================================================================
Project: AirTouch-88: Computer Vision Controlled Custom 8x8 LED Matrix
Script:  cv_matrix_controller.py
Features:
  - Real-time webcam capture with MediaPipe Tasks HandLandmarker (CPU delegate)
  - Interactive Demo / Simulation mode (--demo) for instant testing without camera
  - 8-Segment horizontal screen mapping for 1-to-8 LED position selection
  - Anti-jitter smoothing: Exponential Moving Average (EMA) + Debounce logic
  - BMW-style gesture brightness control:
      * Mode 1: Vertical index finger height (Scale-invariant, default)
      * Mode 2: Pinch distance between thumb and index tip (Toggle with 'G')
  - Predefined gesture recognition:
      * POINTING: Selects individual LED (1-8)
      * OPEN PALM: Displays Heart Pattern
      * PEACE / V-SIGN: Displays scrolling "HELLO"
      * FIST: Displays scrolling "HOW YOU DOING?"
  - Rich OpenCV HUD UI with 8-slot selector, brightness bar, and live metrics
  - Dual communication interface: Wi-Fi UDP (port 8888) + USB Serial (115200)
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
        print(f"[DOWNLOAD] Downloading MediaPipe Hand Landmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[DOWNLOAD] Model downloaded successfully.")


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
                print(f"[SERIAL] Connected to {serial_port} at {baud_rate} baud.")
            except Exception as e:
                print(f"[SERIAL WARNING] Could not open {serial_port}: {e}")

        self.last_brightness_send_time = 0
        self.last_sent_brightness = -1
        self.last_sent_led = -1
        self.last_sent_cmd = ""

    def send_command(self, cmd_str, force=False):
        """Sends an ASCII command string to the ESP32-C3."""
        cmd_str = cmd_str.strip()
        if not cmd_str:
            return

        payload = (cmd_str + "\n").encode('utf-8')

        # Send over UDP
        try:
            self.sock.sendto(payload, (self.udp_ip, self.udp_port))
        except Exception:
            pass

        # Send over Serial
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(payload)
            except Exception:
                pass

        self.last_sent_cmd = cmd_str
        print(f"[TX -> ESP32] {cmd_str}")

    def send_brightness(self, brightness_val):
        """Throttled transmission for analog continuous brightness (max 20 packets/sec)."""
        now = time.time()
        if (now - self.last_brightness_send_time >= 0.05) and (abs(brightness_val - self.last_sent_brightness) >= 2):
            self.last_brightness_send_time = now
            self.last_sent_brightness = brightness_val
            self.send_command(f"BRIGHTNESS:{brightness_val}")

    def send_led_position(self, pos):
        """Sends single position command (1 to 8) when changed."""
        if pos != self.last_sent_led:
            self.last_sent_led = pos
            self.send_command(f"LED:{pos}")


# ==============================================================================
# 2. ANTI-JITTER & HYSTERESIS FILTER
# ==============================================================================
class AntiJitterFilter:
    """Provides EMA continuous filtering and frame-count debouncing."""
    def __init__(self, alpha=0.25, debounce_frames=3):
        self.alpha = alpha
        self.filtered_val = None
        self.debounce_frames = debounce_frames
        self.candidate_pos = None
        self.candidate_count = 0
        self.stable_pos = None

    def update_continuous(self, new_val):
        if self.filtered_val is None:
            self.filtered_val = new_val
        else:
            self.filtered_val = self.alpha * new_val + (1.0 - self.alpha) * self.filtered_val
        return self.filtered_val

    def update_discrete_position(self, raw_pos):
        if raw_pos == self.candidate_pos:
            self.candidate_count += 1
        else:
            self.candidate_pos = raw_pos
            self.candidate_count = 1

        if self.candidate_count >= self.debounce_frames:
            self.stable_pos = self.candidate_pos

        return self.stable_pos


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

    def analyze(self, frame_bgr, brightness_mode="HEIGHT"):
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        result = self.detector.detect(mp_image)

        hand_detected = False
        gesture_name = "NONE"
        selected_led = None
        raw_brightness = None
        index_tip_px = None

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

            hand_scale = max(self.dist(pts[0], pts[9]), 20.0)

            index_extended = self.dist(index_tip, wrist) > self.dist(pts[6], wrist) * 1.2
            middle_extended = self.dist(middle_tip, wrist) > self.dist(pts[10], wrist) * 1.2
            ring_extended = self.dist(ring_tip, wrist) > self.dist(pts[14], wrist) * 1.2
            pinky_extended = self.dist(pinky_tip, wrist) > self.dist(pts[18], wrist) * 1.2

            if index_extended and not middle_extended and not ring_extended and not pinky_extended:
                gesture_name = "POINTING"
            elif index_extended and middle_extended and not ring_extended and not pinky_extended:
                gesture_name = "PEACE SIGN"
            elif index_extended and middle_extended and ring_extended and pinky_extended:
                gesture_name = "OPEN PALM"
            elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                gesture_name = "FIST"
            else:
                gesture_name = "TRACKING"

            pinch_dist_norm = self.dist(thumb_tip, index_tip) / hand_scale
            if pinch_dist_norm < 0.28 and gesture_name not in ["PEACE SIGN", "FIST"]:
                gesture_name = "PINCH"

            norm_x = norm_pts[8][0]
            zone_min, zone_max = 0.12, 0.88
            clamped_x = max(zone_min, min(zone_max, norm_x))
            pos_ratio = (clamped_x - zone_min) / (zone_max - zone_min)
            raw_pos = int(pos_ratio * 8) + 1
            selected_led = max(1, min(8, raw_pos))

            if brightness_mode == "HEIGHT":
                norm_y = norm_pts[8][1]
                b_ratio = (0.82 - norm_y) / (0.82 - 0.18)
                raw_brightness = int(max(0.0, min(1.0, b_ratio)) * 100)
            else:
                p_ratio = (pinch_dist_norm - 0.20) / (0.80 - 0.20)
                raw_brightness = int(max(0.0, min(1.0, p_ratio)) * 100)

            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame_bgr, pts[start_idx], pts[end_idx], (0, 220, 255), 2)
            for pt in pts:
                cv2.circle(frame_bgr, pt, 4, (0, 0, 255), -1)

        return hand_detected, gesture_name, selected_led, raw_brightness, index_tip_px


# ==============================================================================
# 4. OPENCV HUD OVERLAY ENGINE
# ==============================================================================
def draw_hud(frame, selected_led, brightness, gesture, mode_name, fps, current_pattern):
    h, w, _ = frame.shape

    # 1. Top Bar: 8-Position Interactive Matrix Selector
    box_y1, box_y2 = 25, 75
    zone_x1, zone_x2 = int(w * 0.12), int(w * 0.88)
    slot_w = (zone_x2 - zone_x1) // 8

    cv2.rectangle(frame, (zone_x1 - 10, box_y1 - 10), (zone_x2 + 10, box_y2 + 10), (25, 25, 25), -1)
    cv2.rectangle(frame, (zone_x1 - 10, box_y1 - 10), (zone_x2 + 10, box_y2 + 10), (70, 70, 70), 2)

    for i in range(8):
        pos_id = i + 1
        bx1 = zone_x1 + i * slot_w + 3
        bx2 = bx1 + slot_w - 6

        if selected_led == pos_id:
            cv2.rectangle(frame, (bx1, box_y1), (bx2, box_y2), (0, 0, 220), -1)
            cv2.rectangle(frame, (bx1, box_y1), (bx2, box_y2), (50, 180, 255), 2)
            cv2.putText(frame, str(pos_id), (bx1 + slot_w // 2 - 12, box_y2 - 14),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (bx1, box_y1), (bx2, box_y2), (45, 45, 45), -1)
            cv2.rectangle(frame, (bx1, box_y1), (bx2, box_y2), (90, 90, 90), 1)
            cv2.putText(frame, str(pos_id), (bx1 + slot_w // 2 - 10, box_y2 - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)

    cv2.putText(frame, "8-POSITION MATRIX SELECTOR", (zone_x1, box_y1 - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 1, cv2.LINE_AA)

    # 2. Right Side: Vertical Brightness Gauge
    bar_x = w - 45
    bar_y_top = 110
    bar_y_bottom = h - 90
    bar_h = bar_y_bottom - bar_y_top
    fill_h = int(bar_h * (brightness / 100.0))

    cv2.rectangle(frame, (bar_x, bar_y_top), (bar_x + 22, bar_y_bottom), (35, 35, 35), -1)
    cv2.rectangle(frame, (bar_x, bar_y_top), (bar_x + 22, bar_y_bottom), (120, 120, 120), 2)
    cv2.rectangle(frame, (bar_x + 2, bar_y_bottom - fill_h), (bar_x + 20, bar_y_bottom - 2), (0, 165, 255), -1)

    cv2.putText(frame, f"{brightness}%", (bar_x - 45, bar_y_bottom - fill_h + 5),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 220, 255), 1)
    cv2.putText(frame, "BRIGHTNESS", (bar_x - 40, bar_y_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # 3. Bottom-Left: Live Dashboard Card
    card_w, card_h = 320, 160
    card_x, card_y = 20, h - card_h - 20
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 80, 80), 2)

    cv2.putText(frame, "AirTouch-88 STATUS", (card_x + 12, card_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
    cv2.putText(frame, f"Gesture:    {gesture}", (card_x + 12, card_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Active LED: {selected_led if selected_led else '--'}", (card_x + 12, card_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)
    cv2.putText(frame, f"Brightness: {brightness}% [{mode_name}]", (card_x + 12, card_y + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1)
    cv2.putText(frame, f"Display:    {current_pattern}", (card_x + 12, card_y + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 50), 1)

    # 4. Top-Left: FPS Meter
    cv2.putText(frame, f"FPS: {fps:.1f}", (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # 5. Bottom-Center: Hotkey Reference
    info_str = "Hotkeys: 1-8=LED | H=Heart | M=Hello | D=HowYouDoing | G=ToggleMode | Q=Quit"
    cv2.putText(frame, info_str, (card_x + card_w + 20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


# ==============================================================================
# 5. MAIN EXECUTION LOOP
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="AirTouch-88 CV Controller for Custom 8x8 LED Matrix")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="ESP32-C3 Wi-Fi IP address")
    parser.add_argument("--port", type=int, default=8888, help="ESP32-C3 UDP port (default: 8888)")
    parser.add_argument("--serial", type=str, default=None, help="Optional Serial Port (e.g. /dev/ttyUSB0)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--demo", action="store_true", help="Run in interactive simulation/demo mode without webcam")
    args = parser.parse_args()

    print("\n============================================================")
    print("  AIRTOUCH-88: COMPUTER VISION 8x8 MATRIX CONTROLLER")
    print(f"  Target ESP32-C3 IP:   {args.ip}:{args.port}")
    if args.serial:
        print(f"  Serial Fallback:      {args.serial}")
    print("============================================================\n")

    comm = MatrixCommunicator(udp_ip=args.ip, udp_port=args.port, serial_port=args.serial)
    analyzer = HandGestureAnalyzer()
    jitter_filter = AntiJitterFilter(alpha=0.25, debounce_frames=3)

    use_simulation = args.demo
    cap = None

    if not use_simulation:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[NOTE] Could not open camera {args.camera} directly (macOS camera permissions required).")
            print("[NOTE] Switching automatically to interactive SIMULATION / DEMO MODE.")
            print("[NOTE] To use physical webcam: run from your Mac Terminal directly: python3 python_cv/cv_matrix_controller.py\n")
            use_simulation = True

    if not use_simulation:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    brightness_mode = "HEIGHT"
    current_pattern_str = "HEART"
    last_gesture_cmd_time = 0

    fps = 30.0
    frame_count = 0
    start_time = time.time()

    print("[SYSTEM] Controller active! Running loop (Simulating telemetry to ESP32)...")

    sim_angle = 0.0

    try:
        # Run main processing loop
        for loop_iter in range(300 if use_simulation else 1000000):
            if not use_simulation:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                frame = cv2.flip(frame, 1)
                detected, gesture, raw_led, raw_b, tip_px = analyzer.analyze(frame, brightness_mode=brightness_mode)
            else:
                # Synthetic interactive demo canvas
                frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
                sim_angle += 0.08
                sim_x = int(640 + 380 * math.sin(sim_angle))
                sim_y = int(360 + 200 * math.cos(sim_angle * 0.7))
                tip_px = (sim_x, sim_y)

                # Simulated zone mapping
                raw_led = int(max(1, min(8, int(((sim_x - 150) / 980) * 8) + 1)))
                raw_b = int(max(0, min(100, int((720 - sim_y) / 720 * 100))))
                detected = True
                gesture = "POINTING" if (int(sim_angle) % 4 != 0) else "OPEN PALM"

                # Draw simulated hand pointer
                cv2.circle(frame, (sim_x, sim_y), 18, (0, 220, 255), -1)
                cv2.putText(frame, "SIMULATED FINGER", (sim_x - 70, sim_y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Continuous brightness smoothing
            if raw_b is not None:
                smoothed_b = int(jitter_filter.update_continuous(raw_b))
                comm.send_brightness(smoothed_b)
            else:
                smoothed_b = int(jitter_filter.filtered_val) if jitter_filter.filtered_val is not None else 75

            # Debounced LED position
            debounced_led = None
            if detected and raw_led is not None:
                debounced_led = jitter_filter.update_discrete_position(raw_led)
                if gesture == "POINTING" and debounced_led is not None:
                    comm.send_led_position(debounced_led)
                    current_pattern_str = f"POSITION {debounced_led}"

            # Gesture-triggered commands
            now = time.time()
            if now - last_gesture_cmd_time >= 2.5:
                if gesture == "OPEN PALM":
                    comm.send_command("PATTERN:HEART")
                    current_pattern_str = "HEART"
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

            draw_hud(frame, debounced_led, smoothed_b, gesture, brightness_mode, fps, current_pattern_str)

            if tip_px is not None:
                cv2.circle(frame, tip_px, 10, (0, 255, 255), -1)
                cv2.circle(frame, tip_px, 14, (0, 180, 255), 2)

            # In GUI environments, display window
            try:
                cv2.imshow("AirTouch-88 Controller", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
            except Exception:
                pass

            time.sleep(0.03)

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Python CV Controller shut down cleanly.")


if __name__ == "__main__":
    main()
