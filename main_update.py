#!/usr/bin/env python3
"""
Optimized LEGO Sorter - Main Control System
Integrates TFLite classification with hardware control
"""

import RPi.GPIO as GPIO
import time
import pigpio
import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

from NETPIE import publish_to_netpie

# ============================================
# CONFIGURATION
# ============================================

# TFLite Model Paths
MODEL_PATH = "/home/pi/LegoSorter/TensorFlow/full_dataset_lego.tflite"
CLASSES_PATH = "/home/pi/LegoSorter/TensorFlow/full_data_classes.txt"
BACKGROUND_PATH = "/home/pi/Pictures/background.jpg"

# Camera Settings
CAM_WIDTH = 3280
CAM_HEIGHT = 2464

# GPIO Pin Definitions
VIB1_PIN = 19           # Vibration motor 1
VIB2_PIN = 16           # Vibration motor 2
IR_PIN = 26             # IR sensor
IN1 = 5                 # Motor direction 1
IN2 = 6                 # Motor direction 2
ENA = 12                # Motor enable (PWM)
SERVO_PIN = 13          # Servo control

# Image Processing
THRESHOLD = 30          # Background subtraction threshold
BBOX_EXPAND = 20        # Extra pixels around detected object

# Timing
SENSOR_DELAY = 0.03
STABILIZE_DELAY = 2.0

# ============================================
# GLOBAL VARIABLES
# ============================================

picam2 = None
pi = None
interpreter = None
input_details = None
output_details = None
class_names = []
background_image = None

# Counters
red_sort = 0
blue_sort = 0
yellow_sort = 0
white_sort = 0
total_sort = 0
success_sort = 0

# ============================================
# INITIALIZATION FUNCTIONS
# ============================================

def init_gpio():
    """Initialize all GPIO pins"""
    GPIO.setmode(GPIO.BCM)
    
    # Motor pins
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(ENA, GPIO.OUT)
    
    # Vibration motors
    GPIO.setup(VIB1_PIN, GPIO.OUT)
    GPIO.setup(VIB2_PIN, GPIO.OUT)
    
    # IR sensor
    GPIO.setup(IR_PIN, GPIO.IN)
    
    # Servo (using pigpio for precise control)
    global pi
    pi = pigpio.pi()
    if not pi.connected:
        print("⚠️ Warning: pigpiod not running. Run 'sudo pigpiod' first!")
    
    # Motor PWM
    pwm_motor = GPIO.PWM(ENA, 1000)  # 1 kHz
    pwm_motor.start(0)
    
    return pwm_motor

def init_camera():
    """Initialize Picamera2"""
    global picam2
    if picam2 is not None:
        return
    
    print("📷 Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT)},
        buffer_count=2
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    print("✅ Camera ready")

def load_class_names(path):
    """Load class names from text file"""
    classes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                classes.append(name)
    return classes

def init_tflite():
    """Initialize TFLite interpreter"""
    global interpreter, input_details, output_details, class_names
    
    print("🧠 Loading TFLite model...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    class_names = load_class_names(CLASSES_PATH)
    print(f"✅ Model loaded. Classes: {class_names}")

def capture_background():
    """Capture and save background image"""
    global background_image
    
    print("\n🖼️ BACKGROUND CAPTURE")
    print("Please remove all LEGOs from the tray.")
    input("Press ENTER to capture background...")
    
    frame_bgr = capture_frame()
    background_image = frame_bgr
    
    # Save background for future use
    cv2.imwrite(BACKGROUND_PATH, background_image)
    print(f"✅ Background saved to {BACKGROUND_PATH}\n")

def load_background():
    """Load existing background or capture new one"""
    global background_image
    
    if os.path.exists(BACKGROUND_PATH):
        print(f"📂 Loading background from {BACKGROUND_PATH}")
        background_image = cv2.imread(BACKGROUND_PATH)
        if background_image is not None:
            print("✅ Background loaded")
            return
    
    capture_background()

# ============================================
# CAMERA FUNCTIONS
# ============================================

def capture_frame():
    """Capture a frame from camera (returns BGR image)"""
    frame = picam2.capture_array()  # RGB
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame_bgr

def capture_photo():
    """Capture photo and save to disk"""
    save_dir = Path.home() / "Pictures"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = save_dir / f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    file_path_str = str(filename)
    
    try:
        frame_bgr = capture_frame()
        cv2.imwrite(file_path_str, frame_bgr)
        print(f"📸 Image saved: {file_path_str}")
        return file_path_str, frame_bgr
    except Exception as e:
        print(f"❌ Camera error: {e}")
        return None, None

# ============================================
# IMAGE PROCESSING FUNCTIONS
# ============================================

def get_foreground_mask(bg_bgr, img_bgr, threshold_value=30):
    """Background subtraction -> binary mask"""
    if bg_bgr.shape[:2] != img_bgr.shape[:2]:
        bg_bgr = cv2.resize(bg_bgr, (img_bgr.shape[1], img_bgr.shape[0]))
    
    diff = cv2.absdiff(img_bgr, bg_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    _, fg_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Keep only largest contour
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        lego_mask = np.zeros_like(fg_mask)
        cv2.drawContours(lego_mask, [largest], -1, 255, thickness=-1)
        return lego_mask, gray
    
    return fg_mask, gray

def get_largest_bbox(fg_mask):
    """Get bounding box of largest contour"""
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h

def crop_lego_roi(bg_bgr, img_bgr, threshold_value=30, expand=20):
    """Crop ROI around detected LEGO"""
    fg_mask, _ = get_foreground_mask(bg_bgr, img_bgr, threshold_value)
    bbox = get_largest_bbox(fg_mask)
    
    if bbox is None:
        return None, None
    
    x, y, w, h = bbox
    x = max(0, x - expand)
    y = max(0, y - expand)
    x2 = min(img_bgr.shape[1], x + w + expand)
    y2 = min(img_bgr.shape[0], y + h + expand)
    
    roi = img_bgr[y:y2, x:x2]
    return roi, bbox

# ============================================
# CLASSIFICATION FUNCTIONS
# ============================================

def run_inference(roi_bgr, target_size=(128, 128)):
    """Run TFLite inference on ROI"""
    # Resize to model input size
    img_resized = cv2.resize(roi_bgr, target_size, interpolation=cv2.INTER_AREA)
    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Prepare input (model has Rescaling layer, so keep [0,255])
    input_data = img_rgb.astype(np.float32)
    input_data = np.expand_dims(input_data, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    
    probs = output_data[0]
    pred_idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    
    return pred_idx, conf

def classify_lego(img_bgr):
    """Classify LEGO from image - returns color, size, confidence, bbox"""
    roi, bbox = crop_lego_roi(background_image, img_bgr, THRESHOLD, BBOX_EXPAND)
    
    if roi is None or roi.size == 0:
        print("⚠️ No LEGO detected")
        return None, None, 0.0, None
    
    pred_idx, conf = run_inference(roi)
    class_label = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else f"unknown_{pred_idx}"
    
    # Parse class name (e.g., "blue_1x3" -> color="blue", size="1x3")
    if "_" in class_label:
        parts = class_label.split("_", 1)
        color = parts[0]
        size = parts[1] if len(parts) > 1 else "unknown"
    else:
        color = class_label
        size = "unknown"
    
    print(f"Classification: {color} {size} (confidence: {conf:.2f})")
    
    return color, size, conf, bbox

# ============================================
# HARDWARE CONTROL FUNCTIONS
# ============================================

def set_vibration_motor(state):
    """Control vibration motors"""
    if state:
        GPIO.output(VIB1_PIN, GPIO.HIGH)
        GPIO.output(VIB2_PIN, GPIO.HIGH)
    else:
        GPIO.output(VIB1_PIN, GPIO.LOW)
        GPIO.output(VIB2_PIN, GPIO.LOW)

def is_object_detected():
    """Check if IR sensor detects object"""
    return GPIO.input(IR_PIN) == 0

def set_servo_angle(angle):
    """Set servo angle (0-180 degrees) using pigpio"""
    if not pi.connected:
        print("⚠️ pigpiod not running!")
        return
    
    # Map angle to pulse width (500-2500 µs)
    pulse_width = 500 + (angle * 2000 / 180)
    pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)

def push_lego():
    """Servo push sequence"""
    print("👋 Pushing LEGO...")
    set_servo_angle(0)
    time.sleep(0.3)
    
    # Smooth push
    for angle in range(0, 150, 10):
        set_servo_angle(angle)
        time.sleep(0.05)
    
    # Return
    for angle in range(150, -1, -10):
        set_servo_angle(angle)
        time.sleep(0.05)
    
    set_servo_angle(0)

def rotate_motor(direction, pwm_motor, duty_cycle=20, duration=0.1):
    """Rotate DC motor for sorting"""
    if direction == "forward":
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
    else:
        return
    
    pwm_motor.ChangeDutyCycle(duty_cycle)
    time.sleep(duration)
    pwm_motor.ChangeDutyCycle(0)

# ============================================
# MAIN STATE MACHINE
# ============================================

def main():
    global red_sort, blue_sort, yellow_sort, white_sort, total_sort, success_sort
    
    print("\n" + "="*50)
    print("🔷 LEGO SORTER SYSTEM - STARTING 🔷")
    print("="*50 + "\n")
    
    # Initialize all systems
    pwm_motor = init_gpio()
    init_camera()
    init_tflite()
    load_background()
    
    # Set servo to home position
    set_servo_angle(0)
    
    # State machine variables
    state = "STATE_FEED"
    color = ""
    size = ""
    path = ""
    conf = 0.0
    bbox = None
    
    print("\n🚀 System ready! Starting main loop...\n")
    
    try:
        while True:
            
            # STATE 1: FEED - Wait for LEGO
            if state == "STATE_FEED":
                print("📥 [FEED] Waiting for LEGO...")
                set_vibration_motor(True)
                time.sleep(SENSOR_DELAY)
                
                if is_object_detected():
                    set_vibration_motor(False)
                    print("✅ Object detected!")
                    state = "STATE_STABILIZE"
            
            # STATE 2: STABILIZE - Let LEGO settle
            elif state == "STATE_STABILIZE":
                print("⏳ [STABILIZE] Waiting for object to settle...")
                time.sleep(STABILIZE_DELAY)
                state = "STATE_CAPTURE"
            
            # STATE 3: CAPTURE - Take photo
            elif state == "STATE_CAPTURE":
                print("📸 [CAPTURE] Taking photo...")
                path, img_bgr = capture_photo()
                
                if path is None:
                    print("❌ Capture failed, returning to FEED")
                    state = "STATE_FEED"
                    continue
                
                state = "STATE_CLASSIFY"
            
            # STATE 4: CLASSIFY - Identify LEGO
            elif state == "STATE_CLASSIFY":
                print("🧠 [CLASSIFY] Analyzing LEGO...")
                color, size, conf, bbox = classify_lego(img_bgr)
                
                if color is None:
                    print("⚠️ No LEGO detected in image")
                    state = "STATE_FEED"
                    continue
                
                if conf < 0.6:
                    print(f"⚠️ Low confidence ({conf:.2f}), treating as uncertain")
                
                state = "STATE_ROTATE"
            
            # STATE 5: ROTATE - Position sorting chute
            elif state == "STATE_ROTATE":
                print(f"🔄 [ROTATE] Positioning for {color} {size}...")
                
                if color == "red":
                    rotate_motor("forward", pwm_motor)
                    time.sleep(0.2)
                elif color == "blue":
                    rotate_motor("backward", pwm_motor)
                    time.sleep(0.2)
                elif color == "yellow":
                    rotate_motor("forward", pwm_motor, duty_cycle=30, duration=0.15)
                    time.sleep(0.2)
                elif color == "white":
                    rotate_motor("backward", pwm_motor, duty_cycle=30, duration=0.15)
                    time.sleep(0.2)
                # Add more colors as needed
                
                state = "STATE_PUSH"
            
            # STATE 6: PUSH - Push LEGO into chute
            elif state == "STATE_PUSH":
                print("👋 [PUSH] Pushing LEGO...")
                push_lego()
                time.sleep(0.5)
                state = "STATE_PUBLISH"
            
            # STATE 7: PUBLISH - Send data to NETPIE
            elif state == "STATE_PUBLISH":
                print("📤 [PUBLISH] Sending data...")
                
                # Update counters
                total_sort += 1
                if conf >= 0.6:
                    success_sort += 1
                
                if color == "red":
                    red_sort += 1
                elif color == "blue":
                    blue_sort += 1
                elif color == "yellow":
                    yellow_sort += 1
                elif color == "white":
                    white_sort += 1
                
                counters = {
                    "total_sort": total_sort,
                    "success_sort": success_sort,
                    "red_sort": red_sort,
                    "blue_sort": blue_sort,
                    "yellow_sort": yellow_sort,
                    "white_sort": white_sort
                }
                
                bbox_size = bbox[2] * bbox[3] if bbox is not None else 0
                
                publish_to_netpie(
                    state="STATE_PUBLISH",
                    color=color,
                    image_path=path,
                    counters=counters,
                    extra={
                        "size": size,
                        "bbox_area": bbox_size,
                        "confidence": conf
                    }
                )
                
                print(f"✅ Sorted: {color} {size} | Total: {total_sort} | Success: {success_sort}\n")
                state = "STATE_FEED"
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopping system...")
    
    finally:
        # Cleanup
        set_vibration_motor(False)
        pwm_motor.stop()
        if picam2:
            picam2.stop()
        if pi:
            pi.stop()
        GPIO.cleanup()
        print("🧹 Cleanup complete. Goodbye!")

# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()