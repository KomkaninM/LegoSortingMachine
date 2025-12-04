#!/usr/bin/env python3
"""
LEGO dataset capture script

Functions:
- Capture full frame from Raspberry Pi Camera v2.1
- Background subtraction using a stored background image
- Find largest contour (assumed LEGO), crop bounding box (ROI)
- Save cropped ROI to an output directory for training dataset
- Trigger capture via a physical pushbutton

Usage:
    Just run from VS Code / Python:
        python3 lego_dataset_capture.py

Script will ask for:
- Output directory
- Threshold
- Resolution
- Button pin
"""

import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2


# ==========================
# GPIO CONFIG (default)
# ==========================

DEFAULT_BUTTON_PIN = 17  # BCM numbering

# Fixed dataset image size (width, height)
TARGET_SIZE = (128,128)

# ==========================
# BACKGROUND SUBTRACTION HELPERS
# ==========================

def get_foreground_mask(bg_bgr: np.ndarray,
                        img_bgr: np.ndarray,
                        threshold_value: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Background subtraction -> binary mask of LEGO brick.
    Also returns the grayscale difference image.

    Returns:
        lego_mask: binary mask (255 = changed area)
        gray     : grayscale absdiff
    """
    # Ensure same size
    if bg_bgr.shape[:2] != img_bgr.shape[:2]:
        bg_bgr = cv2.resize(bg_bgr, (img_bgr.shape[1], img_bgr.shape[0]))

    # Absolute difference
    diff = cv2.absdiff(img_bgr, bg_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Binary threshold
    _, fg_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only largest contour (assume it's the LEGO brick)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Nothing found, but still return the mask + diff image
        return fg_mask, gray

    largest = max(contours, key=cv2.contourArea)
    lego_mask = np.zeros_like(fg_mask)
    cv2.drawContours(lego_mask, [largest], -1, 255, thickness=-1)

    return lego_mask, gray


def get_largest_bbox(fg_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Find bounding box of largest connected component in the foreground mask.
    Returns (x, y, w, h) or None if no contour found.
    """
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


def crop_lego_roi(bg_bgr: np.ndarray,
                  img_bgr: np.ndarray,
                  threshold_value: int = 30,
                  expand: int = 10) -> Optional[np.ndarray]:
    """
    Given a background image and a new frame, find the largest moving object (LEGO),
    get its bounding box, optionally expand it a bit, and return the cropped ROI.
    Returns None if no object is found.
    """
    fg_mask, _ = get_foreground_mask(bg_bgr, img_bgr, threshold_value=threshold_value)
    bbox = get_largest_bbox(fg_mask)
    if bbox is None:
        return None

    x, y, w, h = bbox

    # Optional: expand bbox a little
    x = max(0, x - expand)
    y = max(0, y - expand)
    x2 = min(img_bgr.shape[1], x + w + expand)
    y2 = min(img_bgr.shape[0], y + h + expand)

    roi = img_bgr[y:y2, x:x2]
    return roi


# ==========================
# CAMERA HELPERS
# ==========================

def init_camera(width: int, height: int) -> Picamera2:
    """
    Initialize Picamera2 and return the camera object.
    """
    picam2 = Picamera2()
    # You can also use create_preview_configuration if still config causes issues
    config = picam2.create_still_configuration(
        main={"size": (width, height)},
        buffer_count=2
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2.0)  # let camera auto-exposure settle
    return picam2


def capture_frame(picam2: Picamera2) -> np.ndarray:
    """
    Capture one frame as a BGR numpy array (for OpenCV).
    Picamera2 returns RGB, so we convert to BGR.
    """
    frame = picam2.capture_array()  # RGB
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame_bgr


# ==========================
# GPIO / BUTTON SETUP
# ==========================

def init_button(pin: int):
    """
    Initialize GPIO for button on given BCM pin, using internal pull-up.
    Button pressed = GPIO.LOW (connected to GND).
    """
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)


# ==========================
# MAIN
# ==========================

def main():
    print("=== LEGO Dataset Capture (Picamera2 + Button) ===")

    # ---- Ask for settings via input() ----
    # Output directory
    out_dir = "/home/pi/LegoSorter/TensorFlow/dataset/example"

    # Threshold for background subtraction
    threshold = 30

    # Resolution
    width = 3280
    height = 2464

    # Button pin
    button_pin = 4

    # ---- Prepare output dir ----
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nUsing settings:")
    print(f"  Output directory : {out_dir}")
    print(f"  Threshold        : {threshold}")
    print(f"  Resolution       : {width} x {height}")
    print(f"  Button pin (BCM) : {button_pin}\n")

    # ---- Init camera + button ----
    print("Initializing camera...")
    picam2 = init_camera(width, height)

    print(f"Initializing button on GPIO{button_pin} (BCM)...")
    init_button(button_pin)

    # ---- Capture background ----
    print("\n=== BACKGROUND CAPTURE ===")
    print("Make sure there is NO LEGO on the tray.")
    input("Press ENTER to capture background...")

    bg_bgr = capture_frame(picam2)
    print("Background captured.\n")

    # Optional: save background for debugging
    bg_path = os.path.join(out_dir, "background_debug.png")
    cv2.imwrite(bg_path, bg_bgr)
    print(f"Saved background debug image to: {bg_path}")

    # Determine starting index from existing files
    existing = [f for f in os.listdir(out_dir) if f.lower().endswith((".jpg", ".png"))]
    index = 1
    if existing:
        numbers = []
        for name in existing:
            base, _ = os.path.splitext(name)
            for part in base.split("_"):
                if part.isdigit():
                    numbers.append(int(part))
        if numbers:
            index = max(numbers) + 1

    print("\n=== CAPTURE MODE ===")
    print("Place a LEGO in the inspection area.")
    print("Each time you PRESS the button, the script will:")
    print("- Capture a frame")
    print("- Detect LEGO using background subtraction")
    print("- Crop the bounding box (ROI)")
    print("- Save ROI as lego_XXXX.jpg in the output directory")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            # Wait for button press (active LOW)
            if GPIO.input(button_pin) == GPIO.LOW:
                # simple debounce
                time.sleep(0.05)
                if GPIO.input(button_pin) == GPIO.LOW:
                    print("Button pressed → capturing frame...")
                    frame_bgr = capture_frame(picam2)
                    roi = crop_lego_roi(bg_bgr, frame_bgr, threshold_value=threshold)

                    if roi is None:
                        print("No LEGO detected (no largest contour found). Try again.")
                    else:
                        # Debug: ดูขนาด ROI ก่อน resize
                        print("ROI shape before resize:", roi.shape)  # (h, w, 3)

                        # Resize ROI to fixed size for dataset
                        roi_resized = cv2.resize(roi, TARGET_SIZE, interpolation=cv2.INTER_AREA)

                        filename = f"lego_{index:04d}.jpg"
                        out_path = os.path.join(out_dir, filename)
                        cv2.imwrite(out_path, roi_resized)
                        print(f"Saved cropped LEGO image ({TARGET_SIZE[0]}x{TARGET_SIZE[1]}): {out_path}")
                        index += 1


                    # Wait until button is released to avoid multiple triggers
                    while GPIO.input(button_pin) == GPIO.LOW:
                        time.sleep(0.05)

            # small sleep to reduce CPU usage
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        picam2.stop()
        GPIO.cleanup()
        print("Camera stopped and GPIO cleaned up.")


if __name__ == "__main__":
    main()
