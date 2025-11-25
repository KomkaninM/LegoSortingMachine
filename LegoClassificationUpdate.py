#!/usr/bin/env python3
"""
LEGO color detection using:
- Background subtraction
- HSV color filtering
- ROI in X (only pixels with x >= roi_x_min)
- Save 'different pixels' images

Exposed function:
    detect_lego_from_paths(bg_path, img_path, threshold, output_dir, roi_x_min=1250)

Returns:
    lego_color: str  ("red", "blue", or "other")
    bbox:       tuple (x, y, w, h) in ROI image coordinates, or None if no LEGO found
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple, Dict

# ==========================
#  COLOR RANGES (RED / BLUE ONLY)
#  HSV in OpenCV: H = [0,179], S = [0,255], V = [0,255]
# ==========================

COLOR_RANGES = {
    # Red often wraps around 0°, so split into two bands

    # Strong red (near 0°)
    "red_1": (
        (0,   120,  60),   # lower H,S,V
        (10,  255, 255)    # upper H,S,V
    ),

    # Strong red (near 180°)
    "red_2": (
        (170, 120,  60),
        (179, 255, 255)
    ),

    # Blue LEGO (typical vivid blue)
    "blue": (
        (90,  120,  60),
        (130, 255, 255)
    ),
}

# Merge sub-colors into final labels
MERGE_LABELS = {
    "red_1": "red",
    "red_2": "red",
    "blue":  "blue",
}

# Default ROI (can be overridden in function call)
ROI_X_MIN = 1250


# -----------------------------
# Helper functions
# -----------------------------
def load_image(path: str) -> np.ndarray:
    """Load image and raise clear error if missing."""
    path = path.strip().strip('"').strip("'")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def get_foreground_mask(bg_bgr: np.ndarray,
                        img_bgr: np.ndarray,
                        threshold_value: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Background subtraction -> binary mask of LEGO brick.
    Also returns the grayscale difference image so you can SEE which pixels changed.

    Returns:
        lego_mask: binary mask (255 = changed area)
        gray     : grayscale absdiff (visual 'different pixels')
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


def detect_lego_color(hsv_img: np.ndarray,
                      fg_mask: np.ndarray,
                      min_pixels: int = 100) -> Tuple[str, Dict[str, int]]:
    """
    For each color range:
        1) Create color mask in HSV
        2) Combine with foreground mask
        3) Count pixels
    Choose color with maximum pixel count.

    Returns:
        best_color: "red", "blue", or "other"
        merged_counts: pixel counts per merged color label
    """
    color_pixel_counts = {name: 0 for name in COLOR_RANGES.keys()}

    for name, (lower, upper) in COLOR_RANGES.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        # Color mask in HSV
        color_mask = cv2.inRange(hsv_img, lower, upper)
        # Combine with foreground (only LEGO area)
        combined_mask = cv2.bitwise_and(color_mask, fg_mask)

        count = cv2.countNonZero(combined_mask)
        color_pixel_counts[name] = count

    # Merge sub-colors like red_1, red_2 into single label "red"
    merged_counts: Dict[str, int] = {}
    for sub_name, count in color_pixel_counts.items():
        main_label = MERGE_LABELS.get(sub_name, sub_name)
        merged_counts[main_label] = merged_counts.get(main_label, 0) + count

    # Find color with max pixel count
    best_color = max(merged_counts, key=merged_counts.get)
    best_count = merged_counts[best_color]

    if best_count < min_pixels:
        # not enough red/blue pixels → classify as "other"
        return "other", merged_counts

    return best_color, merged_counts


def get_largest_bbox(fg_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Find bounding box of largest connected component in the foreground mask.

    Returns:
        (x, y, w, h) or None if no contour found.
    """
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


def draw_result(img_bgr: np.ndarray,
                bbox: Optional[Tuple[int, int, int, int]],
                color_name: str) -> np.ndarray:
    """Draw bounding box around LEGO and put color text."""
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text = f"Detected color: {color_name}"
    cv2.putText(img_bgr, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return img_bgr


# -----------------------------
# Public API function
# -----------------------------
def detect_lego_from_paths(
    bg_path: str,
    img_path: str,
    threshold: int,
    output_dir: str,
    roi_x_min: Optional[int] = ROI_X_MIN,
    save: bool = True,
    show: bool = False,
) -> Tuple[str, Optional[Tuple[int, int, int, int]]]:
    """
    Full pipeline callable from main.py:

    Args:
        bg_path:    path to background image
        img_path:   path to image with LEGO brick
        threshold:  int, threshold for background subtraction
        output_dir: directory to save results
        roi_x_min:  only use pixels with x >= roi_x_min (None = no ROI crop)
        save:       if True, save debug images
        show:       if True, show debug windows (requires GUI)

    Returns:
        lego_color: str ("red", "blue", or "other")
        bbox:       (x, y, w, h) of LEGO in ROI image, or None if not found
    """
    os.makedirs(output_dir, exist_ok=True)

    bg = load_image(bg_path)
    img = load_image(img_path)

    # Resize bg to img if needed
    if bg.shape[:2] != img.shape[:2]:
        bg = cv2.resize(bg, (img.shape[1], img.shape[0]))

    # --- ROI crop on X axis ---
    if roi_x_min is not None:
        print("Applying ROI crop")
        h, w = img.shape[:2]
        if roi_x_min >= w:
            raise ValueError(f"roi_x_min={roi_x_min} is outside image width={w}")
        
        # Define ROI bounds
        x_min, x_max = 0, 1700
        y_min, y_max = 0, 2100
        
        # Optional: Add bounds checking
        x_max = min(x_max, w)
        y_max = min(y_max, h)
        
        # Crop to ROIs
        img = img[y_min:y_max, x_min:x_max]
        bg = bg[y_min:y_max, x_min:x_max]

    # Foreground mask + grayscale diff (pixels that changed)
    fg_mask, diff_gray = get_foreground_mask(bg, img, threshold_value=threshold)

    # Compute bounding box of LEGO in the ROI image
    bbox = get_largest_bbox(fg_mask)

    # Convert to HSV and detect color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lego_color, _ = detect_lego_color(hsv, fg_mask)

    # Draw result image
    result_img = draw_result(img.copy(), bbox, lego_color)

    # Save images to disk
    if save:
        base = os.path.splitext(os.path.basename(img_path))[0]
        prefix = os.path.join(output_dir, base)
        cv2.imwrite(prefix + "_diff_gray.png", diff_gray)
        cv2.imwrite(prefix + "_mask.png", fg_mask)
        cv2.imwrite(prefix + "_result.png", result_img)

    # Optional: show windows
    if show:
        cv2.imshow("ROI Image", img)
        cv2.imshow("Difference Gray (pixels that changed)", diff_gray)
        cv2.imshow("Foreground Mask (binary)", fg_mask)
        cv2.imshow("Result", result_img)
        print("Press any key in image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return lego_color, bbox
