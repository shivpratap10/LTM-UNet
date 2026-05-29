# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 11:25:45 2026

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Image augmentation script

Applies:
- +90 and -90 rotation
- horizontal and vertical flip
- gaussian noise
- brightness reduction (20%)

Saves augmented images in the same folder with suffix names.
"""

import os
import cv2
import numpy as np


# =========================
# CONFIG
# =========================
IMAGE_FOLDER = r"your_image_folder_path"

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# =========================
# AUGMENTATIONS
# =========================
def rotate_image(img, angle):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == -90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Only 90 and -90 supported")


def horizontal_flip(img):
    return cv2.flip(img, 1)


def vertical_flip(img):
    return cv2.flip(img, 0)


def add_gaussian_noise(img, mean=0, std=15):
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255)
    return noisy_img.astype(np.uint8)


def reduce_brightness(img, factor=0.8):
    img = img.astype(np.float32)
    img = img * factor
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


# =========================
# MAIN AUGMENTATION
# =========================
def augment_and_save(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Skipping unreadable file: {image_path}")
        return

    base, ext = os.path.splitext(image_path)

    # 1. Rotation +90
    img_rot90 = rotate_image(img, 90)
    cv2.imwrite(base + "_rot90" + ext, img_rot90)

    # 2. Rotation -90
    img_rot_90 = rotate_image(img, -90)
    cv2.imwrite(base + "_rot-90" + ext, img_rot_90)

    # 3. Horizontal flip
    img_hflip = horizontal_flip(img)
    cv2.imwrite(base + "_hflip" + ext, img_hflip)

    # 4. Vertical flip
    img_vflip = vertical_flip(img)
    cv2.imwrite(base + "_vflip" + ext, img_vflip)

    # 5. Gaussian noise
    img_noise = add_gaussian_noise(img)
    cv2.imwrite(base + "_noise" + ext, img_noise)

    # 6. Brightness reduction
    img_dark = reduce_brightness(img)
    cv2.imwrite(base + "_dark" + ext, img_dark)


# =========================
# RUN
# =========================
def main():
    files = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(VALID_EXTS)
    ]

    print(f"Found {len(files)} images")

    for fname in files:
        path = os.path.join(IMAGE_FOLDER, fname)

        # Avoid re-augmenting already augmented images
        if any(tag in fname for tag in ["_rot90", "_rot-90", "_hflip", "_vflip", "_noise", "_dark"]):
            continue

        augment_and_save(path)

    print("Augmentation completed!")


if __name__ == "__main__":
    main()