# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 2026

@author: Santosh Prakash
Evaluation / Testing script for HybridSegNet on BUSI dataset
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.hybridsegnet import HybridSegNetStable
from datasets.busi_dataset import BUSIDataset
from losses.hybrid_loss import HybridLoss
from utils.metrics import dice_score, iou_score


# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
IMG_SIZE = 256

MODEL_PATH = r"C:\Users\USER\Desktop\Breast_cancer\HybridSegNet-main\Improved_model_with augumentaion\Results_B\Visionmamba_run_B\best_model.pth"
TEST_IMG_DIR = r"C:\Users\USER\Desktop\Breast_cancer\HybridSegNet-main\Improved_model_with augumentaion\data\Dataset_B\Dataset B\Fold22\test\malignant"
TEST_MASK_DIR = r"C:\Users\USER\Desktop\Breast_cancer\HybridSegNet-main\Improved_model_with augumentaion\data\Dataset_B\Dataset B\all_masks"

SAVE_PREDICTIONS = True
PRED_SAVE_DIR = "test_predictions_dataset_B1"
THRESHOLD = 0.5


# =========================
# Utility
# =========================
VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def pixel_accuracy(pred, mask):
    correct = (pred == mask).float().sum()
    total = torch.numel(pred)
    return correct / total


def precision_score(pred, mask, eps=1e-7):
    pred = pred.float()
    mask = mask.float()

    tp = (pred * mask).sum()
    fp = (pred * (1 - mask)).sum()

    return (tp + eps) / (tp + fp + eps)


def recall_score(pred, mask, eps=1e-7):
    pred = pred.float()
    mask = mask.float()

    tp = (pred * mask).sum()
    fn = ((1 - pred) * mask).sum()

    return (tp + eps) / (tp + fn + eps)


def get_stem_to_path(folder):
    files = {}
    for f in os.listdir(folder):
        if f.lower().endswith(VALID_EXTS):
            stem = os.path.splitext(f)[0]
            files[stem] = os.path.join(folder, f)
    return files


def build_matched_pairs(image_dir, mask_dir):
    image_map = get_stem_to_path(image_dir)
    mask_map = get_stem_to_path(mask_dir)

    common_keys = sorted(set(image_map.keys()) & set(mask_map.keys()))

    only_images = sorted(set(image_map.keys()) - set(mask_map.keys()))
    only_masks = sorted(set(mask_map.keys()) - set(image_map.keys()))

    if len(common_keys) == 0:
        raise ValueError(
            "No matching image-mask filename pairs found.\n"
            "Make sure both folders contain files with the same base names."
        )

    if only_images:
        print("\n[WARNING] Images without masks:")
        for k in only_images[:10]:
            print(" ", k)
        if len(only_images) > 10:
            print(f" ... and {len(only_images) - 10} more")

    if only_masks:
        print("\n[WARNING] Masks without images:")
        for k in only_masks[:10]:
            print(" ", k)
        if len(only_masks) > 10:
            print(f" ... and {len(only_masks) - 10} more")

    image_paths = [image_map[k] for k in common_keys]
    mask_paths = [mask_map[k] for k in common_keys]

    print(f"\nMatched test pairs: {len(common_keys)}")
    return image_paths, mask_paths


def align_mask_to_pred(pred, mask):
    if pred.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask, size=pred.shape[-2:], mode="nearest")
    return mask


# =========================
# Dataset / Loader
# =========================
image_paths, mask_paths = build_matched_pairs(TEST_IMG_DIR, TEST_MASK_DIR)

test_dataset = BUSIDataset(
    image_paths=image_paths,
    mask_paths=mask_paths,
    img_size=IMG_SIZE,
    train=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================
# Model / Loss
# =========================
model = HybridSegNetStable().to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()
criterion = HybridLoss()


# =========================
# Prepare Save Folder
# =========================
if SAVE_PREDICTIONS:
    ensure_dir(PRED_SAVE_DIR)


# =========================
# Evaluation Loop
# =========================
test_loss = 0.0
dice_avg = 0.0
iou_avg = 0.0
acc_avg = 0.0
prec_avg = 0.0
rec_avg = 0.0

with torch.no_grad():
    for idx, (img, mask) in enumerate(tqdm(test_loader, desc="Testing")):
        img = img.to(DEVICE, dtype=torch.float32)
        mask = mask.to(DEVICE, dtype=torch.float32)

        pred = model(img)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)

        mask = align_mask_to_pred(pred, mask)

        loss = criterion(pred, mask)
        test_loss += loss.item()

        pred_sig = torch.sigmoid(pred)
        pred_bin = (pred_sig > THRESHOLD).float()

        dice_avg += dice_score(pred_sig, mask).item()
        iou_avg += iou_score(pred_sig, mask).item()
        acc_avg += pixel_accuracy(pred_bin, mask).item()
        prec_avg += precision_score(pred_bin, mask).item()
        rec_avg += recall_score(pred_bin, mask).item()

        if SAVE_PREDICTIONS:
            save_path = os.path.join(PRED_SAVE_DIR, f"pred_{idx:03d}.png")
            prob_path = os.path.join(PRED_SAVE_DIR, f"prob_{idx:03d}.png")

            save_image(pred_bin, save_path)
            save_image(pred_sig, prob_path)


# =========================
# Final Results
# =========================
num_samples = len(test_loader)

if num_samples == 0:
    raise ValueError("No test samples found after matching image-mask pairs.")

test_loss /= num_samples
dice_avg /= num_samples
iou_avg /= num_samples
acc_avg /= num_samples
prec_avg /= num_samples
rec_avg /= num_samples

print("\n================ TEST RESULTS ================")
print(f"Test Loss      : {test_loss:.4f}")
print(f"Dice Score     : {dice_avg:.4f}")
print(f"IoU Score      : {iou_avg:.4f}")
print(f"Pixel Accuracy : {acc_avg:.4f}")
print(f"Precision      : {prec_avg:.4f}")
print(f"Recall         : {rec_avg:.4f}")
print("==============================================")