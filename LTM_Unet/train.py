import os
import copy
import json
import random
import inspect

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models.hybridsegnet import HybridSegNetStable
from datasets.busi_dataset import BUSIDataset
from losses.hybrid_loss import HybridLoss
from utils.metrics import dice_score, iou_score



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

IMAGE_DIR = r"data\Dataset_B\augmented_images_B"
MASK_DIR = r"data\Dataset_B\augmented_masks_B"

SAVE_ROOT = "Results_B"
RUN_NAME = "Visionmamba_run_B"

IMG_SIZE = 256
NUM_WORKERS = 0
PIN_MEMORY = True if DEVICE == "cuda" else False

VAL_SIZE = 0.15

CONFIG = {
    "batch_size": 16,
    "lr": 5e-5,
    "epochs": 200,
    "weight_decay": 1e-4,
    "grad_clip": 0.5,
    "threshold": 0.4,
}

USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 100

USE_SCHEDULER = True
SCHEDULER_PATIENCE = 20
SCHEDULER_FACTOR = 0.9



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_sorted_file_list(folder, valid_exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(valid_exts)
    ]
    files.sort()
    return files


def build_image_mask_pairs(image_dir, mask_dir):
    image_paths = get_sorted_file_list(image_dir)
    mask_paths = get_sorted_file_list(mask_dir)

    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"Number of images ({len(image_paths)}) and masks ({len(mask_paths)}) do not match."
        )

    return list(zip(image_paths, mask_paths))



def pixel_accuracy(pred_bin, mask):
    correct = (pred_bin == mask).float().sum()
    total = torch.numel(pred_bin)
    return correct / total


def precision_score(pred_bin, mask, eps=1e-7):
    pred_bin = pred_bin.float()
    mask = mask.float()
    tp = (pred_bin * mask).sum()
    fp = (pred_bin * (1 - mask)).sum()
    return (tp + eps) / (tp + fp + eps)


def recall_score(pred_bin, mask, eps=1e-7):
    pred_bin = pred_bin.float()
    mask = mask.float()
    tp = (pred_bin * mask).sum()
    fn = ((1 - pred_bin) * mask).sum()
    return (tp + eps) / (tp + fn + eps)



def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def print_config(config, title="CONFIG"):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    for k, v in config.items():
        print(f"{k}: {v}")
    print("=" * 70)


def compute_loss(criterion, pred, mask, epoch):
    try:
        sig = inspect.signature(criterion.forward)
        if "epoch" in sig.parameters:
            return criterion(pred, mask, epoch=epoch)
        return criterion(pred, mask)
    except Exception:
        try:
            return criterion(pred, mask, epoch)
        except TypeError:
            return criterion(pred, mask)


def align_mask_to_pred(pred, mask):
    if pred.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask, size=pred.shape[-2:], mode="nearest")
    return mask



def run_one_epoch_train(model, loader, optimizer, criterion, epoch, grad_clip, threshold):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_acc = 0.0
    running_prec = 0.0
    running_rec = 0.0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}", leave=False)

    for step, (img, mask) in enumerate(pbar):
        img = img.to(DEVICE, dtype=torch.float32)
        mask = mask.to(DEVICE, dtype=torch.float32)

        if epoch == 0 and step == 0:
            print(f"[DEBUG] Train image dtype: {img.dtype}, shape: {tuple(img.shape)}, min: {img.min().item():.4f}, max: {img.max().item():.4f}")
            print(f"[DEBUG] Train mask  dtype: {mask.dtype}, shape: {tuple(mask.shape)}, min: {mask.min().item():.4f}, max: {mask.max().item():.4f}")

        pred = model(img)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)

        mask = align_mask_to_pred(pred, mask)
        loss = compute_loss(criterion, pred, mask, epoch)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        pred_sig = torch.sigmoid(pred)
        pred_bin = (pred_sig > threshold).float()

        batch_dice = dice_score(pred_sig, mask).item()
        batch_iou = iou_score(pred_sig, mask).item()
        batch_acc = pixel_accuracy(pred_bin, mask).item()
        batch_prec = precision_score(pred_bin, mask).item()
        batch_rec = recall_score(pred_bin, mask).item()

        running_loss += loss.item()
        running_dice += batch_dice
        running_iou += batch_iou
        running_acc += batch_acc
        running_prec += batch_prec
        running_rec += batch_rec

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{batch_dice:.4f}",
            "iou": f"{batch_iou:.4f}",
        })

    n = len(loader)
    return {
        "loss": running_loss / n,
        "dice": running_dice / n,
        "iou": running_iou / n,
        "acc": running_acc / n,
        "precision": running_prec / n,
        "recall": running_rec / n,
    }


def run_one_epoch_val(model, loader, criterion, epoch, threshold=0.5):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_acc = 0.0
    running_prec = 0.0
    running_rec = 0.0

    pred_min_all = []
    pred_max_all = []

    pbar = tqdm(loader, desc=f"Val   Epoch {epoch+1}", leave=False)

    with torch.no_grad():
        for step, (img, mask) in enumerate(pbar):
            img = img.to(DEVICE, dtype=torch.float32)
            mask = mask.to(DEVICE, dtype=torch.float32)

            if epoch == 0 and step == 0:
                print(f"[DEBUG] Val image dtype: {img.dtype}, shape: {tuple(img.shape)}, min: {img.min().item():.4f}, max: {img.max().item():.4f}")
                print(f"[DEBUG] Val mask  dtype: {mask.dtype}, shape: {tuple(mask.shape)}, min: {mask.min().item():.4f}, max: {mask.max().item():.4f}")

            pred = model(img)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)

            mask = align_mask_to_pred(pred, mask)
            loss = compute_loss(criterion, pred, mask, epoch)

            pred_sig = torch.sigmoid(pred)
            pred_bin = (pred_sig > threshold).float()

            batch_dice = dice_score(pred_sig, mask).item()
            batch_iou = iou_score(pred_sig, mask).item()
            batch_acc = pixel_accuracy(pred_bin, mask).item()
            batch_prec = precision_score(pred_bin, mask).item()
            batch_rec = recall_score(pred_bin, mask).item()

            running_loss += loss.item()
            running_dice += batch_dice
            running_iou += batch_iou
            running_acc += batch_acc
            running_prec += batch_prec
            running_rec += batch_rec

            pred_min_all.append(pred.min().item())
            pred_max_all.append(pred.max().item())

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{batch_dice:.4f}",
                "iou": f"{batch_iou:.4f}",
            })

    n = len(loader)
    return {
        "loss": running_loss / n,
        "dice": running_dice / n,
        "iou": running_iou / n,
        "acc": running_acc / n,
        "precision": running_prec / n,
        "recall": running_rec / n,
        "pred_min": min(pred_min_all) if len(pred_min_all) > 0 else 0.0,
        "pred_max": max(pred_max_all) if len(pred_max_all) > 0 else 0.0,
    }



def train_single_run(config, run_name="single_run"):
    set_seed(SEED)

    run_dir = os.path.join(SAVE_ROOT, run_name)
    ensure_dir(run_dir)

    print_config(config, title=f"RUN CONFIG: {run_name}")

    samples = build_image_mask_pairs(IMAGE_DIR, MASK_DIR)
    print(f"\nTotal samples available: {len(samples)}")

    train_samples, val_samples = train_test_split(
        samples,
        test_size=VAL_SIZE,
        random_state=SEED,
        shuffle=True
    )

    print(f"Training samples  : {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    train_image_paths = [x[0] for x in train_samples]
    train_mask_paths = [x[1] for x in train_samples]

    val_image_paths = [x[0] for x in val_samples]
    val_mask_paths = [x[1] for x in val_samples]

    train_dataset = BUSIDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        img_size=IMG_SIZE,
        train=True
    )

    val_dataset = BUSIDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        img_size=IMG_SIZE,
        train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    model = HybridSegNetStable().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    criterion = HybridLoss().to(DEVICE)

    if USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE
        )
    else:
        scheduler = None

    best_dice = -1.0
    best_epoch = -1
    best_model_wts = None
    epochs_no_improve = 0
    history = []

    for epoch in range(config["epochs"]):
        current_lr = optimizer.param_groups[0]["lr"]

        print("\n" + "-" * 80)
        print(f"Epoch {epoch+1}/{config['epochs']} | LR: {current_lr:.8f}")
        print("-" * 80)

        train_metrics = run_one_epoch_train(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            grad_clip=config["grad_clip"],
            threshold=config["threshold"]
        )

        val_metrics = run_one_epoch_val(
            model=model,
            loader=val_loader,
            criterion=criterion,
            epoch=epoch,
            threshold=config["threshold"]
        )

        epoch_record = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_acc": train_metrics["acc"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_acc": val_metrics["acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "pred_min": val_metrics["pred_min"],
            "pred_max": val_metrics["pred_max"],
        }
        history.append(epoch_record)

        print(f"Pred range: {val_metrics['pred_min']:.6f} to {val_metrics['pred_max']:.6f}")
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Train Dice: {train_metrics['dice']:.4f} | Train IoU: {train_metrics['iou']:.4f}")
        print(f"Train Acc : {train_metrics['acc']:.4f} | Prec: {train_metrics['precision']:.4f} | Rec: {train_metrics['recall']:.4f}")
        print(f"Val Loss  : {val_metrics['loss']:.4f}")
        print(f"Val Dice  : {val_metrics['dice']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
        print(f"Val Acc   : {val_metrics['acc']:.4f} | Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f}")

        if scheduler is not None:
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_metrics["dice"])
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                print(f"Learning rate reduced from {old_lr:.8f} to {new_lr:.8f}")

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            best_model_path = os.path.join(run_dir, "best_model.pth")
            torch.save(best_model_wts, best_model_path)

            print(f"New best model saved at epoch {best_epoch}")
            print(f"Best Val Dice: {best_dice:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        save_json(history, os.path.join(run_dir, "history.json"))

        if USE_EARLY_STOPPING and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    summary = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "final_train_loss": history[-1]["train_loss"],
        "final_train_dice": history[-1]["train_dice"],
        "final_train_iou": history[-1]["train_iou"],
        "final_val_loss": history[-1]["val_loss"],
        "final_val_dice": history[-1]["val_dice"],
        "final_val_iou": history[-1]["val_iou"],
        "history_path": os.path.join(run_dir, "history.json"),
        "best_model_path": os.path.join(run_dir, "best_model.pth"),
        "config": config,
    }

    save_json(summary, os.path.join(run_dir, "training_summary.json"))

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Epoch    : {best_epoch}")
    print(f"Best Val Dice : {best_dice:.4f}")
    print(f"Best Model    : {os.path.join(run_dir, 'best_model.pth')}")
    print("=" * 80)

    return summary



if __name__ == "__main__":
    ensure_dir(SAVE_ROOT)
    set_seed(SEED)
    train_single_run(config=CONFIG, run_name=RUN_NAME)
