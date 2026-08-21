from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data_parser import SpectrogramDataset, build_label_encoder

BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "train_set_resized"
VAL_DIR = BASE_DIR / "val_set_resized"
TEST_DIR = BASE_DIR / "test_set_resized"

TEACHER_DIR = BASE_DIR / "birdnet_teacher"
TEACHER_PROBS = TEACHER_DIR / "teacher_probs.npy"
TEACHER_CLASSES = TEACHER_DIR / "class_order.npy"

DISTILLATION_ALPHA = 0.3
DISTILLATION_TEMPERATURE = 3.0

MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
BALANCE_POWER = 0.5

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, scaler, start_epoch=0, best_val_accuracy=-1.0, epochs_without_improvement=0):
    for epoch in range(start_epoch, EPOCHS):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images, labels, teacher_targets in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            teacher_targets = teacher_targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                hard_loss = criterion(outputs, labels)
                temperature = DISTILLATION_TEMPERATURE

                student_log_probs = torch.log_softmax(outputs / temperature, dim = 1)

                teacher_probs_soft = teacher_targets.pow(1.0 / temperature)

                teacher_probs_soft = (teacher_probs_soft / teacher_probs_soft.sum(dim=1, keepdim = True))

                soft_loss = (torch.nn.functional.kl_div(student_log_probs, teacher_probs_soft, reduction="batchmean") * (temperature ** 2))

                loss = ((1.0 - DISTILLATION_ALPHA) * hard_loss + DISTILLATION_ALPHA * soft_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100 * correct / total:.1f}%"
            )

        train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_top5_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = outputs.argmax(dim=1)

                val_correct += ((predicted == labels).sum().item())
                top5_predictions = outputs.topk(5, dim=1).indices

                val_top5_correct += ((top5_predictions == labels.unsqueeze(1)).any(dim=1).sum().item())

                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = (100 * val_correct / val_total)
        val_top5_accuracy = (100 * val_top5_correct / val_total)

        scheduler.step(val_loss)

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.2f}% | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_accuracy:.2f}% | "
            f"Val top-5: {val_top5_accuracy:.2f}%"
        )

        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")

            print(f"Saved best model: {val_accuracy:.2f}%")

        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{PATIENCE}")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "epochs_without_improvement": epochs_without_improvement,
        }

        torch.save(checkpoint, MODEL_DIR / "training_checkpoint.pth")

        print(f"Saved training checkpoint after epoch {epoch + 1}.")

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    return best_val_accuracy
    
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)

            correct += ((predicted == labels).sum().item())

            top5_predictions = outputs.topk(5, dim=1).indices
            top5_correct += ((top5_predictions == labels.unsqueeze(1)).any(dim=1).sum().item())

            total += labels.size(0)

    return (total_loss / len(loader), 100 * correct / total,100 * top5_correct / total)


def create_balanced_sampler(dataset,balance_power=0.5):
    labels = np.array(
        [
            sample[2]
            for sample in dataset.samples
        ]
    )

    class_counts = np.bincount(labels)
    class_weights = (1.0 / (class_counts ** balance_power))

    sample_weights = class_weights[labels]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler

def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    label_encoder = build_label_encoder(str(TRAIN_DIR))
    num_classes = len(label_encoder.classes_)
    teacher_probs = np.load(TEACHER_PROBS)
    teacher_classes = np.load(TEACHER_CLASSES, allow_pickle=True)

    print(f"Teacher matrix loaded: {teacher_probs.shape}")
    print(f"Teacher probability range: {teacher_probs.min():.6f} - {teacher_probs.max():.6f}")
    print(f"Teacher row sums: {teacher_probs.sum(axis=1).min():.6f} - {teacher_probs.sum(axis=1).max():.6f}")

    if teacher_probs.shape != (num_classes, num_classes):
        raise ValueError(f"Teacher matrix shape {teacher_probs.shape} does not match {num_classes} classes.")

    if not np.array_equal(teacher_classes, label_encoder.classes_):
        raise ValueError("Teacher class order does not match training class order.")

    train_dataset = SpectrogramDataset(str(TRAIN_DIR), label_encoder,teacher_probs)
    val_dataset = SpectrogramDataset(str(VAL_DIR),label_encoder)
    test_dataset = SpectrogramDataset(str(TEST_DIR), label_encoder)

    print(f"\nFinal number of classes: {num_classes}")
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"\nValidation samples: {len(val_dataset)}")
    print(f"\nTest samples: {len(test_dataset)}")

    train_sampler = create_balanced_sampler(
        train_dataset,
        balance_power=BALANCE_POWER
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model = models.efficientnet_b0(weights="DEFAULT")
    in_features = (model.classifier[1].in_features)

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    for param in model.features.parameters():
        param.requires_grad = False

    feature_blocks = list(model.features.children())
    fine_tune_start = int(len(feature_blocks) * 0.5)

    for i, block in enumerate(feature_blocks):
        if i >= fine_tune_start:
            for param in block.parameters():
                param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    scaler = torch.amp.GradScaler("cuda",enabled=device.type == "cuda")

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"\nTrainable parameters: {trainable:,}/{total_params:,}")

    checkpoint_path = (MODEL_DIR / "training_checkpoint.pth")
    start_epoch = 0
    best_val_accuracy = -1.0
    epochs_without_improvement = 0

    if checkpoint_path.exists():
        print(f"\nLoading training checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint["epoch"]
        best_val_accuracy = (checkpoint["best_val_accuracy"])
        epochs_without_improvement = (checkpoint.get("epochs_without_improvement",0))

        print(f"Checkpoint loaded successfully.")
        print(f"Resuming from epoch {start_epoch + 1}/{EPOCHS}")
        print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
        print(f"Epochs without improvement: {epochs_without_improvement}")

    else:
        print("\nNo training checkpoint found. Starting from epoch 1.")

    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        scaler,
        start_epoch=start_epoch,
        best_val_accuracy=best_val_accuracy,
        epochs_without_improvement=epochs_without_improvement
    )

    best_model_path = (MODEL_DIR / "best_model.pth")

    if not best_model_path.exists():
        print("Error: best_model.pth was not created.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    test_loss, test_accuracy, test_top5 = evaluate_model(model, test_loader, criterion, device)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test top-1 accuracy: {test_accuracy:.2f}%")
    print(f"Test top-5 accuracy: {test_top5:.2f}%")
    final_path = (MODEL_DIR / "bird_recognition_effnet.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nModel saved to: {final_path}")

if __name__ == "__main__":
    main()