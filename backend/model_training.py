import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from data_parser import SpectrogramDataset, build_label_encoder

def train_model(model,train_loader,val_loader,criterion,optimizer,scheduler,device,epochs,patience,model_directory,scaler):
    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(train_loader,desc=f"Epoch {epoch + 1}/{epochs}")

        for images, labels in progress:
            images = images.to(device,non_blocking=True)
            labels = labels.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_avg_loss = ( running_loss / len(train_loader))

        train_accuracy = (100.0 * correct / total)

        model.eval()
        val_total_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device,non_blocking=True)

                labels = labels.to(device,non_blocking=True)

                with torch.autocast(device_type="cuda",enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs,labels)

                val_total_loss += loss.item()

                _, predicted = torch.max(outputs,1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_avg_loss = (val_total_loss / len(val_loader))
        val_accuracy = (100.0 * val_correct / val_total)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train loss: {train_avg_loss:.4f} | Train accuracy: {train_accuracy:.2f}%")
        print(f"Val loss: {val_avg_loss:.4f} | Val accuracy: {val_accuracy:.2f}%")
        print(f"Learning rate: {current_lr:.2e}")

        # Reduce learning rate when validation loss stops improving
        scheduler.step(val_avg_loss)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(),os.path.join(model_directory,"best_model.pth"))
            print(f"Saved best model (validation accuracy: {val_accuracy:.2f}%)")

        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{patience} epochs")

        if epochs_without_improvement >= patience:
            print(f"Early stopping, validation accuracy did not improve for {patience} epochs.")
            break

    return best_val_accuracy

def evaluate_model(model,test_loader,criterion,device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device,non_blocking=True)
            labels = labels.to(device,non_blocking=True)

            with torch.autocast(device_type="cuda",enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs,labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_loss = (total_loss / len(test_loader))
    accuracy = (100.0 * correct / total)
    return average_loss, accuracy

def main():
    train_set_dir = "train_set_resized"
    val_set_dir = "val_set_resized"
    test_set_dir = "test_set_resized"

    model_directory = "model"
    os.makedirs(model_directory, exist_ok=True)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
        )

    print("Using device:",device)
    if device.type == "cuda":print("GPU:",torch.cuda.get_device_name(0))

    BATCH_SIZE = 64
    EPOCHS = 10
    EPOCH_PATIENCE = 7
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    label_encoder = build_label_encoder(train_set_dir)
    num_classes = len(label_encoder.classes_)
    print("Classes:",num_classes)

    np.save(
        os.path.join(model_directory,"label_encoder_classes.npy"),
        label_encoder.classes_
    )

    train_dataset = SpectrogramDataset(train_set_dir,label_encoder)
    val_dataset = SpectrogramDataset(val_set_dir,label_encoder)
    test_dataset = SpectrogramDataset(test_set_dir,label_encoder)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()

    model = models.efficientnet_b0(weights="DEFAULT")
    in_features = (model.classifier[1].in_features)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features,256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256,num_classes)
    )

    for param in model.features.parameters():
        param.requires_grad = False

    feature_blocks = list(model.features.children())

    fine_tune_start = int(len(feature_blocks) * 0.6)

    for i, block in enumerate(feature_blocks):
        if i >= fine_tune_start:
            for param in block.parameters():
                param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    )

    scaler = torch.amp.GradScaler("cuda",enabled=(device.type == "cuda"))

    best_val_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=EPOCHS,
        patience=EPOCH_PATIENCE,
        model_directory=model_directory,
        scaler=scaler
    )

    best_model_path = os.path.join(model_directory,"best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_accuracy = evaluate_model(model,test_loader,criterion,device)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2f}%")

    final_model_path = os.path.join(model_directory,"bird_recognition_effnet.pth")
    torch.save(model.state_dict(),final_model_path)
    print(f"Model saved to: {final_model_path}")

if __name__ == "__main__":
    main()