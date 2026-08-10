import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from data_parser import SpectrogramDataset, build_label_encoder

def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                 device, epochs, patience, model_directory, best_val_accuracy, phase_name,
                 scaler):
    """
    Runs one training phase with early stopping. Validation is done inline
    at the end of every epoch, rather than through a separate function.
    """
    epochs_without_improvement = 0

    for epoch in range(epochs):
        #training pass
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        progress = tqdm(train_loader, desc=f"[{phase_name}] Epoch {epoch+1}/{epochs}")

        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
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

        train_avg_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        #validation pass
        model.eval()
        val_total_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_avg_loss = val_total_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        print()
        print(f"[{phase_name}] Epoch {epoch+1}")
        print(f"Train loss: {train_avg_loss:.4f} | Train accuracy: {train_accuracy:.2f}%")
        print(f"Val loss:   {val_avg_loss:.4f} | Val accuracy:   {val_accuracy:.2f}%")

        scheduler.step(val_avg_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(model_directory, "best_model.pth"))
            print("Saved best model (based on val accuracy)")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nNo val improvement for {patience} epochs, stopping {phase_name} early.")
            break

    return best_val_accuracy


def main():

    #folders containing all the train sets and the model directory
    train_set_dir = "train_set"
    val_set_dir = "val_set"
    test_set_dir = "test_set"

    model_directory = "model"
    os.makedirs(model_directory, exist_ok=True)

    #testing if the model recognizes the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    #settings for the model later on
    BATCH_SIZE = 32
    PHASE1_EPOCHS = 20
    PHASE2_EPOCHS = 20

    EPOCH_PATIENCE = 6

    # labels
    label_encoder = build_label_encoder(train_set_dir)
    num_classes = len(label_encoder.classes_)
    print("Classes:", num_classes)
    np.save(os.path.join(model_directory, "label_encoder_classes.npy"), label_encoder.classes_)

    # datasets
    train_dataset = SpectrogramDataset(train_set_dir, label_encoder)
    val_dataset = SpectrogramDataset(val_set_dir, label_encoder)
    test_dataset = SpectrogramDataset(test_set_dir, label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # class weights - accounts for uneven sample counts across species
    train_labels = [label for _, _, label in train_dataset.samples]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=train_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # EfficientNet model
    model = models.efficientnet_b0(weights="DEFAULT")

    # freeze the entire backbone for phase 1
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = (model.classifier[1].in_features)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    model.to(device)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_val_accuracy = 0

    #phase 1

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    best_val_accuracy = train_phase(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, PHASE1_EPOCHS, EPOCH_PATIENCE, model_directory, best_val_accuracy,
        phase_name="Phase 1", scaler=scaler
    )

    #phase 2
    model.load_state_dict(torch.load(os.path.join(model_directory, "best_model.pth")))

    feature_blocks = list(model.features.children())
    fine_tune_at = int(len(feature_blocks) * 0.8)

    for i, block in enumerate(feature_blocks):
        if i >= fine_tune_at:
            for param in block.parameters():
                param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7)

    best_val_accuracy = train_phase(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, PHASE2_EPOCHS, EPOCH_PATIENCE, model_directory, best_val_accuracy,
        phase_name="Phase 2", scaler=scaler
    )


    #final testing and evaluation
    model.load_state_dict(torch.load(os.path.join(model_directory, "best_model.pth")))
    model.eval()

    test_total_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            test_total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss = test_total_loss / len(test_loader)
    test_accuracy = 100 * test_correct / test_total

    print()
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    torch.save(model.state_dict(), os.path.join(model_directory, "bird_recognition_effnet.pth"))
    print("Final model saved")

if __name__ == "__main__":
    main()