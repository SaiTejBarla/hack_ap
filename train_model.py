if __name__ == "__main__":
    import os
    import json
    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    from tqdm import tqdm

    # =======================
    # DEVICE CHECK
    # =======================
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"[INFO] GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
        print("[INFO] GPU not detected. Using CPU.")

    device_type = "cuda" if DEVICE.type == "cuda" else "cpu"

    # =======================
    # CONFIG
    # =======================
    DATA_DIR = "dataset/New Plant Diseases Dataset(Augmented)"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VALID_DIR = os.path.join(DATA_DIR, "valid")
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    MODEL_SAVE_DIR = "best_model"
    LABELS_PATH = "labels.json"
    CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, "checkpoint.pth")

    # =======================
    # DATA PREPROCESSING
    # =======================
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    transform_valid = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
    valid_dataset = datasets.ImageFolder(VALID_DIR, transform=transform_valid)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Save label mapping
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    with open(LABELS_PATH, "w") as f:
        json.dump(idx_to_class, f, indent=4)

    # =======================
    # MODEL SETUP
    # =======================
    print("[INFO] Loading pretrained MobileNetV2...")
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(train_dataset.classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    # =======================
    # CHECKPOINT RESUME
    # =======================
    best_acc = 0.0
    start_epoch = 0
    start_batch = 0
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"[INFO] Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        best_acc = checkpoint.get("best_acc", 0.0)
        start_epoch = checkpoint.get("epoch", 0)
        start_batch = checkpoint.get("batch", 0)
        print(f"[INFO] Resuming training from epoch {start_epoch+1}, batch {start_batch}")

    # =======================
    # TRAINING LOOP
    # =======================
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        epoch_start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training", ncols=100)):
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device_type, enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 20 == 0:
                elapsed = time.time() - epoch_start_time
                batches_left = len(train_loader) - batch_idx
                est_remaining = elapsed / (batch_idx+1) * batches_left
                print(f"[INFO] Estimated time remaining for epoch: {est_remaining/60:.2f} mins")

            torch.save({
                "epoch": epoch,
                "batch": batch_idx,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_acc": best_acc
            }, CHECKPOINT_PATH)

        train_acc = correct / total
        train_loss = running_loss / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader, desc="Validation", ncols=100):
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=device_type, enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} "
              f"Train Loss: {train_loss:.4f} "
              f"Train Acc: {train_acc:.4f} "
              f"Val Acc: {val_acc:.4f} "
              f"Epoch Time: {epoch_time/60:.2f} mins")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "model.pth"))
            print(f"[INFO] Saved best model (val_acc={best_acc:.4f})")

        torch.save({
            "epoch": epoch,
            "batch": 0,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_acc": best_acc
        }, CHECKPOINT_PATH)
        print(f"[INFO] Checkpoint saved at epoch {epoch+1}")

    print(f"\n[INFO] Training complete. Best Val Acc: {best_acc:.4f}")
