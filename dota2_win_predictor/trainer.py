import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from models.predictor import Dota2MLP
from data.dataset import Dota2OneHotDataset
import os
from tqdm import tqdm

# Настройки
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
HERO_COUNT = 126
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "data/processed/matches_indexed.jsonl"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def binary_accuracy(y_pred, y_true):
    preds = torch.sigmoid(y_pred)
    predicted = (preds >= 0.5).float()
    correct = (predicted == y_true).float().sum()
    return correct / y_true.numel()

# Загрузка датасета
dataset = Dota2OneHotDataset(DATA_PATH, hero_count=HERO_COUNT)
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# Модель, оптимизатор, loss
model = Dota2MLP(hero_count=HERO_COUNT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for x, y in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}] Training"):
        x, y = x.to(DEVICE), y.to(DEVICE).float().unsqueeze(1)  # Убедись, что y float и размер (batch,1)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        acc = binary_accuracy(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")

    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}] Validation"):
            x, y = x.to(DEVICE), y.to(DEVICE).float().unsqueeze(1)
            logits = model(x)
            loss = criterion(logits, y)
            acc = binary_accuracy(logits, y)

            val_loss += loss.item()
            val_acc += acc.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    print(f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {avg_val_acc:.4f}")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch{epoch+1}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch + 1,
    }, checkpoint_path)
    print(f"Сохранена модель: {checkpoint_path}")
