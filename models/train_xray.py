import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.dirname(__file__))
from xray_cnn import XrayCNN

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data", "xray_synth")
MODEL_PATH = os.path.join(HERE, "xray_model.pt")
METRICS_PATH = os.path.join(HERE, "xray_metrics.json")
IMG_SIZE = 128

CLASS_NAMES = ["normal", "abnormal"]


class XrayDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        for label, cls in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(root, cls)
            for fn in sorted(os.listdir(cls_dir)):
                self.samples.append((os.path.join(cls_dir, fn), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
        tensor = torch.from_numpy(arr).unsqueeze(0)  # 1xHxW
        return tensor, label


def main():
    ds = XrayDataset(DATA_DIR)
    n_val = int(len(ds) * 0.2)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XrayCNN(num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 8
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        print(f"Epoch {epoch+1}/{EPOCHS} - train loss: {total_loss/len(train_ds):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())

    metrics = {
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "precision": round(precision_score(all_labels, all_preds), 4),
        "recall": round(recall_score(all_labels, all_preds), 4),
        "f1_score": round(f1_score(all_labels, all_preds), 4),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "class_names": CLASS_NAMES,
    }

    torch.save(model.state_dict(), MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to", MODEL_PATH)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
