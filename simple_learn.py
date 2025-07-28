import os
import random
import gc
import warnings
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from matplotlib import cm

from sklearn.metrics import precision_recall_curve, average_precision_score
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------------------
# 1) KONFIGURACJA
# -------------------------------------------------------------------------
DATA_CSV = "output_labels_speed.csv"
DATA_ROOT = "light_speed"
NUM_CLASS = 5
# 'resnet18', 'resnet50', 'efficientnet_b0'
MODEL_NAME = "efficientnet_b0"
IMG_SIZE = (600, 700)
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# 2) DANESET + TRANSFORMACJE
# -------------------------------------------------------------------------


class SimpleImageDS(Dataset):
    """Dataset oczekujący pliku CSV (img_rel_path,label) i folderu ze zdjęciami."""

    def __init__(self, csv, root, tfm=None):
        import pandas as pd
        self.ann = pd.read_csv(csv, header=None)
        self.root = root
        self.tfm = tfm

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.ann.iloc[idx, 0])
        label = int(self.ann.iloc[idx, 1])
        img = Image.open(img_path).convert("RGB")
        if self.tfm:
            img = self.tfm(img)
        return img, label, img_path


transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

ds = SimpleImageDS(DATA_CSV, DATA_ROOT, transform)

# Podział: 70% train, 15% val, 15% test
indices = np.random.permutation(len(ds))
n_total = len(indices)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

train_loader = DataLoader(torch.utils.data.Subset(
    ds, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(torch.utils.data.Subset(
    ds, val_idx),   batch_size=BATCH_SIZE)
test_loader = DataLoader(torch.utils.data.Subset(
    ds, test_idx),  batch_size=BATCH_SIZE)


# -------------------------------------------------------------------------
# 3) WYBÓR I MODYFIKACJA PRETRENOWANEGO MODELU
# -------------------------------------------------------------------------

def get_model(name: str):
    """Zwraca wybrany model z wymienionymi ostatnimi warstwami."""
    if name.startswith("resnet"):
        model = getattr(models, name)(weights="DEFAULT")
        for p in model.parameters():                       # zamranie 
            p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASS)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, NUM_CLASS)
    else:
        raise ValueError("Nieznana nazwa modelu")
    return model


model = get_model(MODEL_NAME).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------------------------------------------------------
# 4) PĘTLA TRENINGOWA
# -------------------------------------------------------------------------
train_hist, acc_hist = [], []
for epoch in range(EPOCHS):
    # ------- TRENING -------
    model.train()
    running_loss = correct = total = 0
    for x, y, _ in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    train_loss, train_acc = running_loss / total, correct / total

    # ------- WALIDACJA -------
    model.eval()
    vloss = vcorrect = vtotal = 0
    with torch.no_grad():
        for x, y, _ in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            vloss += criterion(out, y).item() * y.size(0)
            vcorrect += (out.argmax(1) == y).sum().item()
            vtotal += y.size(0)
    val_loss, val_acc = vloss / vtotal, vcorrect / vtotal

    train_hist.append((train_loss, val_loss))
    acc_hist.append((train_acc,  val_acc))
    print(f"Ep {epoch+1}/{EPOCHS} | loss {train_loss:.3f}/{val_loss:.3f} | acc {train_acc:.2%}/{val_acc:.2%}")

# -------------------------------------------------------------------------
# 5) EWALUACJA NA ZBIORZE TESTOWYM
# -------------------------------------------------------------------------
model.eval()
test_loss = test_correct = test_total = 0

with torch.no_grad():
    for x, y, _ in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        test_loss += criterion(out, y).item() * y.size(0)
        test_correct += (out.argmax(1) == y).sum().item()
        test_total += y.size(0)

final_test_loss = test_loss / test_total
final_test_acc = test_correct / test_total
print(
    f"\nTest loss: {final_test_loss:.4f} | Test accuracy: {final_test_acc:.2%}")


# ---------------------------------------------------------------
# 6a) PRECISION-RECALL CURVES
# ---------------------------------------------------------------
def compute_pr_curve(labels, scores):
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    tp = 0
    fp = 0
    fn = np.sum(sorted_labels)

    precisions = []
    recalls = []

    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    unique_recalls, indices = np.unique(recalls, return_index=True)
    unique_precisions = np.array(precisions)[indices]

    return unique_recalls, unique_precisions


all_targets = []
all_probs = []

model.eval()
with torch.no_grad():
    for x, y, _ in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.numpy())

all_probs = np.concatenate(all_probs, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

pr_dir = os.path.join(OUT_DIR, "pr_curves")
os.makedirs(pr_dir, exist_ok=True)

plt.figure(figsize=(10, 8))
for i in range(NUM_CLASS):
    binary_labels = (all_targets == i).astype(int)
    scores = all_probs[:, i]

    recalls, precisions = compute_pr_curve(binary_labels, scores)
    ap = np.trapz(precisions, recalls)

    plt.plot(recalls, precisions, label=f"Klasa {i} (AP={ap:.2f})")

plt.xlabel("Czułość (Recall)")
plt.ylabel("Precyzja (Precision)")
plt.title("Precision-Recall Curves dla każdej klasy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pr_dir, f"precision_recall_all_classes.png"))
plt.close()
print(f"Krzywa PR dla wszystkich klas zapisana w {pr_dir}")


# -------------------------------------------------------------------------
# 6b) STRATA + DOKŁADNOŚĆ
# -------------------------------------------------------------------------

epoch_axis = np.arange(1, EPOCHS + 1)
train_loss, val_loss = zip(*train_hist)
train_acc,  val_acc = zip(*acc_hist)
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(epoch_axis, train_loss, label="train loss", marker="o")
ax1.plot(epoch_axis, val_loss,   label="val loss",   marker="o")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax2 = ax1.twinx()
ax2.plot(epoch_axis, train_acc, label="train acc", linestyle="--")
ax2.plot(epoch_axis, val_acc,   label="val acc",   linestyle="--")
ax2.set_ylabel("accuracy")
fig.legend(loc="upper center", ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "training_curve.png"))
plt.close(fig)

# -------------------------------------------------------------------------
# 7) ATRYBUCJE GRADIENTOWE – 5 obrazów / klasę
# -------------------------------------------------------------------------
ig = IntegratedGradients(model)
model.eval()
samples_per_class = 5
class_slots = {c: [] for c in range(NUM_CLASS)}

for idx in val_idx:
    _, label, _ = ds[idx]
    if len(class_slots[label]) < samples_per_class:
        class_slots[label].append(idx)
    if all(len(v) == samples_per_class for v in class_slots.values()):
        break

heat_dir = os.path.join(OUT_DIR, "attributions")
os.makedirs(heat_dir, exist_ok=True)

for cls, idx_list in class_slots.items():
    for j, idx in enumerate(idx_list):
        img, label, path = ds[idx]
        inp = img.unsqueeze(0).to(DEVICE)
        attr = ig.attribute(inp, target=label).squeeze().cpu().numpy()
        attr = np.sum(np.abs(attr), axis=0)
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)

        orig = np.transpose(img.numpy(), (1, 2, 0))
        orig = (orig * np.array([0.229, 0.224, 0.225]
                                ) + np.array([0.485, 0.456, 0.406]))
        orig = orig.clip(0, 1)

        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(orig)
        axs[0].set_title("Oryginalny")
        axs[0].axis("off")

        attr_contrast = np.power(attr, 0.5)
        axs[1].imshow(1 - attr_contrast, cmap='gray')
        axs[1].set_title("Int.Gradients")
        axs[1].axis("off")

        fig.tight_layout()
        fig.savefig(os.path.join(heat_dir, f"cls{cls}_s{j}.png"))
        plt.close(fig)
        del inp, attr
        torch.cuda.empty_cache()
        gc.collect()

print("Gotowe! Wyniki w folderze:", OUT_DIR)
