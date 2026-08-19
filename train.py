"""
train.py — teaches a model to tell cracked concrete from clean concrete.

WHAT YOU NEED BEFORE RUNNING
----------------------------
A folder of images arranged like this:

    dataset/
        cracked/
            img001.jpg
            img002.jpg
            ...
        not_cracked/
            img500.jpg
            ...

That's it. The folder NAMES are the labels. This script splits the data
into train/val/test by itself, so don't pre-split it.

HOW TO RUN
----------
    python train.py

Change DATA_DIR below to point at your folder.

WHAT YOU GET
------------
    crack_model.pt   <- the trained model, used later by predict.py
"""

import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ----------------------------------------------------------------------
# SETTINGS — these are the only lines you normally need to touch
# ----------------------------------------------------------------------

DATA_DIR    = "dataset_small"        # folder containing cracked/ and not_cracked/
CLASSES     = ["not_cracked", "cracked"]   # index 0 and 1. Order matters!
IMG_SIZE    = 160              # images get resized to 224x224
BATCH_SIZE  = 32               # lower this to 16 or 8 if you run out of memory
EPOCHS_HEAD = 2                # warmup passes with the backbone frozen
EPOCHS_FULL = 2               # passes where the whole model learns
LR_HEAD     = 1e-3             # learning rate during warmup
LR_FULL     = 1e-4             # learning rate during full fine-tuning
SEED        = 42
MODEL_OUT   = "crack_model.pt"

# Group images by the part of the filename before the first "-" or "_".
# WHY: many crack datasets are one wall photo chopped into many small tiles,
# named like 7001-1.jpg, 7001-2.jpg. If tiles of the same wall land in both
# train and test, the model memorises the wall and your score is a lie.
# Set this to False if your filenames are unrelated to each other.
GROUP_BY_FILENAME_PREFIX = False

# ----------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ----------------------------------------------------------------------
# STEP 1 — find every image and give it a label
# ----------------------------------------------------------------------

def collect_files():
    """Returns a list of (filepath, label) where label is 0 or 1."""
    items = []
    for label_index, class_name in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(folder):
            raise SystemExit(
                f"Could not find the folder '{folder}'.\n"
                f"Expected: {DATA_DIR}/cracked/ and {DATA_DIR}/not_cracked/"
            )
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(IMAGE_EXTS):
                items.append((os.path.join(folder, name), label_index))
    if not items:
        raise SystemExit(f"No images found inside '{DATA_DIR}'.")
    return items


def group_key(filepath):
    """
    Works out which 'parent photo' an image came from, so we can keep all
    its pieces on the same side of the split.
    e.g. '7001-3.jpg' -> '7001'.  If grouping is off, every file is its own group.
    """
    name = os.path.basename(filepath)
    if not GROUP_BY_FILENAME_PREFIX:
        return name
    stem = os.path.splitext(name)[0]
    match = re.split(r"[-_]", stem)[0]
    return match if match else stem


def split_data(items, val_frac=0.15, test_frac=0.15):
    """Splits into train/val/test WITHOUT breaking groups apart."""
    groups = defaultdict(list)
    for path, label in items:
        groups[group_key(path)].append((path, label))

    keys = list(groups.keys())
    random.shuffle(keys)

    n_test = int(len(keys) * test_frac)
    n_val = int(len(keys) * val_frac)

    test_keys = keys[:n_test]
    val_keys = keys[n_test:n_test + n_val]
    train_keys = keys[n_test + n_val:]

    def gather(key_list):
        out = []
        for k in key_list:
            out.extend(groups[k])
        return out

    return gather(train_keys), gather(val_keys), gather(test_keys)


# ----------------------------------------------------------------------
# STEP 2 — how images are loaded and altered
# ----------------------------------------------------------------------

# These numbers are the colour averages of ImageNet. The pretrained model
# expects its input adjusted this way — just copy them.
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# TRAINING images get randomly flipped/rotated/brightened. A flipped crack is
# still a crack, so the label stays true, but the model sees more variety and
# can't just memorise individual pictures.
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    NORMALIZE,
])

# VALIDATION / TEST images get NO random changes — we want a stable score.
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    NORMALIZE,
])


class CrackDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.float32)


# ----------------------------------------------------------------------
# STEP 3 — build the model
# ----------------------------------------------------------------------

def build_model():
    """
    Downloads ResNet-18 already trained on millions of everyday photos,
    then replaces its final layer with one that answers our question.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)   # 1 number out: high = cracked
    return model.to(DEVICE)


def set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable


# ----------------------------------------------------------------------
# STEP 4 — the training and evaluation loops
# ----------------------------------------------------------------------

def run_epoch(model, loader, loss_fn, optimizer=None):
    """One full pass over a dataset. Trains if an optimizer is given."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss, all_probs, all_labels = 0.0, [], []

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images).squeeze(1)
            loss = loss_fn(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / len(loader.dataset), np.array(all_probs), np.array(all_labels)


def scores(probs, labels, threshold=0.5):
    """Precision, recall and F1 — far more informative than accuracy alone."""
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(labels), 1)

    return {"acc": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    items = collect_files()
    train_items, val_items, test_items = split_data(items)

    n_cracked = sum(1 for _, lab in items if lab == 1)
    print(f"\nFound {len(items)} images "
          f"({n_cracked} cracked, {len(items) - n_cracked} not cracked)")
    print(f"Train: {len(train_items)}   Val: {len(val_items)}   Test: {len(test_items)}")

    train_loader = DataLoader(CrackDataset(train_items, train_tf),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers= 4)
    val_loader = DataLoader(CrackDataset(val_items, eval_tf),
                            batch_size=BATCH_SIZE, num_workers= 4)
    test_loader = DataLoader(CrackDataset(test_items, eval_tf),
                             batch_size=BATCH_SIZE, num_workers= 4)

    model = build_model()

    # If one class has far more images, tell the loss to care more about
    # the rarer one, otherwise the model just guesses the common class.
    n_pos = sum(1 for _, lab in train_items if lab == 1)
    n_neg = len(train_items) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = -1.0

    def validate_and_save(tag):
        nonlocal best_f1
        _, probs, labels = run_epoch(model, val_loader, loss_fn)
        m = scores(probs, labels)
        print(f"  {tag} | val acc {m['acc']:.3f}  precision {m['precision']:.3f}  "
              f"recall {m['recall']:.3f}  F1 {m['f1']:.3f}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save({"state_dict": model.state_dict(),
                        "classes": CLASSES,
                        "img_size": IMG_SIZE}, MODEL_OUT)
            print(f"        ^ best so far, saved to {MODEL_OUT}")

    # --- Phase 1: warm up only the new final layer -----------------------
    # The new layer starts random. Letting it loose on the whole pretrained
    # network straight away would scramble the useful weights.
    print("\nPhase 1 — training the new final layer only")
    set_backbone_trainable(model, False)
    opt = torch.optim.AdamW(model.fc.parameters(), lr=LR_HEAD)
    for epoch in range(1, EPOCHS_HEAD + 1):
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, opt)
        print(f"Epoch {epoch}/{EPOCHS_HEAD}  train loss {train_loss:.4f}")
        validate_and_save(f"epoch {epoch}")

    # --- Phase 2: fine-tune everything, gently ---------------------------
    print("\nPhase 2 — fine-tuning the whole network")
    set_backbone_trainable(model, True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_FULL)
    for epoch in range(1, EPOCHS_FULL + 1):
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, opt)
        sched.step()
        print(f"Epoch {epoch}/{EPOCHS_FULL}  train loss {train_loss:.4f}")
        validate_and_save(f"epoch {epoch}")

    # --- Final exam: the test set, touched exactly once -------------------
    print("\nLoading the best saved model and testing it")
    model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE)["state_dict"])
    _, probs, labels = run_epoch(model, test_loader, loss_fn)
    m = scores(probs, labels)

    print("\n=========== TEST RESULTS ===========")
    print(f"Accuracy   {m['acc']:.3f}")
    print(f"Precision  {m['precision']:.3f}   (of those called cracked, how many were)")
    print(f"Recall     {m['recall']:.3f}   (of the real cracks, how many were caught)")
    print(f"F1         {m['f1']:.3f}")
    print("\nConfusion matrix")
    print(f"  correctly called cracked      {m['tp']}")
    print(f"  correctly called clean        {m['tn']}")
    print(f"  false alarms (clean->cracked) {m['fp']}")
    print(f"  MISSED cracks                 {m['fn']}")
    print("====================================")

    # Show the worst mistakes so you can go and look at those images.
    wrong = [(test_items[i][0], probs[i], int(labels[i]))
             for i in range(len(labels))
             if (probs[i] >= 0.5) != bool(labels[i])]
    if wrong:
        wrong.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)
        print("\nMost confident mistakes — open these images and look at them:")
        for path, prob, true_label in wrong[:10]:
            print(f"  {path}  predicted {prob:.2f}, actually {CLASSES[true_label]}")

    print(f"\nDone. Model saved as {MODEL_OUT} — now run predict.py")


if __name__ == "__main__":
    main()
