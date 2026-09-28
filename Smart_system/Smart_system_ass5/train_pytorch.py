"""
Assignment 5 - PyTorch Training Pipeline (NEW DATASETS on RTX 3060 GPU)
Trains 3-layer and 5-layer CNNs across all 3 benchmark datasets:
1. Fashion-MNIST (Zalando 10 classes, 28x28 grayscale)
2. SVHN (Street View House Numbers 10 classes, 32x32x3 RGB)
3. Diabetes (1D-CNN on 513,703 rows)

Outputs saved into distinct folders:
- models_pytorch/01_fashion_mnist/
- models_pytorch/02_svhn/
- models_pytorch/03_diabetes/
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_BASE = os.path.join(BASE_DIR, "models_pytorch")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PyTorchCNN3LayerImage(nn.Module):
    def __init__(self, in_channels, num_classes, spatial_dim=28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        flat_dim = 64 * (spatial_dim // 4) * (spatial_dim // 4)
        self.fc = nn.Linear(flat_dim, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


class PyTorchCNN5LayerImage(nn.Module):
    def __init__(self, in_channels, num_classes, spatial_dim=28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)
        
        flat_dim = 64 * (spatial_dim // 4) * (spatial_dim // 4)
        self.fc1 = nn.Linear(flat_dim, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu5 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.drop1(self.pool1(self.relu2(self.bn2(self.conv2(x)))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.drop2(self.pool2(self.relu4(self.bn4(self.conv4(x)))))
        x = torch.flatten(x, 1)
        x = self.drop3(self.relu5(self.bn5(self.fc1(x))))
        return self.fc2(x)


class PyTorch1DCNN3LayerTabular(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2, 2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2, 2)
        
        flat_dim = 64 * (in_features // 4)
        self.fc = nn.Linear(flat_dim, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


class PyTorch1DCNN5LayerTabular(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2, 2)
        self.drop1 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2, 2)
        self.drop2 = nn.Dropout(0.2)
        
        flat_dim = 64 * (in_features // 4)
        self.fc1 = nn.Linear(flat_dim, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.relu5 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.drop1(self.pool1(self.relu2(self.bn2(self.conv2(x)))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.drop2(self.pool2(self.relu4(self.bn4(self.conv4(x)))))
        x = torch.flatten(x, 1)
        x = self.drop3(self.relu5(self.bn5(self.fc1(x))))
        return self.fc2(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def plot_and_save_pytorch_history(h3, h5, out_dir, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(h3['train_acc'], label='3-Layer Train Acc', color='#3498db', linestyle='--')
    axes[0].plot(h3['val_acc'], label='3-Layer Val Acc', color='#2980b9', linewidth=2)
    axes[0].plot(h5['train_acc'], label='5-Layer Train Acc', color='#e74c3c', linestyle='--')
    axes[0].plot(h5['val_acc'], label='5-Layer Val Acc', color='#c0392b', linewidth=2)
    axes[0].set_title(f"{dataset_name} (PyTorch) — Accuracy over Epochs", fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(h3['train_loss'], label='3-Layer Train Loss', color='#3498db', linestyle='--')
    axes[1].plot(h3['val_loss'], label='3-Layer Val Loss', color='#2980b9', linewidth=2)
    axes[1].plot(h5['train_loss'], label='5-Layer Train Loss', color='#e74c3c', linestyle='--')
    axes[1].plot(h5['val_loss'], label='5-Layer Val Loss', color='#c0392b', linewidth=2)
    axes[1].set_title(f"{dataset_name} (PyTorch) — Loss over Epochs", fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f"PyTorch Training Curves: 3-Layer vs 5-Layer on {dataset_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart_path = os.path.join(out_dir, "training_history.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    return chart_path


def plot_and_save_pytorch_confusion_matrices(cm3, cm5, class_names, out_dir, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.heatmap(cm3, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title("3-Layer CNN Confusion Matrix", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    
    sns.heatmap(cm5, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title("5-Layer CNN Confusion Matrix", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    
    plt.suptitle(f"Confusion Matrices: PyTorch 3-Layer vs 5-Layer on {dataset_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    return cm_path


def save_sample_predictions_pytorch(model, x_test_t, y_test_np, class_names, out_dir, is_image=True):
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
    sample_indices = np.random.RandomState(42).choice(len(x_test_t), 10, replace=False)
    
    with torch.no_grad():
        sub_inputs = x_test_t[sample_indices].to(DEVICE)
        outputs = model(sub_inputs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        pred_labels = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        
    for i, idx in enumerate(sample_indices):
        ax = axes.flat[i]
        true_cls = class_names[y_test_np[idx]]
        pred_cls = class_names[pred_labels[i]]
        conf = confidences[i] * 100
        is_correct = (y_test_np[idx] == pred_labels[i])
        color = 'green' if is_correct else 'red'
        
        if is_image:
            img = x_test_t[idx].numpy()
            if img.shape[0] == 1:
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                ax.imshow(np.transpose(img, (1, 2, 0)))
            ax.axis('off')
        else:
            ax.barh(range(min(len(class_names), 5)), probs[i][:5], color=color, alpha=0.7)
            ax.set_yticks(range(min(len(class_names), 5)))
            ax.set_yticklabels(class_names[:5], fontsize=8)
            ax.set_xlim(0, 1.0)
            
        ax.set_title(f"True: {true_cls}\nPred: {pred_cls} ({conf:.1f}%)",
                     fontsize=8.5, fontweight='bold', color=color)
                     
    plt.suptitle("Sample Predictions Showcase (PyTorch 5-Layer Model on GPU)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    pred_path = os.path.join(out_dir, "sample_predictions.png")
    plt.savefig(pred_path, dpi=200)
    plt.close()
    return pred_path


def train_pytorch_pipeline(dataset_key):
    print("\n" + "=" * 70)
    print(f"RUNNING PYTORCH PIPELINE FOR: {dataset_key.upper()} (DEVICE: {DEVICE})")
    print("=" * 70)
    
    if dataset_key == "fashion_mnist":
        out_dir = os.path.join(OUTPUT_BASE, "01_fashion_mnist")
        npz_file = os.path.join(DATA_DIR, "fashion_mnist", "fashion_mnist_data.npz")
        data = np.load(npz_file)
        x_train = np.expand_dims(data['x_train'].astype(np.float32) / 255.0, 1)
        y_train = data['y_train'].astype(np.int64)
        x_test = np.expand_dims(data['x_test'].astype(np.float32) / 255.0, 1)
        y_test = data['y_test'].astype(np.int64)
        
        num_classes = 10
        class_names = [
            'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'
        ]
        is_image = True
        epochs = 10
        batch_size = 128
        
        m3 = PyTorchCNN3LayerImage(in_channels=1, num_classes=num_classes, spatial_dim=28).to(DEVICE)
        m5 = PyTorchCNN5LayerImage(in_channels=1, num_classes=num_classes, spatial_dim=28).to(DEVICE)
        
    elif dataset_key == "svhn":
        out_dir = os.path.join(OUTPUT_BASE, "02_svhn")
        npz_file = os.path.join(DATA_DIR, "svhn", "svhn_data.npz")
        data = np.load(npz_file)
        # Convert from NHWC (N, 32, 32, 3) to NCHW (N, 3, 32, 32)
        x_train = np.transpose(data['x_train'].astype(np.float32) / 255.0, (0, 3, 1, 2))
        y_train = data['y_train'].astype(np.int64)
        x_test = np.transpose(data['x_test'].astype(np.float32) / 255.0, (0, 3, 1, 2))
        y_test = data['y_test'].astype(np.int64)
        
        num_classes = 10
        class_names = [f"Digit {i}" for i in range(10)]
        is_image = True
        epochs = 10
        batch_size = 128
        
        m3 = PyTorchCNN3LayerImage(in_channels=3, num_classes=num_classes, spatial_dim=32).to(DEVICE)
        m5 = PyTorchCNN5LayerImage(in_channels=3, num_classes=num_classes, spatial_dim=32).to(DEVICE)
        
    elif dataset_key == "diabetes":
        out_dir = os.path.join(OUTPUT_BASE, "03_diabetes")
        npz_file = os.path.join(DATA_DIR, "diabetes", "diabetes_data.npz")
        data = np.load(npz_file)
        x_train = np.expand_dims(data['x_train'].astype(np.float32), 1)
        y_train = data['y_train'].astype(np.int64)
        x_test = np.expand_dims(data['x_test'].astype(np.float32), 1)
        y_test = data['y_test'].astype(np.int64)
        
        num_classes = 3
        class_names = ['No Diabetes', 'Prediabetes', 'Diabetes']
        is_image = False
        epochs = 10
        batch_size = 256
        
        in_features = x_train.shape[2]
        m3 = PyTorch1DCNN3LayerTabular(in_features=in_features, num_classes=num_classes).to(DEVICE)
        m5 = PyTorch1DCNN5LayerTabular(in_features=in_features, num_classes=num_classes).to(DEVICE)
    else:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Save Architecture Summary
    arch_file = os.path.join(out_dir, "architecture_summary.txt")
    with open(arch_file, "w", encoding="utf-8") as f:
        f.write(f"=== PYTORCH ARCHITECTURE SPECIFICATION: {dataset_key.upper()} (DEVICE: {DEVICE}) ===\n\n")
        f.write(f"--- 3-LAYER CNN ({count_parameters(m3):,} params) ---\n{m3}\n\n")
        f.write(f"--- 5-LAYER CNN ({count_parameters(m5):,} params) ---\n{m5}\n")
        
    val_split = int(0.9 * len(x_train))
    x_tr, x_val = torch.tensor(x_train[:val_split]), torch.tensor(x_train[val_split:])
    y_tr, y_val = torch.tensor(y_train[:val_split]), torch.tensor(y_train[val_split:])
    x_te_tensor = torch.tensor(x_test)
    y_te_tensor = torch.tensor(y_test)
    
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_te_tensor, y_te_tensor), batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    
    # Train 3-Layer
    opt3 = optim.Adam(m3.parameters(), lr=0.001)
    h3 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    print(f"\n--- Training 3-Layer PyTorch Model ({count_parameters(m3):,} params) on {DEVICE} ---")
    t0_3 = time.time()
    for ep in range(epochs):
        tr_loss, tr_acc = train_epoch(m3, train_loader, criterion, opt3, DEVICE)
        va_loss, va_acc, _, _ = eval_epoch(m3, val_loader, criterion, DEVICE)
        h3['train_loss'].append(tr_loss); h3['train_acc'].append(tr_acc)
        h3['val_loss'].append(va_loss); h3['val_acc'].append(va_acc)
        print(f"   Epoch {ep+1:02d}/{epochs:02d} | Train Acc: {tr_acc*100:.2f}% | Val Acc: {va_acc*100:.2f}%")
    t_train_3 = time.time() - t0_3
    
    # Train 5-Layer
    opt5 = optim.Adam(m5.parameters(), lr=0.001)
    h5 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    print(f"\n--- Training 5-Layer PyTorch Model ({count_parameters(m5):,} params) on {DEVICE} ---")
    t0_5 = time.time()
    for ep in range(epochs):
        tr_loss, tr_acc = train_epoch(m5, train_loader, criterion, opt5, DEVICE)
        va_loss, va_acc, _, _ = eval_epoch(m5, val_loader, criterion, DEVICE)
        h5['train_loss'].append(tr_loss); h5['train_acc'].append(tr_acc)
        h5['val_loss'].append(va_loss); h5['val_acc'].append(va_acc)
        print(f"   Epoch {ep+1:02d}/{epochs:02d} | Train Acc: {tr_acc*100:.2f}% | Val Acc: {va_acc*100:.2f}%")
    t_train_5 = time.time() - t0_5
    
    te_loss_3, te_acc_3, preds_3, labels_3 = eval_epoch(m3, test_loader, criterion, DEVICE)
    te_loss_5, te_acc_5, preds_5, labels_5 = eval_epoch(m5, test_loader, criterion, DEVICE)
    
    f1_macro_3 = f1_score(labels_3, preds_3, average='macro')
    f1_macro_5 = f1_score(labels_5, preds_5, average='macro')
    
    cm3 = confusion_matrix(labels_3, preds_3)
    cm5 = confusion_matrix(labels_5, preds_5)
    
    m3_path = os.path.join(out_dir, "model_3layer.pth")
    m5_path = os.path.join(out_dir, "model_5layer.pth")
    torch.save(m3.state_dict(), m3_path)
    torch.save(m5.state_dict(), m5_path)
    
    with open(os.path.join(out_dir, "history_3layer.json"), "w", encoding="utf-8") as f:
        json.dump(h3, f, indent=4)
    with open(os.path.join(out_dir, "history_5layer.json"), "w", encoding="utf-8") as f:
        json.dump(h5, f, indent=4)
        
    plot_and_save_pytorch_history(h3, h5, out_dir, dataset_key.upper())
    plot_and_save_pytorch_confusion_matrices(cm3, cm5, class_names, out_dir, dataset_key.upper())
    save_sample_predictions_pytorch(m5, x_te_tensor, y_test, class_names, out_dir, is_image=is_image)
    
    metrics = {
        "dataset": dataset_key,
        "framework": "PyTorch (GPU)",
        "device": str(DEVICE),
        "models": {
            "3_layer": {
                "params": count_parameters(m3),
                "training_time_seconds": round(t_train_3, 2),
                "test_loss": round(float(te_loss_3), 4),
                "test_accuracy": round(float(te_acc_3), 4),
                "macro_f1": round(float(f1_macro_3), 4),
                "epochs": epochs,
                "batch_size": batch_size
            },
            "5_layer": {
                "params": count_parameters(m5),
                "training_time_seconds": round(t_train_5, 2),
                "test_loss": round(float(te_loss_5), 4),
                "test_accuracy": round(float(te_acc_5), 4),
                "macro_f1": round(float(f1_macro_5), 4),
                "epochs": epochs,
                "batch_size": batch_size
            }
        }
    }
    with open(os.path.join(out_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"   3-Layer: Acc = {te_acc_3:.4f}, Macro-F1 = {f1_macro_3:.4f}, Time = {t_train_3:.1f}s")
    print(f"   5-Layer: Acc = {te_acc_5:.4f}, Macro-F1 = {f1_macro_5:.4f}, Time = {t_train_5:.1f}s")
    return metrics


if __name__ == "__main__":
    for ds in ["fashion_mnist", "svhn"]:
        train_pytorch_pipeline(ds)
    # If diabetes metrics already exist and valid, keep it or re-evaluate
    diab_met = os.path.join(OUTPUT_BASE, "03_diabetes", "evaluation_metrics.json")
    if not os.path.exists(diab_met):
        train_pytorch_pipeline("diabetes")
    print("\n[SUCCESS] ALL NEW PYTORCH MODELS TRAINED!")
