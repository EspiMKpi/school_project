"""
Prepare NEW Image Datasets for Assignment 5:
1. Fashion-MNIST (70,000 fashion articles, 28x28 grayscale, 10 classes)
2. SVHN (Street View House Numbers, 99,289 real-world color images 32x32x3, 10 classes)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FASHION_DIR = os.path.join(DATA_DIR, "fashion_mnist")
SVHN_DIR = os.path.join(DATA_DIR, "svhn")

def prepare_fashion_mnist():
    print("=" * 60)
    print("1. Preparing NEW Dataset: Fashion-MNIST...")
    os.makedirs(FASHION_DIR, exist_ok=True)
    raw_dir = os.path.join(FASHION_DIR, "raw")
    
    train_set = datasets.FashionMNIST(root=raw_dir, train=True, download=True)
    test_set = datasets.FashionMNIST(root=raw_dir, train=False, download=True)
    
    x_train = train_set.data.numpy()
    y_train = train_set.targets.numpy()
    x_test = test_set.data.numpy()
    y_test = test_set.targets.numpy()
    
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    npz_path = os.path.join(FASHION_DIR, "fashion_mnist_data.npz")
    np.savez_compressed(npz_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print(f"   [+] Saved Fashion-MNIST archive: {npz_path}")
    print(f"   Train samples: {x_train.shape}, Test samples: {x_test.shape}")
    
    # Save visual presentation samples
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_train[i], cmap='gray')
        ax.set_title(class_names[y_train[i]], fontsize=10, fontweight='bold')
        ax.axis('off')
    plt.suptitle("Fashion-MNIST Samples (Zalando Clothing Articles 28x28 Grayscale)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    sample_img_path = os.path.join(FASHION_DIR, "sample_fashion.png")
    plt.savefig(sample_img_path, dpi=200)
    plt.close()
    print(f"   [+] Saved presentation sample: {sample_img_path}")
    
    info = {
        "dataset_name": "Fashion-MNIST",
        "type": "2D Grayscale Images",
        "input_shape": [28, 28, 1],
        "num_classes": 10,
        "classes": class_names,
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "archive_file": "fashion_mnist_data.npz",
        "sample_figure": "sample_fashion.png"
    }
    with open(os.path.join(FASHION_DIR, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
    print("   [+] Fashion-MNIST preparation complete.")


def prepare_svhn():
    print("=" * 60)
    print("2. Preparing NEW Dataset: SVHN (Street View House Numbers)...")
    os.makedirs(SVHN_DIR, exist_ok=True)
    raw_dir = os.path.join(SVHN_DIR, "raw")
    
    train_set = datasets.SVHN(root=raw_dir, split='train', download=True)
    test_set = datasets.SVHN(root=raw_dir, split='test', download=True)
    
    # SVHN data shape: (N, 3, 32, 32), targets in [0..9] (digit '0' is labeled as 0 in torchvision SVHN)
    # Convert to NHWC (N, 32, 32, 3) for standard format
    x_train = np.transpose(train_set.data, (0, 2, 3, 1))
    y_train = train_set.labels.astype(np.int64)
    x_test = np.transpose(test_set.data, (0, 2, 3, 1))
    y_test = test_set.labels.astype(np.int64)
    
    class_names = [f"Digit {i}" for i in range(10)]
    
    npz_path = os.path.join(SVHN_DIR, "svhn_data.npz")
    np.savez_compressed(npz_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print(f"   [+] Saved SVHN archive: {npz_path}")
    print(f"   Train samples: {x_train.shape}, Test samples: {x_test.shape}")
    
    # Save visual presentation samples
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_train[i])
        ax.set_title(f"Label: {y_train[i]}", fontsize=10, fontweight='bold')
        ax.axis('off')
    plt.suptitle("SVHN Samples (Real-world House Numbers 32x32x3 RGB Color)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    sample_img_path = os.path.join(SVHN_DIR, "sample_svhn.png")
    plt.savefig(sample_img_path, dpi=200)
    plt.close()
    print(f"   [+] Saved presentation sample: {sample_img_path}")
    
    info = {
        "dataset_name": "SVHN (Street View House Numbers)",
        "type": "3D RGB Color Images",
        "input_shape": [32, 32, 3],
        "num_classes": 10,
        "classes": class_names,
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "archive_file": "svhn_data.npz",
        "sample_figure": "sample_svhn.png"
    }
    with open(os.path.join(SVHN_DIR, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
    print("   [+] SVHN preparation complete.")


if __name__ == "__main__":
    prepare_fashion_mnist()
    prepare_svhn()
    print("\n" + "=" * 60)
    print("[SUCCESS] BOTH NEW IMAGE DATASETS DOWNLOADED AND SAVED!")
    print("=" * 60)
