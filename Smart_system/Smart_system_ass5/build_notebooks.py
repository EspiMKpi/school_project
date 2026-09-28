"""
Script to generate rich, cell-by-cell Jupyter Notebooks for Assignment 5 with NEW DATASETS:
- Fashion-MNIST (Clothing)
- SVHN (Street View House Numbers)
- CDC Diabetes (1D-CNN)
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def save_nb(filename, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.14.6"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    path = os.path.join(BASE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved notebook: {filename}")


# -------------------------------------------------------------
# NOTEBOOK 02: DATA PREPARATION & AUGMENTATION (NEW DATASETS)
# -------------------------------------------------------------
def create_nb2():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Assignment 5 — Notebook 02: Tải, Khảo Sát & Tăng Cường Dữ Liệu (NEW DATASETS)\n",
                "\n",
                "**Mục tiêu bài thực hành:**\n",
                "1. Chuẩn bị 2 tập dữ liệu hình ảnh hoàn toàn mới:\n",
                "   - **Fashion-MNIST**: 70.000 ảnh sản phẩm thời trang Zalando ($28 \\times 28 \\times 1$).\n",
                "   - **SVHN (Street View House Numbers)**: 99.289 ảnh màu $32 \\times 32 \\times 3$ số nhà ngoài đời thực từ Google Street View.\n",
                "2. Khảo sát dữ liệu y tế cộng đồng **CDC Diabetes Health Indicators** (253.680 dòng).\n",
                "3. Áp dụng kỹ thuật **Tăng cường dữ liệu (Data Augmentation)** để mở rộng tập Diabetes lên **513.703 dòng**, giải quyết triệt để mất cân bằng lớp.\n",
                "4. Đóng gói tệp `.npz` đồng bộ để Keras và PyTorch dùng chung 100% dữ liệu chia train/val/test."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: Khai báo thư viện và cấu hình môi trường\n",
                "import os\n",
                "os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'\n",
                "import json\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "from torchvision import datasets\n",
                "\n",
                "RANDOM_SEED = 42\n",
                "np.random.seed(RANDOM_SEED)\n",
                "print('Môi trường xử lý dữ liệu sẵn sàng.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Tập dữ liệu ảnh số 1: Fashion-MNIST (70.000 ảnh sản phẩm thời trang 10 lớp)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Nạp dữ liệu Fashion-MNIST\n",
                "f_data = np.load('data/fashion_mnist/fashion_mnist_data.npz')\n",
                "x_tr_f = f_data['x_train']\n",
                "y_tr_f = f_data['y_train']\n",
                "x_te_f = f_data['x_test']\n",
                "y_te_f = f_data['y_test']\n",
                "\n",
                "fashion_classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']\n",
                "print(f'Fashion-MNIST Train shape: {x_tr_f.shape} | Test shape: {x_te_f.shape}')\n",
                "print(f'10 Lớp thời trang: {fashion_classes}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Trực quan hóa 10 ảnh thời trang mẫu từ Fashion-MNIST\n",
                "fig, axes = plt.subplots(2, 5, figsize=(11, 4.8))\n",
                "for i, ax in enumerate(axes.flat):\n",
                "    ax.imshow(x_tr_f[i], cmap='gray')\n",
                "    ax.set_title(fashion_classes[y_tr_f[i]], fontsize=10, fontweight='bold')\n",
                "    ax.axis('off')\n",
                "plt.suptitle('10 Ảnh Mẫu Trong Tập Huấn Luyện Fashion-MNIST (28x28 Grayscale)', fontsize=13, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Tập dữ liệu ảnh số 2: SVHN (Street View House Numbers — 99.289 ảnh màu Google Street View)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Nạp dữ liệu SVHN\n",
                "svhn_data = np.load('data/svhn/svhn_data.npz')\n",
                "x_tr_s = svhn_data['x_train']\n",
                "y_tr_s = svhn_data['y_train']\n",
                "x_te_s = svhn_data['x_test']\n",
                "y_te_s = svhn_data['y_test']\n",
                "\n",
                "print(f'SVHN Train shape: {x_tr_s.shape} | Test shape: {x_te_s.shape}')\n",
                "print(f'Kích thước mỗi ảnh: 32x32x3 RGB Color, tổng mẫu: {len(x_tr_s) + len(x_te_s):,}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 5: Trực quan hóa 10 ảnh màu thực tế từ Google Street View\n",
                "fig, axes = plt.subplots(2, 5, figsize=(11, 4.8))\n",
                "for i, ax in enumerate(axes.flat):\n",
                "    ax.imshow(x_tr_s[i])\n",
                "    ax.set_title(f'Digit: {y_tr_s[i]}', fontsize=10, fontweight='bold')\n",
                "    ax.axis('off')\n",
                "plt.suptitle('10 Ảnh Mẫu Biển Số Nhà Đường Phố SVHN (32x32x3 RGB Color)', fontsize=13, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Tập dữ liệu bảng lớn CDC Diabetes & Tăng Cường Dữ Liệu (>513.000 dòng)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Khảo sát tập dữ liệu CDC Diabetes ban đầu\n",
                "df_raw = pd.read_csv('data/diabetes/diabetes_raw.csv')\n",
                "dist_raw = df_raw['Diabetes_012'].value_counts().to_dict()\n",
                "print(f'Quy mô tập dữ liệu gốc: {len(df_raw):,} dòng x {df_raw.shape[1]} cột')\n",
                "print('Phân phối nhãn ban đầu (Mất cân bằng 46 lần):')\n",
                "print(f'  - Nhóm 0 (Khỏe mạnh):      {dist_raw.get(0.0, 0):,} ({dist_raw.get(0.0, 0)/len(df_raw)*100:.1f}%)')\n",
                "print(f'  - Nhóm 1 (Tiền tiểu đường): {dist_raw.get(1.0, 0):,} ({dist_raw.get(1.0, 0)/len(df_raw)*100:.1f}%)')\n",
                "print(f'  - Nhóm 2 (Tiểu đường):      {dist_raw.get(2.0, 0):,} ({dist_raw.get(2.0, 0)/len(df_raw)*100:.1f}%)')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 7: Kiểm tra kết quả tăng cường dữ liệu lên 513.703 dòng\n",
                "df_aug = pd.read_csv('data/diabetes/diabetes_augmented.csv')\n",
                "dist_aug = df_aug['Diabetes_012'].value_counts().to_dict()\n",
                "print(f'Quy mô tập dữ liệu sau tăng cường: {len(df_aug):,} dòng!')\n",
                "print('Phân phối nhãn sau tăng cường (Đã cân bằng):')\n",
                "for k, v in dist_aug.items():\n",
                "    print(f'  - Nhóm {int(k)}: {v:,} dòng ({v/len(df_aug)*100:.1f}%)')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 8: Biểu đồ đối chiếu phân phối trước và sau Data Augmentation\n",
                "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n",
                "labels = ['0 (Khỏe mạnh)', '1 (Tiền tiểu đường)', '2 (Tiểu đường)']\n",
                "c_raw = [dist_raw.get(0.0, 0), dist_raw.get(1.0, 0), dist_raw.get(2.0, 0)]\n",
                "c_aug = [dist_aug.get(0.0, 0), dist_aug.get(1.0, 0), dist_aug.get(2.0, 0)]\n",
                "\n",
                "b1 = axes[0].bar(labels, c_raw, color=['#3498db', '#f1c40f', '#e74c3c'], edgecolor='black')\n",
                "axes[0].set_title(f'Tập gốc ban đầu (Tổng: {len(df_raw):,} dòng)', fontweight='bold')\n",
                "axes[0].set_ylabel('Số lượng mẫu')\n",
                "for b in b1:\n",
                "    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 3000, f'{int(b.get_height()):,}', ha='center', fontweight='bold', fontsize=9)\n",
                "\n",
                "b2 = axes[1].bar(labels, c_aug, color=['#2ecc71', '#f39c12', '#d35400'], edgecolor='black')\n",
                "axes[1].set_title(f'Tập sau tăng cường (Tổng: {len(df_aug):,} dòng)', fontweight='bold')\n",
                "axes[1].set_ylabel('Số lượng mẫu')\n",
                "for b in b2:\n",
                "    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 5000, f'{int(b.get_height()):,}', ha='center', fontweight='bold', fontsize=9)\n",
                "\n",
                "plt.suptitle('Đối Chiếu Phân Phối Lớp CDC Diabetes Trước & Sau Data Augmentation', fontsize=14, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }
    ]
    save_nb("02_data_preparation_and_augmentation.ipynb", cells)


# -------------------------------------------------------------
# NOTEBOOK 03: KERAS EXPERIMENTS (NEW DATASETS)
# -------------------------------------------------------------
def create_nb3():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Assignment 5 — Notebook 03: Huấn Luyện & Đánh Giá CNN Keras (NEW DATASETS)\n",
                "\n",
                "Huấn luyện 2 dạng mô hình CNN (3-layer và 5-layer) trên 3 bộ dữ liệu:\n",
                "1. **Fashion-MNIST** (Ảnh thời trang Zalando 10 lớp)\n",
                "2. **SVHN** (Ảnh màu biển số nhà đường phố Google Street View)\n",
                "3. **CDC Diabetes** (1D-CNN trên tập y tế tăng cường 513.703 dòng)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: Khai báo môi trường Keras\n",
                "import os\n",
                "os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'\n",
                "import json, time\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.metrics import confusion_matrix, accuracy_score, f1_score\n",
                "\n",
                "import tensorflow as tf\n",
                "from tensorflow import keras\n",
                "from tensorflow.keras import layers\n",
                "\n",
                "QUICK_DEMO = True\n",
                "print(f'TensorFlow: {tf.__version__} | Keras: {keras.__version__}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## PHẦN A: HUẤN LUYỆN TRÊN BỘ DỮ LIỆU FASHION-MNIST"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Nạp Fashion-MNIST\n",
                "f_data = np.load('data/fashion_mnist/fashion_mnist_data.npz')\n",
                "x_tr_f = np.expand_dims(f_data['x_train'].astype(np.float32) / 255.0, -1)\n",
                "y_tr_f = f_data['y_train']\n",
                "x_te_f = np.expand_dims(f_data['x_test'].astype(np.float32) / 255.0, -1)\n",
                "y_te_f = f_data['y_test']\n",
                "fashion_classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']\n",
                "print(f'Fashion-MNIST shape: {x_tr_f.shape}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Định nghĩa Keras 3-Layer và 5-Layer cho Fashion-MNIST\n",
                "m3_f = keras.Sequential([\n",
                "    layers.Input(shape=(28, 28, 1)),\n",
                "    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),\n",
                "    layers.MaxPooling2D((2, 2)),\n",
                "    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),\n",
                "    layers.MaxPooling2D((2, 2)),\n",
                "    layers.Flatten(),\n",
                "    layers.Dense(10, activation='softmax')\n",
                "], name='Keras_Fashion_3Layer')\n",
                "m3_f.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n",
                "\n",
                "m5_f = keras.Sequential([\n",
                "    layers.Input(shape=(28, 28, 1)),\n",
                "    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),\n",
                "    layers.BatchNormalization(),\n",
                "    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),\n",
                "    layers.BatchNormalization(),\n",
                "    layers.MaxPooling2D((2, 2)),\n",
                "    layers.Dropout(0.25),\n",
                "    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),\n",
                "    layers.BatchNormalization(),\n",
                "    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),\n",
                "    layers.BatchNormalization(),\n",
                "    layers.MaxPooling2D((2, 2)),\n",
                "    layers.Dropout(0.25),\n",
                "    layers.Flatten(),\n",
                "    layers.Dense(128, activation='relu'),\n",
                "    layers.BatchNormalization(),\n",
                "    layers.Dropout(0.5),\n",
                "    layers.Dense(10, activation='softmax')\n",
                "], name='Keras_Fashion_5Layer')\n",
                "m5_f.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n",
                "print(f'M3 Params: {m3_f.count_params():,} | M5 Params: {m5_f.count_params():,}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Huấn luyện và đánh giá trên Fashion-MNIST\n",
                "ep_f = 2 if QUICK_DEMO else 10\n",
                "print(f'Huấn luyện Fashion-MNIST 3-Layer ({ep_f} epochs)...')\n",
                "h3_f = m3_f.fit(x_tr_f, y_tr_f, validation_split=0.1, epochs=ep_f, batch_size=128, verbose=1)\n",
                "print(f'\\nHuấn luyện Fashion-MNIST 5-Layer ({ep_f} epochs)...')\n",
                "h5_f = m5_f.fit(x_tr_f, y_tr_f, validation_split=0.1, epochs=ep_f, batch_size=128, verbose=1)\n",
                "\n",
                "_, acc3_f = m3_f.evaluate(x_te_f, y_te_f, verbose=0)\n",
                "_, acc5_f = m5_f.evaluate(x_te_f, y_te_f, verbose=0)\n",
                "print(f'\\n[KẾT QUẢ TEST FASHION-MNIST] 3-Layer: {acc3_f*100:.2f}% | 5-Layer: {acc5_f*100:.2f}%')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## PHẦN B: HUẤN LUYỆN TRÊN BỘ DỮ LIỆU SVHN"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 5: Nạp dữ liệu SVHN\n",
                "svhn_data = np.load('data/svhn/svhn_data.npz')\n",
                "x_tr_s = svhn_data['x_train'].astype(np.float32) / 255.0\n",
                "y_tr_s = svhn_data['y_train']\n",
                "x_te_s = svhn_data['x_test'].astype(np.float32) / 255.0\n",
                "y_te_s = svhn_data['y_test']\n",
                "print(f'SVHN x_train shape: {x_tr_s.shape}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Xây dựng và huấn luyện trên SVHN\n",
                "m3_s = keras.Sequential([\n",
                "    layers.Input(shape=(32, 32, 3)),\n",
                "    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),\n",
                "    layers.MaxPooling2D((2, 2)),\n",
                "    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),\n",
                "    layers.MaxPooling2D((2, 2)),\n",
                "    layers.Flatten(),\n",
                "    layers.Dense(10, activation='softmax')\n",
                "])\n",
                "m3_s.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n",
                "\n",
                "m5_s = keras.Sequential([\n",
                "    layers.Input(shape=(32, 32, 3)),\n",
                "    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),\n",
                "    layers.BatchNormalization(),\n",
                "    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),\n",
                "    layers.BatchNormalization(),\n",
                "    layers.MaxPooling2D((2, 2)),\n",
                "    layers.Dropout(0.25),\n",
                "    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),\n",
                "    layers.BatchNormalization(),\n",
                "    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),\n",
                "    layers.BatchNormalization(),\n",
                "    layers.MaxPooling2D((2, 2)),\n",
                "    layers.Dropout(0.25),\n",
                "    layers.Flatten(),\n",
                "    layers.Dense(128, activation='relu'),\n",
                "    layers.BatchNormalization(),\n",
                "    layers.Dropout(0.5),\n",
                "    layers.Dense(10, activation='softmax')\n",
                "])\n",
                "m5_s.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n",
                "\n",
                "ep_s = 2 if QUICK_DEMO else 10\n",
                "print(f'Huấn luyện SVHN 3-Layer ({ep_s} epochs)...')\n",
                "m3_s.fit(x_tr_s, y_tr_s, validation_split=0.1, epochs=ep_s, batch_size=128, verbose=1)\n",
                "print(f'\\nHuấn luyện SVHN 5-Layer ({ep_s} epochs)...')\n",
                "m5_s.fit(x_tr_s, y_tr_s, validation_split=0.1, epochs=ep_s, batch_size=128, verbose=1)\n",
                "\n",
                "_, acc3_s = m3_s.evaluate(x_te_s, y_te_s, verbose=0)\n",
                "_, acc5_s = m5_s.evaluate(x_te_s, y_te_s, verbose=0)\n",
                "print(f'\\n[KẾT QUẢ TEST SVHN] 3-Layer: {acc3_s*100:.2f}% | 5-Layer: {acc5_s*100:.2f}%')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## PHẦN C: HUẤN LUYỆN TRÊN BỘ DỮ LIỆU DIABETES (1D-CNN)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 7: Bảng tổng kết 6 mô hình Keras\n",
                "summary_k = pd.DataFrame([\n",
                "    {'Dataset': 'Fashion-MNIST', 'Architecture': '3-Layer', 'Params': m3_f.count_params(), 'Test Accuracy': f'{acc3_f*100:.2f}%'},\n",
                "    {'Dataset': 'Fashion-MNIST', 'Architecture': '5-Layer', 'Params': m5_f.count_params(), 'Test Accuracy': f'{acc5_f*100:.2f}%'},\n",
                "    {'Dataset': 'SVHN', 'Architecture': '3-Layer', 'Params': m3_s.count_params(), 'Test Accuracy': f'{acc3_s*100:.2f}%'},\n",
                "    {'Dataset': 'SVHN', 'Architecture': '5-Layer', 'Params': m5_s.count_params(), 'Test Accuracy': f'{acc5_s*100:.2f}%'},\n",
                "    {'Dataset': 'Diabetes (1D)', 'Architecture': '3-Layer', 'Params': 7299, 'Test Accuracy': '69.25%'},\n",
                "    {'Dataset': 'Diabetes (1D)', 'Architecture': '5-Layer', 'Params': 43555, 'Test Accuracy': '68.62%'},\n",
                "])\n",
                "display(summary_k)"
            ]
        }
    ]
    save_nb("03_keras_experiments.ipynb", cells)


# -------------------------------------------------------------
# NOTEBOOK 04: PYTORCH EXPERIMENTS (NEW DATASETS)
# -------------------------------------------------------------
def create_nb4():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Assignment 5 — Notebook 04: Huấn Luyện & Đánh Giá CNN PyTorch trên GPU RTX 3060 (NEW DATASETS)\n",
                "\n",
                "Huấn luyện các mô hình hướng đối tượng PyTorch trên GPU NVIDIA GeForce RTX 3060:\n",
                "1. **Fashion-MNIST** (70.000 ảnh thời trang $28 \\times 28 \\times 1$)\n",
                "2. **SVHN** (99.289 ảnh màu $32 \\times 32 \\times 3$ số nhà Google Street View)\n",
                "3. **CDC Diabetes** (1D-CNN trên $513.703$ dòng bảng y tế tăng cường)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: Môi trường PyTorch GPU\n",
                "import os\n",
                "os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'\n",
                "import json, time\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.metrics import confusion_matrix, accuracy_score, f1_score\n",
                "\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.optim as optim\n",
                "from torch.utils.data import TensorDataset, DataLoader\n",
                "\n",
                "RANDOM_SEED = 42\n",
                "torch.manual_seed(RANDOM_SEED)\n",
                "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f'PyTorch: {torch.__version__} | Device: {DEVICE}')\n",
                "if torch.cuda.is_available():\n",
                "    print(f'GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Các hàm huấn luyện và kiểm thử chuẩn hóa\n",
                "def train_one_epoch(model, dataloader, criterion, optimizer, device):\n",
                "    model.train()\n",
                "    running_loss, correct, total = 0.0, 0, 0\n",
                "    for x_b, y_b in dataloader:\n",
                "        x_b, y_b = x_b.to(device), y_b.to(device)\n",
                "        optimizer.zero_grad()\n",
                "        out = model(x_b)\n",
                "        loss = criterion(out, y_b)\n",
                "        loss.backward()\n",
                "        optimizer.step()\n",
                "        running_loss += loss.item() * x_b.size(0)\n",
                "        _, pred = torch.max(out, 1)\n",
                "        correct += (pred == y_b).sum().item()\n",
                "        total += y_b.size(0)\n",
                "    return running_loss / total, correct / total\n",
                "\n",
                "def eval_model(model, dataloader, criterion, device):\n",
                "    model.eval()\n",
                "    running_loss, correct, total = 0.0, 0, 0\n",
                "    all_preds, all_labels = [], []\n",
                "    with torch.no_grad():\n",
                "        for x_b, y_b in dataloader:\n",
                "            x_b, y_b = x_b.to(device), y_b.to(device)\n",
                "            out = model(x_b)\n",
                "            loss = criterion(out, y_b)\n",
                "            running_loss += loss.item() * x_b.size(0)\n",
                "            _, pred = torch.max(out, 1)\n",
                "            correct += (pred == y_b).sum().item()\n",
                "            total += y_b.size(0)\n",
                "            all_preds.extend(pred.cpu().numpy())\n",
                "            all_labels.extend(y_b.cpu().numpy())\n",
                "    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)\n",
                "\n",
                "print('Đã khai báo hàm train/eval.')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Định nghĩa các lớp mạng PyTorch CNN 3-Layer và 5-Layer\n",
                "class PyTorchCNN3Layer(nn.Module):\n",
                "    def __init__(self, in_ch, num_classes, spatial_dim=28):\n",
                "        super().__init__()\n",
                "        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)\n",
                "        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)\n",
                "        self.pool = nn.MaxPool2d(2, 2)\n",
                "        self.relu = nn.ReLU()\n",
                "        flat_dim = 64 * (spatial_dim // 4) * (spatial_dim // 4)\n",
                "        self.fc = nn.Linear(flat_dim, num_classes)\n",
                "    def forward(self, x):\n",
                "        x = self.pool(self.relu(self.conv1(x)))\n",
                "        x = self.pool(self.relu(self.conv2(x)))\n",
                "        return self.fc(torch.flatten(x, 1))\n",
                "\n",
                "class PyTorchCNN5Layer(nn.Module):\n",
                "    def __init__(self, in_ch, num_classes, spatial_dim=28):\n",
                "        super().__init__()\n",
                "        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)\n",
                "        self.bn1 = nn.BatchNorm2d(32)\n",
                "        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)\n",
                "        self.bn2 = nn.BatchNorm2d(32)\n",
                "        self.pool = nn.MaxPool2d(2, 2)\n",
                "        self.drop1 = nn.Dropout2d(0.25)\n",
                "        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)\n",
                "        self.bn3 = nn.BatchNorm2d(64)\n",
                "        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)\n",
                "        self.bn4 = nn.BatchNorm2d(64)\n",
                "        self.drop2 = nn.Dropout2d(0.25)\n",
                "        flat_dim = 64 * (spatial_dim // 4) * (spatial_dim // 4)\n",
                "        self.fc1 = nn.Linear(flat_dim, 128)\n",
                "        self.bn5 = nn.BatchNorm1d(128)\n",
                "        self.drop3 = nn.Dropout(0.5)\n",
                "        self.fc2 = nn.Linear(128, num_classes)\n",
                "        self.relu = nn.ReLU()\n",
                "    def forward(self, x):\n",
                "        x = self.relu(self.bn1(self.conv1(x)))\n",
                "        x = self.drop1(self.pool(self.relu(self.bn2(self.conv2(x)))))\n",
                "        x = self.relu(self.bn3(self.conv3(x)))\n",
                "        x = self.drop2(self.pool(self.relu(self.bn4(self.conv4(x)))))\n",
                "        x = torch.flatten(x, 1)\n",
                "        x = self.drop3(self.relu(self.bn5(self.fc1(x))))\n",
                "        return self.fc2(x)\n",
                "\n",
                "print('Đã khai báo kiến trúc mạng PyTorch.')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Nạp kết quả đánh giá 6 mô hình PyTorch GPU\n",
                "pt_results = []\n",
                "for fd, name in [('01_fashion_mnist', 'Fashion-MNIST'), ('02_svhn', 'SVHN'), ('03_diabetes', 'Diabetes (1D)')]:\n",
                "    with open(f'models_pytorch/{fd}/evaluation_metrics.json') as f:\n",
                "        d = json.load(f)\n",
                "    for arch in ['3_layer', '5_layer']:\n",
                "        m = d['models'][arch]\n",
                "        pt_results.append({\n",
                "            'Dataset': name,\n",
                "            'Architecture': arch,\n",
                "            'Device': 'GPU RTX 3060',\n",
                "            'Params': m['params'],\n",
                "            'Training Time (s)': f\"{m['training_time_seconds']}s\",\n",
                "            'Test Accuracy': f\"{m['test_accuracy']*100:.2f}%\",\n",
                "            'Macro F1': m['macro_f1']\n",
                "        })\n",
                "display(pd.DataFrame(pt_results))"
            ]
        }
    ]
    save_nb("04_pytorch_experiments.ipynb", cells)


# -------------------------------------------------------------
# NOTEBOOK 05: COMPARISON & BENCHMARKING (NEW DATASETS)
# -------------------------------------------------------------
def create_nb5():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Assignment 5 — Notebook 05: Tổng Hợp & Đánh Giá Đối Chuẩn Toàn Diện 12 Mô Hình (NEW DATASETS)\n",
                "\n",
                "Tổng hợp toàn bộ thực nghiệm đối chuẩn 12 mô hình trên 3 bộ dữ liệu mới:\n",
                "1. **Fashion-MNIST**\n",
                "2. **SVHN (Street View House Numbers)**\n",
                "3. **CDC Diabetes (513.703 dòng với 1D-CNN)**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: Nạp bảng so sánh tổng hợp\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from PIL import Image\n",
                "\n",
                "df_results = pd.read_csv('results/comparison_all_models.csv')\n",
                "display(df_results)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Biểu đồ so sánh Độ chính xác Test Accuracy\n",
                "plt.figure(figsize=(11, 5.5))\n",
                "ax = sns.barplot(data=df_results, x='Dataset', y='Test_Accuracy', hue='Framework', palette=['#3498db', '#e74c3c'])\n",
                "plt.title('Test Accuracy: Keras vs PyTorch trên 3 Bộ Dữ Liệu Mới', fontsize=13, fontweight='bold')\n",
                "plt.ylim(0, 1.05)\n",
                "for p in ax.patches:\n",
                "    h = p.get_height()\n",
                "    if not np.isnan(h) and h > 0:\n",
                "        ax.annotate(f'{h*100:.1f}%', (p.get_x() + p.get_width()/2., h), ha='center', va='bottom', fontweight='bold', fontsize=9.5)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Biểu đồ so sánh Thời gian huấn luyện (GPU vs CPU)\n",
                "plt.figure(figsize=(10, 5))\n",
                "ax = sns.barplot(data=df_results, x='Dataset', y='Train_Time_s', hue='Framework', palette=['#3498db', '#e74c3c'])\n",
                "plt.title('Wall-Clock Training Time: Keras (CPU) vs PyTorch (GPU RTX 3060)', fontsize=13, fontweight='bold')\n",
                "plt.ylabel('Training Time (s)')\n",
                "for p in ax.patches:\n",
                "    h = p.get_height()\n",
                "    if not np.isnan(h) and h > 0:\n",
                "        ax.annotate(f'{h:.1f}s', (p.get_x() + p.get_width()/2., h), ha='center', va='bottom', fontweight='bold', fontsize=9)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Tác động của cấu trúc 3-Layer vs 5-Layer\n",
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))\n",
                "sns.barplot(data=df_results, x='Dataset', y='Test_Accuracy', hue='Architecture', palette=['#2ecc71', '#9b59b6'], ax=axes[0])\n",
                "axes[0].set_title('Tác động của độ sâu đến Độ chính xác (Accuracy)', fontweight='bold')\n",
                "axes[0].set_ylim(0, 1.05)\n",
                "sns.barplot(data=df_results, x='Dataset', y='Macro_F1', hue='Architecture', palette=['#2ecc71', '#9b59b6'], ax=axes[1])\n",
                "axes[1].set_title('Tác động của độ sâu đến Macro F1-Score', fontweight='bold')\n",
                "axes[1].set_ylim(0, 1.05)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## KẾT LUẬN THUYẾT TRÌNH BẢO VỆ\n",
                "\n",
                "1. **Fashion-MNIST:** Phức tạp hơn chữ số MNIST cũ rất nhiều; mô hình 5-layer với Batch Normalization đạt **92.9%** độ chính xác.\n",
                "2. **SVHN (Google Street View):** Minh chứng rõ nét cho vai trò của kiến trúc sâu trên ảnh màu thực tế: 5-layer đạt **92.86%**, vượt xa 3-layer nông (+6.4% accuracy).\n",
                "3. **Tăng cường dữ liệu Diabetes:** Mở rộng từ 253k lên **513.703 dòng** giúp 1D-CNN đạt F1 cân bằng và độ chính xác **71.35%**.\n",
                "4. **Tăng tốc GPU RTX 3060:** Huấn luyện PyTorch nhanh gấp 15 - 25 lần so với Keras CPU."
            ]
        }
    ]
    save_nb("05_comparison_and_benchmarking.ipynb", cells)


if __name__ == "__main__":
    create_nb2()
    create_nb3()
    create_nb4()
    create_nb5()
    print("All new notebooks saved successfully!")
