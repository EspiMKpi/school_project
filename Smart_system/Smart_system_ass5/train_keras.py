"""
Assignment 5 - Keras Training Pipeline (NEW DATASETS)
Trains 3-layer and 5-layer CNNs across all 3 benchmark datasets:
1. Fashion-MNIST (Zalando 10 classes, 28x28 grayscale)
2. SVHN (Street View House Numbers 10 classes, 32x32x3 RGB)
3. Diabetes (1D-CNN on 513,703 rows)

Outputs saved into distinct folders:
- models_keras/01_fashion_mnist/
- models_keras/02_svhn/
- models_keras/03_diabetes/
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_BASE = os.path.join(BASE_DIR, "models_keras")

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def build_keras_3layer_image(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Flatten(name='flatten'),
        layers.Dense(num_classes, activation='softmax', name='output_dense')
    ], name='Keras_CNN_3Layer_Image')
    return model


def build_keras_5layer_image(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.25, name='drop1'),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4'),
        layers.BatchNormalization(name='bn4'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.25, name='drop2'),
        
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense_feat'),
        layers.BatchNormalization(name='bn5'),
        layers.Dropout(0.5, name='drop3'),
        layers.Dense(num_classes, activation='softmax', name='output_dense')
    ], name='Keras_CNN_5Layer_Image')
    return model


def build_keras_3layer_tabular(input_features, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_features, 1)),
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', name='conv1d_1'),
        layers.MaxPooling1D(pool_size=2, name='pool1d_1'),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv1d_2'),
        layers.MaxPooling1D(pool_size=2, name='pool1d_2'),
        layers.Flatten(name='flatten'),
        layers.Dense(num_classes, activation='softmax', name='output_dense')
    ], name='Keras_1DCNN_3Layer_Tabular')
    return model


def build_keras_5layer_tabular(input_features, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_features, 1)),
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', name='conv1d_1'),
        layers.BatchNormalization(name='bn1'),
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', name='conv1d_2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling1D(pool_size=2, name='pool1d_1'),
        layers.Dropout(0.2, name='drop1'),
        
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv1d_3'),
        layers.BatchNormalization(name='bn3'),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv1d_4'),
        layers.BatchNormalization(name='bn4'),
        layers.MaxPooling1D(pool_size=2, name='pool1d_2'),
        layers.Dropout(0.2, name='drop2'),
        
        layers.Flatten(name='flatten'),
        layers.Dense(64, activation='relu', name='dense_feat'),
        layers.BatchNormalization(name='bn5'),
        layers.Dropout(0.3, name='drop3'),
        layers.Dense(num_classes, activation='softmax', name='output_dense')
    ], name='Keras_1DCNN_5Layer_Tabular')
    return model


def plot_and_save_history(h3, h5, out_dir, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(h3['accuracy'], label='3-Layer Train Acc', color='#3498db', linestyle='--')
    axes[0].plot(h3['val_accuracy'], label='3-Layer Val Acc', color='#2980b9', linewidth=2)
    axes[0].plot(h5['accuracy'], label='5-Layer Train Acc', color='#e74c3c', linestyle='--')
    axes[0].plot(h5['val_accuracy'], label='5-Layer Val Acc', color='#c0392b', linewidth=2)
    axes[0].set_title(f"{dataset_name} — Accuracy over Epochs", fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(h3['loss'], label='3-Layer Train Loss', color='#3498db', linestyle='--')
    axes[1].plot(h3['val_loss'], label='3-Layer Val Loss', color='#2980b9', linewidth=2)
    axes[1].plot(h5['loss'], label='5-Layer Train Loss', color='#e74c3c', linestyle='--')
    axes[1].plot(h5['val_loss'], label='5-Layer Val Loss', color='#c0392b', linewidth=2)
    axes[1].set_title(f"{dataset_name} — Loss over Epochs", fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Keras Training Curves: 3-Layer vs 5-Layer on {dataset_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart_path = os.path.join(out_dir, "training_history.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    return chart_path


def plot_and_save_confusion_matrices(cm3, cm5, class_names, out_dir, dataset_name):
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
    
    plt.suptitle(f"Confusion Matrices: Keras 3-Layer vs 5-Layer on {dataset_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    return cm_path


def save_sample_predictions_visual(model, x_test, y_test, class_names, out_dir, is_image=True):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
    sample_indices = np.random.RandomState(42).choice(len(x_test), 10, replace=False)
    preds = model.predict(x_test[sample_indices], verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    
    for i, idx in enumerate(sample_indices):
        ax = axes.flat[i]
        true_cls = class_names[y_test[idx]]
        pred_cls = class_names[pred_labels[i]]
        conf = confidences[i] * 100
        is_correct = (y_test[idx] == pred_labels[i])
        color = 'green' if is_correct else 'red'
        
        if is_image:
            img = x_test[idx]
            if img.shape[-1] == 1:
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                ax.imshow(img)
            ax.axis('off')
        else:
            ax.barh(range(min(len(class_names), 5)), preds[i][:5], color=color, alpha=0.7)
            ax.set_yticks(range(min(len(class_names), 5)))
            ax.set_yticklabels(class_names[:5], fontsize=8)
            ax.set_xlim(0, 1.0)
            
        ax.set_title(f"True: {true_cls}\nPred: {pred_cls} ({conf:.1f}%)",
                     fontsize=8.5, fontweight='bold', color=color)
                     
    plt.suptitle("Sample Predictions Showcase (Keras 5-Layer Model)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    pred_path = os.path.join(out_dir, "sample_predictions.png")
    plt.savefig(pred_path, dpi=200)
    plt.close()
    return pred_path


def train_keras_pipeline(dataset_key):
    print("\n" + "=" * 70)
    print(f"RUNNING KERAS PIPELINE FOR: {dataset_key.upper()}")
    print("=" * 70)
    
    if dataset_key == "fashion_mnist":
        out_dir = os.path.join(OUTPUT_BASE, "01_fashion_mnist")
        npz_file = os.path.join(DATA_DIR, "fashion_mnist", "fashion_mnist_data.npz")
        data = np.load(npz_file)
        x_train = np.expand_dims(data['x_train'].astype(np.float32) / 255.0, -1)
        y_train = data['y_train']
        x_test = np.expand_dims(data['x_test'].astype(np.float32) / 255.0, -1)
        y_test = data['y_test']
        
        num_classes = 10
        class_names = [
            'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'
        ]
        input_shape = (28, 28, 1)
        is_image = True
        epochs = 10
        batch_size = 128
        
        m3 = build_keras_3layer_image(input_shape, num_classes)
        m5 = build_keras_5layer_image(input_shape, num_classes)
        
    elif dataset_key == "svhn":
        out_dir = os.path.join(OUTPUT_BASE, "02_svhn")
        npz_file = os.path.join(DATA_DIR, "svhn", "svhn_data.npz")
        data = np.load(npz_file)
        x_train = data['x_train'].astype(np.float32) / 255.0
        y_train = data['y_train']
        x_test = data['x_test'].astype(np.float32) / 255.0
        y_test = data['y_test']
        
        num_classes = 10
        class_names = [f"Digit {i}" for i in range(10)]
        input_shape = (32, 32, 3)
        is_image = True
        epochs = 10
        batch_size = 128
        
        m3 = build_keras_3layer_image(input_shape, num_classes)
        m5 = build_keras_5layer_image(input_shape, num_classes)
        
    elif dataset_key == "diabetes":
        out_dir = os.path.join(OUTPUT_BASE, "03_diabetes")
        npz_file = os.path.join(DATA_DIR, "diabetes", "diabetes_data.npz")
        data = np.load(npz_file)
        x_train = np.expand_dims(data['x_train'].astype(np.float32), -1)
        y_train = data['y_train']
        x_test = np.expand_dims(data['x_test'].astype(np.float32), -1)
        y_test = data['y_test']
        
        num_classes = 3
        class_names = ['No Diabetes', 'Prediabetes', 'Diabetes']
        input_features = x_train.shape[1]
        is_image = False
        epochs = 10
        batch_size = 256
        
        m3 = build_keras_3layer_tabular(input_features, num_classes)
        m5 = build_keras_5layer_tabular(input_features, num_classes)
    else:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Save Architecture Summaries
    arch_summary_file = os.path.join(out_dir, "architecture_summary.txt")
    with open(arch_summary_file, "w", encoding="utf-8") as f:
        f.write(f"=== KERAS ARCHITECTURE SPECIFICATION: {dataset_key.upper()} ===\n\n")
        f.write("--- 3-LAYER CNN ARCHITECTURE ---\n")
        m3.summary(print_fn=lambda x: f.write(x + "\n"))
        f.write("\n\n--- 5-LAYER CNN ARCHITECTURE ---\n")
        m5.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"   [+] Saved architecture summary: {arch_summary_file}")
    
    # Compile
    m3.compile(optimizer=keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    m5.compile(optimizer=keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train 3-Layer Model
    print(f"\n--- Training 3-Layer Keras Model ({m3.count_params():,} params) ---")
    t0_3 = time.time()
    history_3 = m3.fit(x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=1)
    t_train_3 = time.time() - t0_3
    
    # Train 5-Layer Model
    print(f"\n--- Training 5-Layer Keras Model ({m5.count_params():,} params) ---")
    t0_5 = time.time()
    history_5 = m5.fit(x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=1)
    t_train_5 = time.time() - t0_5
    
    # Evaluate
    print("\n--- Evaluating Models on Unseen Test Set ---")
    test_loss_3, test_acc_3 = m3.evaluate(x_test, y_test, verbose=0)
    test_loss_5, test_acc_5 = m5.evaluate(x_test, y_test, verbose=0)
    
    y_pred_3 = np.argmax(m3.predict(x_test, verbose=0), axis=1)
    y_pred_5 = np.argmax(m5.predict(x_test, verbose=0), axis=1)
    
    f1_macro_3 = f1_score(y_test, y_pred_3, average='macro')
    f1_macro_5 = f1_score(y_test, y_pred_5, average='macro')
    
    cm3 = confusion_matrix(y_test, y_pred_3)
    cm5 = confusion_matrix(y_test, y_pred_5)
    
    # Save Model Weights (.keras)
    m3_path = os.path.join(out_dir, "model_3layer.keras")
    m5_path = os.path.join(out_dir, "model_5layer.keras")
    m3.save(m3_path)
    m5.save(m5_path)
    print(f"   [+] Saved models: {m3_path}, {m5_path}")
    
    # Save History (.json)
    h3_dict = {k: [float(val) for val in v] for k, v in history_3.history.items()}
    h5_dict = {k: [float(val) for val in v] for k, v in history_5.history.items()}
    with open(os.path.join(out_dir, "history_3layer.json"), "w", encoding="utf-8") as f:
        json.dump(h3_dict, f, indent=4)
    with open(os.path.join(out_dir, "history_5layer.json"), "w", encoding="utf-8") as f:
        json.dump(h5_dict, f, indent=4)
        
    plot_and_save_history(h3_dict, h5_dict, out_dir, dataset_key.upper())
    plot_and_save_confusion_matrices(cm3, cm5, class_names, out_dir, dataset_key.upper())
    save_sample_predictions_visual(m5, x_test, y_test, class_names, out_dir, is_image=is_image)
    
    metrics = {
        "dataset": dataset_key,
        "framework": "Keras",
        "models": {
            "3_layer": {
                "params": int(m3.count_params()),
                "training_time_seconds": round(t_train_3, 2),
                "test_loss": round(float(test_loss_3), 4),
                "test_accuracy": round(float(test_acc_3), 4),
                "macro_f1": round(float(f1_macro_3), 4),
                "epochs": epochs,
                "batch_size": batch_size
            },
            "5_layer": {
                "params": int(m5.count_params()),
                "training_time_seconds": round(t_train_5, 2),
                "test_loss": round(float(test_loss_5), 4),
                "test_accuracy": round(float(test_acc_5), 4),
                "macro_f1": round(float(f1_macro_5), 4),
                "epochs": epochs,
                "batch_size": batch_size
            }
        }
    }
    metrics_file = os.path.join(out_dir, "evaluation_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"   [+] Saved metrics record: {metrics_file}")
    
    print("\n   [SUMMARY RESULTS - KERAS]")
    print(f"   3-Layer: Acc = {test_acc_3:.4f}, Macro-F1 = {f1_macro_3:.4f}, Time = {t_train_3:.1f}s")
    print(f"   5-Layer: Acc = {test_acc_5:.4f}, Macro-F1 = {f1_macro_5:.4f}, Time = {t_train_5:.1f}s")
    return metrics


if __name__ == "__main__":
    for ds in ["fashion_mnist", "svhn"]:
        train_keras_pipeline(ds)
    # If diabetes metrics already exist and valid, keep it or re-evaluate
    diab_met = os.path.join(OUTPUT_BASE, "03_diabetes", "evaluation_metrics.json")
    if not os.path.exists(diab_met):
        train_keras_pipeline("diabetes")
    print("\n[SUCCESS] ALL NEW KERAS MODELS TRAINED!")
