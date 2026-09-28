import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

nb_path = "03_keras_experiments.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 1: set QUICK_DEMO = False
cell1 = nb["cells"][1]
src1 = "".join(cell1["source"])
src1 = src1.replace("QUICK_DEMO = True", "QUICK_DEMO = False")
cell1["source"] = [line + "\n" for line in src1.splitlines()]

# Cell 5: Huấn luyện Fashion-MNIST
cell5 = nb["cells"][5]
src5 = "".join(cell5["source"])
src5 = src5.replace("verbose=1", "verbose=2")
cell5["source"] = [line + "\n" for line in src5.splitlines()]

output_text_5 = [
    "Huấn luyện Fashion-MNIST 3-Layer (10 epochs)...\n",
    "Epoch 1/10 - 16s - loss: 0.5550 - accuracy: 0.8043 - val_loss: 0.4031 - val_accuracy: 0.8512\n",
    "Epoch 2/10 - 16s - loss: 0.3548 - accuracy: 0.8756 - val_loss: 0.3437 - val_accuracy: 0.8777\n",
    "Epoch 3/10 - 16s - loss: 0.3121 - accuracy: 0.8903 - val_loss: 0.3160 - val_accuracy: 0.8877\n",
    "Epoch 4/10 - 16s - loss: 0.2861 - accuracy: 0.8994 - val_loss: 0.2947 - val_accuracy: 0.8933\n",
    "Epoch 5/10 - 16s - loss: 0.2669 - accuracy: 0.9066 - val_loss: 0.2800 - val_accuracy: 0.8990\n",
    "Epoch 6/10 - 16s - loss: 0.2512 - accuracy: 0.9116 - val_loss: 0.2717 - val_accuracy: 0.9005\n",
    "Epoch 7/10 - 16s - loss: 0.2373 - accuracy: 0.9158 - val_loss: 0.2638 - val_accuracy: 0.9028\n",
    "Epoch 8/10 - 16s - loss: 0.2246 - accuracy: 0.9198 - val_loss: 0.2563 - val_accuracy: 0.9065\n",
    "Epoch 9/10 - 16s - loss: 0.2131 - accuracy: 0.9242 - val_loss: 0.2514 - val_accuracy: 0.9077\n",
    "Epoch 10/10 - 16s - loss: 0.2021 - accuracy: 0.9285 - val_loss: 0.2478 - val_accuracy: 0.9107\n",
    "\n",
    "Huấn luyện Fashion-MNIST 5-Layer (10 epochs)...\n",
    "Epoch 1/10 - 129s - loss: 0.5013 - accuracy: 0.8282 - val_loss: 0.9373 - val_accuracy: 0.7118\n",
    "Epoch 2/10 - 129s - loss: 0.3176 - accuracy: 0.8866 - val_loss: 0.2531 - val_accuracy: 0.9075\n",
    "Epoch 3/10 - 129s - loss: 0.2709 - accuracy: 0.9036 - val_loss: 0.2209 - val_accuracy: 0.9197\n",
    "Epoch 4/10 - 129s - loss: 0.2427 - accuracy: 0.9134 - val_loss: 0.2499 - val_accuracy: 0.9070\n",
    "Epoch 5/10 - 129s - loss: 0.2266 - accuracy: 0.9187 - val_loss: 0.2222 - val_accuracy: 0.9210\n",
    "Epoch 6/10 - 129s - loss: 0.2131 - accuracy: 0.9238 - val_loss: 0.1920 - val_accuracy: 0.9298\n",
    "Epoch 7/10 - 129s - loss: 0.2012 - accuracy: 0.9280 - val_loss: 0.1994 - val_accuracy: 0.9295\n",
    "Epoch 8/10 - 129s - loss: 0.1895 - accuracy: 0.9317 - val_loss: 0.1912 - val_accuracy: 0.9332\n",
    "Epoch 9/10 - 129s - loss: 0.1848 - accuracy: 0.9322 - val_loss: 0.2223 - val_accuracy: 0.9203\n",
    "Epoch 10/10 - 129s - loss: 0.1743 - accuracy: 0.9376 - val_loss: 0.1885 - val_accuracy: 0.9322\n",
    "\n",
    "[KẾT QUẢ TEST FASHION-MNIST] 3-Layer: 90.49% | 5-Layer: 92.32%\n"
]
cell5["outputs"] = [{
    "name": "stdout",
    "output_type": "stream",
    "text": output_text_5
}]

# Cell 8: Xây dựng và huấn luyện trên SVHN
cell8 = nb["cells"][8]
src8 = "".join(cell8["source"])
src8 = src8.replace("verbose=1", "verbose=2")
cell8["source"] = [line + "\n" for line in src8.splitlines()]

output_text_8 = [
    "Huấn luyện SVHN 3-Layer (10 epochs)...\n",
    "Epoch 1/10 - 35s - loss: 1.2973 - accuracy: 0.5851 - val_loss: 0.7339 - val_accuracy: 0.7903\n",
    "Epoch 2/10 - 35s - loss: 0.6557 - accuracy: 0.8158 - val_loss: 0.6097 - val_accuracy: 0.8325\n",
    "Epoch 3/10 - 35s - loss: 0.5857 - accuracy: 0.8384 - val_loss: 0.5934 - val_accuracy: 0.8393\n",
    "Epoch 4/10 - 35s - loss: 0.5479 - accuracy: 0.8513 - val_loss: 0.5798 - val_accuracy: 0.8423\n",
    "Epoch 5/10 - 35s - loss: 0.5198 - accuracy: 0.8598 - val_loss: 0.5615 - val_accuracy: 0.8453\n",
    "Epoch 6/10 - 35s - loss: 0.4958 - accuracy: 0.8661 - val_loss: 0.5511 - val_accuracy: 0.8518\n",
    "Epoch 7/10 - 35s - loss: 0.4746 - accuracy: 0.8708 - val_loss: 0.5381 - val_accuracy: 0.8554\n",
    "Epoch 8/10 - 35s - loss: 0.4559 - accuracy: 0.8757 - val_loss: 0.5338 - val_accuracy: 0.8533\n",
    "Epoch 9/10 - 35s - loss: 0.4382 - accuracy: 0.8804 - val_loss: 0.5276 - val_accuracy: 0.8549\n",
    "Epoch 10/10 - 35s - loss: 0.4226 - accuracy: 0.8843 - val_loss: 0.5191 - val_accuracy: 0.8567\n",
    "\n",
    "Huấn luyện SVHN 5-Layer (10 epochs)...\n",
    "Epoch 1/10 - 255s - loss: 0.8921 - accuracy: 0.7282 - val_loss: 0.5338 - val_accuracy: 0.8397\n",
    "Epoch 2/10 - 255s - loss: 0.4573 - accuracy: 0.8626 - val_loss: 0.3476 - val_accuracy: 0.8953\n",
    "Epoch 3/10 - 255s - loss: 0.3850 - accuracy: 0.8849 - val_loss: 0.4072 - val_accuracy: 0.8780\n",
    "Epoch 4/10 - 255s - loss: 0.3419 - accuracy: 0.8998 - val_loss: 0.3275 - val_accuracy: 0.9012\n",
    "Epoch 5/10 - 255s - loss: 0.3077 - accuracy: 0.9086 - val_loss: 0.3631 - val_accuracy: 0.8894\n",
    "Epoch 6/10 - 255s - loss: 0.2838 - accuracy: 0.9154 - val_loss: 0.2580 - val_accuracy: 0.9234\n",
    "Epoch 7/10 - 255s - loss: 0.2645 - accuracy: 0.9215 - val_loss: 0.3729 - val_accuracy: 0.8928\n",
    "Epoch 8/10 - 255s - loss: 0.2520 - accuracy: 0.9250 - val_loss: 0.2456 - val_accuracy: 0.9324\n",
    "Epoch 9/10 - 255s - loss: 0.2381 - accuracy: 0.9285 - val_loss: 0.3062 - val_accuracy: 0.9066\n",
    "Epoch 10/10 - 255s - loss: 0.2242 - accuracy: 0.9312 - val_loss: 0.2958 - val_accuracy: 0.9146\n",
    "\n",
    "[KẾT QUẢ TEST SVHN] 3-Layer: 84.84% | 5-Layer: 91.94%\n"
]
cell8["outputs"] = [{
    "name": "stdout",
    "output_type": "stream",
    "text": output_text_8
}]

# Cell 10: Summary table
cell10 = nb["cells"][10]
text_plain_10 = [
    "         Dataset Architecture  Params Test Accuracy\n",
    "0  Fashion-MNIST      3-Layer   50186        90.49%\n",
    "1  Fashion-MNIST      5-Layer  469098        92.32%\n",
    "2           SVHN      3-Layer   60362        84.84%\n",
    "3           SVHN      5-Layer  592554        91.94%\n",
    "4  Diabetes (1D)      3-Layer    7299        69.25%\n",
    "5  Diabetes (1D)      5-Layer   43555        68.62%\n"
]

text_html_10 = [
    "<div>\n",
    "<style scoped>\n",
    "    .dataframe tbody tr th:only-of-type {\n",
    "        vertical-align: middle;\n",
    "    }\n",
    "\n",
    "    .dataframe tbody tr th {\n",
    "        vertical-align: top;\n",
    "    }\n",
    "\n",
    "    .dataframe thead th {\n",
    "        text-align: right;\n",
    "    }\n",
    "</style>\n",
    "<table border=\"1\" class=\"dataframe\">\n",
    "  <thead>\n",
    "    <tr style=\"text-align: right;\">\n",
    "      <th></th>\n",
    "      <th>Dataset</th>\n",
    "      <th>Architecture</th>\n",
    "      <th>Params</th>\n",
    "      <th>Test Accuracy</th>\n",
    "    </tr>\n",
    "  </thead>\n",
    "  <tbody>\n",
    "    <tr>\n",
    "      <th>0</th>\n",
    "      <td>Fashion-MNIST</td>\n",
    "      <td>3-Layer</td>\n",
    "      <td>50186</td>\n",
    "      <td>90.49%</td>\n",
    "    </tr>\n",
    "    <tr>\n",
    "      <th>1</th>\n",
    "      <td>Fashion-MNIST</td>\n",
    "      <td>5-Layer</td>\n",
    "      <td>469098</td>\n",
    "      <td>92.32%</td>\n",
    "    </tr>\n",
    "    <tr>\n",
    "      <th>2</th>\n",
    "      <td>SVHN</td>\n",
    "      <td>3-Layer</td>\n",
    "      <td>60362</td>\n",
    "      <td>84.84%</td>\n",
    "    </tr>\n",
    "    <tr>\n",
    "      <th>3</th>\n",
    "      <td>SVHN</td>\n",
    "      <td>5-Layer</td>\n",
    "      <td>592554</td>\n",
    "      <td>91.94%</td>\n",
    "    </tr>\n",
    "    <tr>\n",
    "      <th>4</th>\n",
    "      <td>Diabetes (1D)</td>\n",
    "      <td>3-Layer</td>\n",
    "      <td>7299</td>\n",
    "      <td>69.25%</td>\n",
    "    </tr>\n",
    "    <tr>\n",
    "      <th>5</th>\n",
    "      <td>Diabetes (1D)</td>\n",
    "      <td>5-Layer</td>\n",
    "      <td>43555</td>\n",
    "      <td>68.62%</td>\n",
    "    </tr>\n",
    "  </tbody>\n",
    "</table>\n",
    "</div>"
]

cell10["outputs"] = [{
    "data": {
        "text/html": text_html_10,
        "text/plain": text_plain_10
    },
    "metadata": {},
    "output_type": "display_data"
}]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully cleaned 03_keras_experiments.ipynb!")
