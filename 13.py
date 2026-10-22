"""
Practical 13: Multilayer Perceptron for Classification
Compare MLPs with different numbers of hidden layers, neurons and
activation functions (sigmoid, tanh, ReLU).
Dataset: Fashion-MNIST
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

tf.random.set_seed(42)
CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
EPOCHS, BATCH = 10, 128

# ---------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1); plt.imshow(x_train[i], cmap="gray"); plt.title(CLASS_NAMES[y_train[i]], fontsize=7)
    plt.axis("off")
plt.show()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
print("Train:", x_train.shape, "Test:", x_test.shape)


def build_mlp(hidden_layers, activation, name):
    model = keras.Sequential(name=name)
    model.add(layers.Input(shape=(784,)))
    for units in hidden_layers:
        model.add(layers.Dense(units, activation=activation))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


experiments = {
    "Depth": {
        "1 layer [128] relu": ([128], "relu"),
        "2 layers [128,128] relu": ([128, 128], "relu"),
        "3 layers [128,128,128] relu": ([128, 128, 128], "relu"),
    },
    "Width": {
        "2 layers [64,64] relu": ([64, 64], "relu"),
        "2 layers [256,256] relu": ([256, 256], "relu"),
        "2 layers [512,512] relu": ([512, 512], "relu"),
    },
    "Activation": {
        "2 layers [256,128] sigmoid": ([256, 128], "sigmoid"),
        "2 layers [256,128] tanh": ([256, 128], "tanh"),
        "2 layers [256,128] relu": ([256, 128], "relu"),
    },
}

rows, histories, models = [], {}, {}
for group, configs in experiments.items():
    for name, (hidden, act) in configs.items():
        model = build_mlp(hidden, act, name.replace(" ", "_").replace("[", "").replace("]", "").replace(",", "-"))
        model.summary()
        h = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_split=0.1, verbose=0)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"{group:10s} | {name:30s} | test acc {acc:.4f}")
        rows.append({"group": group, "config": name, "params": model.count_params(),
                     "train_acc": h.history["accuracy"][-1], "val_acc": h.history["val_accuracy"][-1],
                     "test_acc": acc})
        histories[name], models[name] = h, model

# ---------------------------------------------------------------
# Training curves per experiment group
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(17, 8))
for col, (group, configs) in enumerate(experiments.items()):
    for name in configs:
        axes[0, col].plot(histories[name].history["val_accuracy"], label=name)
        axes[1, col].plot(histories[name].history["val_loss"], label=name)
    axes[0, col].set_title(f"{group}: validation accuracy"); axes[1, col].set_title(f"{group}: validation loss")
    for r in range(2): axes[r, col].legend(fontsize=8); axes[r, col].set_xlabel("epoch")
plt.tight_layout(); plt.show()

# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------
res = pd.DataFrame(rows)
print(res.round(4).to_string(index=False))
res.set_index("config")["test_acc"].plot(kind="barh", figsize=(8, 5), xlim=(0.8, 0.92), title="Test accuracy")
plt.tight_layout(); plt.show()

best_name = res.loc[res.test_acc.idxmax(), "config"]
print("Best configuration:", best_name)
y_pred = np.argmax(models[best_name].predict(x_test, verbose=0), axis=1)
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=CLASS_NAMES).plot(
    cmap="Blues", xticks_rotation=45)
plt.title(f"Confusion matrix - {best_name}"); plt.show()
