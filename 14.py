"""
Practical 14: Binary Classification Using Deep Neural Networks
Feedforward network for malignant vs benign tumours with binary
cross-entropy; compare activations and optimizers.
Dataset: Breast Cancer Wisconsin (sklearn)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, RocCurveDisplay)

tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame
print(df.iloc[:, :6].head())
print("Shape:", df.shape, "| missing values:", df.isnull().sum().sum())
print("Target names:", data.target_names, "| counts:", np.bincount(data.target))   # 0 malignant, 1 benign

X, y = data.data.values.astype("float32"), data.target.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)


def build(activation, optimizer, name):
    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(32, activation=activation),
        layers.Dropout(0.2),
        layers.Dense(16, activation=activation),
        layers.Dense(1, activation="sigmoid"),             # probability of class 1 (benign)
    ], name=name)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


make_opt = {"SGD": lambda: keras.optimizers.SGD(0.01, momentum=0.9),
            "Adam": lambda: keras.optimizers.Adam(1e-3),
            "RMSprop": lambda: keras.optimizers.RMSprop(1e-3)}
configs = [("relu", "Adam"), ("tanh", "Adam"), ("sigmoid", "Adam"),      # activation comparison
           ("relu", "SGD"), ("relu", "RMSprop")]                          # optimizer comparison

rows, histories, models = [], {}, {}
for act, opt in configs:
    name = f"{act}_{opt}"
    model = build(act, make_opt[opt](), name)
    model.summary()
    h = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    prob = model.predict(X_test, verbose=0).ravel()
    rows.append({"model": name, "activation": act, "optimizer": opt, "epochs_run": len(h.history["loss"]),
                 "test_loss": loss, "test_accuracy": acc, "ROC_AUC": roc_auc_score(y_test, prob)})
    histories[name], models[name] = h, model
    print(f"{name:14s} test acc {acc:.4f}")

# ---------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for name, h in histories.items():
    axes[0].plot(h.history["val_loss"], label=name)
    axes[1].plot(h.history["val_accuracy"], label=name)
axes[0].set_title("Validation loss (binary cross-entropy)"); axes[1].set_title("Validation accuracy")
for ax in axes: ax.set_xlabel("epoch"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

best = max(histories, key=lambda n: [r for r in rows if r["model"] == n][0]["test_accuracy"])
h = histories[best]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, m in zip(axes, ["loss", "accuracy"]):
    ax.plot(h.history[m], label="train"); ax.plot(h.history["val_" + m], label="validation")
    ax.set_title(f"{best}: {m}"); ax.set_xlabel("epoch"); ax.legend()
plt.tight_layout(); plt.show()

# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------
res = pd.DataFrame(rows).set_index("model")
print(res.round(4))
print("Best model:", best)
prob = models[best].predict(X_test, verbose=0).ravel()
y_pred = (prob > 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=data.target_names, digits=4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=data.target_names).plot(cmap="Blues")
plt.title(f"Confusion matrix - {best}"); plt.show()
RocCurveDisplay.from_predictions(y_test, prob, name=best); plt.plot([0, 1], [0, 1], "k--"); plt.show()
