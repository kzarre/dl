"""
Practical 24: Deep Learning for Anomaly Detection
Autoencoder trained on NORMAL transactions only; fraud is flagged by high
reconstruction error. Choose a threshold and report precision, recall, F1.
Dataset: Credit Card Fraud Detection (ULB, downloaded automatically)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_recall_curve, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay, average_precision_score)

tf.random.set_seed(42)

# ---------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------
path = keras.utils.get_file("creditcard.csv", "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")
df = pd.read_csv(path)
print(df.head())
print("Shape:", df.shape)
print("Class counts:", df.Class.value_counts().to_dict(), f"-> fraud rate {df.Class.mean():.4%}")

df = df.drop(columns=["Time"])
df["Amount"] = np.log1p(df["Amount"])                     # heavy-tailed -> log scale
X, y = df.drop(columns="Class").values.astype("float32"), df["Class"].values

# 60 / 20 / 20 stratified split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

X_train_normal = X_train[y_train == 0]                    # autoencoder sees ONLY normal data
scaler = StandardScaler().fit(X_train_normal)
X_train_normal, X_val_s, X_test_s = scaler.transform(X_train_normal), scaler.transform(X_val), scaler.transform(X_test)
X_val_normal = X_val_s[y_val == 0]
print("Normal training samples:", X_train_normal.shape)

# ---------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------
n_features = X.shape[1]
autoencoder = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(4, activation="relu"),      # bottleneck
    layers.Dense(8, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(n_features),                # linear output (standardised features)
], name="fraud_autoencoder")
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

history = autoencoder.fit(X_train_normal, X_train_normal, epochs=30, batch_size=256,
                          validation_data=(X_val_normal, X_val_normal), verbose=2,
                          callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
plt.plot(history.history["loss"], label="train"); plt.plot(history.history["val_loss"], label="validation")
plt.title("Reconstruction loss on normal transactions"); plt.xlabel("epoch"); plt.legend(); plt.show()


def recon_error(Xs):
    return np.mean((Xs - autoencoder.predict(Xs, verbose=0, batch_size=2048)) ** 2, axis=1)


val_err, test_err = recon_error(X_val_s), recon_error(X_test_s)

plt.figure(figsize=(9, 4))
bins = np.logspace(np.log10(test_err.min()), np.log10(test_err.max()), 80)
plt.hist(test_err[y_test == 0], bins=bins, alpha=0.6, label="normal", density=True)
plt.hist(test_err[y_test == 1], bins=bins, alpha=0.6, label="fraud", density=True)
plt.xscale("log"); plt.xlabel("reconstruction error (log)"); plt.legend(); plt.title("Error distribution (test)")
plt.show()

# ---------------------------------------------------------------
# Threshold selection on the VALIDATION set
# ---------------------------------------------------------------
prec, rec, thr = precision_recall_curve(y_val, val_err)
f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
best_thr = thr[np.argmax(f1[:-1])]
pct_thr = np.percentile(val_err[y_val == 0], 99)          # alternative: 99th pct of normal errors
print(f"Best-F1 threshold (validation): {best_thr:.4f}")
print(f"99th-percentile threshold     : {pct_thr:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(thr, prec[:-1], label="precision"); plt.plot(thr, rec[:-1], label="recall"); plt.plot(thr, f1[:-1], label="F1")
plt.axvline(best_thr, color="k", ls="--", label="chosen threshold")
plt.xscale("log"); plt.xlabel("threshold"); plt.legend(); plt.title("Metrics vs threshold (validation)"); plt.show()

# ---------------------------------------------------------------
# Evaluation on the TEST set
# ---------------------------------------------------------------
print(f"Average precision (PR-AUC) on test: {average_precision_score(y_test, test_err):.4f}")
for name, t in [("best-F1", best_thr), ("99th percentile", pct_thr)]:
    y_pred = (test_err > t).astype(int)
    print(f"\nThreshold = {name} ({t:.4f})")
    print(f"  precision {precision_score(y_test, y_pred):.4f}  recall {recall_score(y_test, y_pred):.4f}"
          f"  F1 {f1_score(y_test, y_pred):.4f}")

y_pred = (test_err > best_thr).astype(int)
print(classification_report(y_test, y_pred, target_names=["normal", "fraud"], digits=4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["normal", "fraud"]).plot(cmap="Blues")
plt.title("Confusion matrix (best-F1 threshold)"); plt.show()

p, r, _ = precision_recall_curve(y_test, test_err)
plt.plot(r, p); plt.xlabel("recall"); plt.ylabel("precision"); plt.title("Precision-recall curve (test)"); plt.show()
