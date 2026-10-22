"""
Practical 2: Regression Using Keras
House price prediction trained with gradient descent (SGD); compare
no regularization vs L1 vs L2 vs Dropout to study overfitting.
Dataset: California Housing (sklearn)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------
data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.head())
print(df.describe().T)
print("Missing values:", df.isnull().sum().sum())

X = data.data.values.astype("float32")
y = data.target.values.astype("float32")        # median house value in $100,000s

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)          # fit on train only (no leakage)
X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
print("Train/Val/Test:", X_train.shape, X_val.shape, X_test.shape)


# ---------------------------------------------------------------
# Model builder (deliberately large so overfitting is visible)
# ---------------------------------------------------------------
def build_model(kind):
    reg = None
    if kind == "L1":
        reg = regularizers.l1(1e-3)
    elif kind == "L2":
        reg = regularizers.l2(1e-3)
    model = keras.Sequential(name=f"housing_{kind}")
    model.add(layers.Input(shape=(X_train.shape[1],)))
    for units in (256, 128, 64):
        model.add(layers.Dense(units, activation="relu", kernel_regularizer=reg))
        if kind == "Dropout":
            model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))                  # linear output for regression
    # plain gradient descent (mini-batch SGD) with momentum
    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, clipnorm=1.0)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


EPOCHS, BATCH = 60, 32
results, histories, models = [], {}, {}
for kind in ["NoReg", "L1", "L2", "Dropout"]:
    model = build_model(kind)
    model.summary()
    h = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=EPOCHS, batch_size=BATCH, verbose=0)
    histories[kind], models[kind] = h, model
    tr_pred = model.predict(X_train, verbose=0).ravel()
    te_pred = model.predict(X_test, verbose=0).ravel()
    tr_mse, te_mse = mean_squared_error(y_train, tr_pred), mean_squared_error(y_test, te_pred)
    results.append({"model": kind, "train_MSE": tr_mse, "test_MSE": te_mse,
                    "test_MAE": mean_absolute_error(y_test, te_pred),
                    "test_R2": r2_score(y_test, te_pred), "gap(test-train)": te_mse - tr_mse})
    print(f"{kind}: train MSE={tr_mse:.4f}  test MSE={te_mse:.4f}")

# ---------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (kind, h) in zip(axes.ravel(), histories.items()):
    ax.plot(h.history["loss"], label="train loss")
    ax.plot(h.history["val_loss"], label="val loss")
    ax.set_title(kind); ax.set_xlabel("epoch"); ax.set_ylabel("MSE (+penalty)"); ax.legend()
plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 4))
for kind, h in histories.items():
    plt.plot(h.history["val_mae"], label=kind)
plt.title("Validation MAE by regularization"); plt.xlabel("epoch"); plt.legend(); plt.show()

# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------
res = pd.DataFrame(results).set_index("model")
print(res.round(4))
print("\nNote: a smaller train/test gap means less overfitting. "
      "Reported 'loss' for L1/L2 includes the penalty, so compare MSE from predictions.")

best = res["test_MSE"].idxmin()
pred = models[best].predict(X_test, verbose=0).ravel()
plt.figure(figsize=(5, 5))
plt.scatter(y_test, pred, s=4, alpha=0.4)
plt.plot([0, 5], [0, 5], "r--")
plt.xlabel("Actual price ($100k)"); plt.ylabel("Predicted"); plt.title(f"Predicted vs actual ({best})")
plt.show()
