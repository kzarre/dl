import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

tf.random.set_seed(42)
np.random.seed(42)

data = load_wine(as_frame=True)
print(data.frame.head())
print("Shape:", data.frame.shape, "| missing:", data.frame.isnull().sum().sum())
print("Classes:", data.target_names, "| counts:", np.bincount(data.target))

X, y = data.data.values.astype("float32"), data.target.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

y_train_oh = keras.utils.to_categorical(y_train, 3)
y_test_oh = keras.utils.to_categorical(y_test, 3)
print("One-hot example:", y_train[0], "->", y_train_oh[0])

model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(3, activation="softmax"),
], name="wine_softmax")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(X_train, y_train_oh, epochs=150, batch_size=8, validation_split=0.2, verbose=0,
                    callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
print("Epochs trained:", len(history.history["loss"]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, m in zip(axes, ["loss", "accuracy"]):
    ax.plot(history.history[m], label="train"); ax.plot(history.history["val_" + m], label="validation")
    ax.set_title(m); ax.set_xlabel("epoch"); ax.legend()
plt.tight_layout(); plt.show()

loss, acc = model.evaluate(X_test, y_test_oh, verbose=0)
probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(probs, axis=1)
print(f"Test loss: {loss:.4f}")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall    (macro): {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1        (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
print(classification_report(y_test, y_pred, target_names=data.target_names, digits=4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=data.target_names).plot(cmap="Blues")
plt.title("Confusion matrix"); plt.show()

print("Softmax output for first 5 test samples:")
for p, t in zip(probs[:5], y_test[:5]):
    print(np.round(p, 3), "sum =", round(float(p.sum()), 3), "| true:", t)

