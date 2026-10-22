import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

tf.random.set_seed(42)
np.random.seed(42)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()
print("Train:", x_train.shape, "Test:", x_test.shape)
print("Class counts:", np.bincount(y_train))

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1); plt.imshow(x_train[i]); plt.title(CLASS_NAMES[y_train[i]]); plt.axis("off")
plt.suptitle("Sample CIFAR-10 images"); plt.show()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.1
print(f"batch={BATCH_SIZE} epochs={EPOCHS} lr={LEARNING_RATE} val_split={VALIDATION_SPLIT}")

model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
], name="cifar10_cnn")
model.summary()

model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_split=VALIDATION_SPLIT, callbacks=callbacks, verbose=2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, m in zip(axes, ["loss", "accuracy"]):
    ax.plot(history.history[m], label="train"); ax.plot(history.history["val_" + m], label="validation")
    ax.set_title(m); ax.set_xlabel("epoch"); ax.legend()
plt.tight_layout(); plt.show()

model.save("cifar10_cnn.keras")
reloaded = keras.models.load_model("cifar10_cnn.keras")
print("Model saved to cifar10_cnn.keras and reloaded.")

loss1, acc1 = model.evaluate(x_test, y_test, verbose=0)
loss2, acc2 = reloaded.evaluate(x_test, y_test, verbose=0)
print(f"Original model  test acc = {acc1:.4f}")
print(f"Reloaded model  test acc = {acc2:.4f}  (should match)")

y_pred = np.argmax(reloaded.predict(x_test, verbose=0), axis=1)
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=CLASS_NAMES).plot(
    cmap="Blues", xticks_rotation=45)
plt.title("Confusion matrix"); plt.show()

plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1); plt.imshow(x_test[i])
    ok = y_pred[i] == y_test[i]
    plt.title(f"P:{CLASS_NAMES[y_pred[i]]}\nT:{CLASS_NAMES[y_test[i]]}", color="green" if ok else "red", fontsize=9)
    plt.axis("off")
plt.tight_layout(); plt.show()

