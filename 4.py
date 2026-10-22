import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

tf.random.set_seed(42)
IMG_SIZE = 160
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

(train_raw, val_raw, test_raw), info = tfds.load(
    "cats_vs_dogs", split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
    as_supervised=True, with_info=True)
label_names = info.features["label"].names
print("Classes:", label_names, "| total images:", info.splits["train"].num_examples)

plt.figure(figsize=(10, 4))
for i, (img, lbl) in enumerate(train_raw.take(8)):
    plt.subplot(2, 4, i + 1); plt.imshow(img); plt.title(label_names[int(lbl)]); plt.axis("off")
plt.suptitle("Sample images"); plt.show()

def preprocess(img, label):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label

train_ds = train_raw.map(preprocess, AUTOTUNE).shuffle(1000).batch(BATCH).prefetch(AUTOTUNE)
val_ds = val_raw.map(preprocess, AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE)
test_ds = test_raw.map(preprocess, AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE)

base = keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                      include_top=False, weights="imagenet")
base.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs, name="mobilenetv2_cats_dogs")

def trainable_params(m):
    return int(sum(np.prod(w.shape) for w in m.trainable_weights))

model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
FE_EPOCHS, FT_EPOCHS = 5, 5
fe_params = trainable_params(model)
t0 = time.perf_counter()
hist_fe = model.fit(train_ds, validation_data=val_ds, epochs=FE_EPOCHS, verbose=2)
fe_time = time.perf_counter() - t0
_, fe_acc = model.evaluate(test_ds, verbose=0)
print(f"Feature extraction test accuracy: {fe_acc:.4f}")

base.trainable = True
FINE_TUNE_AT = 100
for layer in base.layers[:FINE_TUNE_AT]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
ft_params = trainable_params(model)
t0 = time.perf_counter()
hist_ft = model.fit(train_ds, validation_data=val_ds, epochs=FE_EPOCHS + FT_EPOCHS,
                    initial_epoch=FE_EPOCHS, verbose=2)
ft_time = time.perf_counter() - t0
_, ft_acc = model.evaluate(test_ds, verbose=0)
print(f"Fine-tuning test accuracy: {ft_acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, m in zip(axes, ["accuracy", "loss"]):
    ax.plot(hist_fe.history[m] + hist_ft.history[m], label="train")
    ax.plot(hist_fe.history["val_" + m] + hist_ft.history["val_" + m], label="validation")
    ax.axvline(FE_EPOCHS - 1, color="gray", ls="--", label="start fine-tuning")
    ax.set_title(m); ax.set_xlabel("epoch"); ax.legend()
plt.tight_layout(); plt.show()

print(f"\n{'Approach':20s}{'Test acc':>10s}{'Trainable params':>18s}{'Train time (s)':>16s}")
print(f"{'Feature extraction':20s}{fe_acc:10.4f}{fe_params:18,d}{fe_time:16.1f}")
print(f"{'Fine-tuning':20s}{ft_acc:10.4f}{ft_params:18,d}{ft_time:16.1f}")

y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_prob = model.predict(test_ds, verbose=0).ravel()
y_pred = (y_prob > 0.5).astype(int)
print(classification_report(y_true, y_pred, target_names=label_names, digits=4))
ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=label_names).plot(cmap="Blues")
plt.title("Confusion matrix (fine-tuned)"); plt.show()

plt.figure(figsize=(10, 4))
for i, (img, lbl) in enumerate(test_raw.take(8)):
    p = y_prob[i]
    plt.subplot(2, 4, i + 1); plt.imshow(img); plt.axis("off")
    plt.title(f"{label_names[int(p > 0.5)]} ({p:.2f})", color="green" if int(p > 0.5) == int(lbl) else "red")
plt.suptitle("Predictions (prob of dog)"); plt.show()

