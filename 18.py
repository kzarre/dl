import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
IMG = 150
BATCH = 32
EPOCHS = 25
AUTOTUNE = tf.data.AUTOTUNE

(train_raw, val_raw, test_raw), info = tfds.load(
    "cats_vs_dogs", split=["train[:30%]", "train[30%:40%]", "train[40%:50%]"],
    as_supervised=True, with_info=True)
names = info.features["label"].names
print("Train/val/test sizes:", [int(d.cardinality()) for d in (train_raw, val_raw, test_raw)])

def prep(img, label):
    return tf.image.resize(img, (IMG, IMG)) / 255.0, label

train_ds = train_raw.map(prep, AUTOTUNE).cache().shuffle(1000).batch(BATCH).prefetch(AUTOTUNE)
val_ds = val_raw.map(prep, AUTOTUNE).cache().batch(BATCH).prefetch(AUTOTUNE)
test_ds = test_raw.map(prep, AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE)

augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
], name="augmentation")

sample_img = next(iter(train_raw.map(prep).take(1)))[0]
plt.figure(figsize=(10, 5))
for i in range(8):
    aug = augmentation(sample_img[None], training=True)[0]
    plt.subplot(2, 4, i + 1); plt.imshow(np.clip(aug, 0, 1)); plt.axis("off")
plt.suptitle("Augmented versions of one image"); plt.show()

def build(augment):
    model = keras.Sequential(name="cnn_with_aug" if augment else "cnn_no_aug")
    model.add(layers.Input(shape=(IMG, IMG, 3)))
    if augment:
        model.add(augmentation)
    for f in (32, 64, 128, 128):
        model.add(layers.Conv2D(f, 3, activation="relu"))
        model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

rows, histories = [], {}
for augment in [False, True]:
    model = build(augment)
    model.summary()
    h = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
    _, test_acc = model.evaluate(test_ds, verbose=0)
    tr, va = h.history["accuracy"][-1], h.history["val_accuracy"][-1]
    rows.append({"model": model.name, "final_train_acc": tr, "final_val_acc": va,
                 "overfit_gap": tr - va, "best_val_acc": max(h.history["val_accuracy"]), "test_acc": test_acc})
    histories[model.name] = h

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for (name, h), c in zip(histories.items(), ["tab:red", "tab:blue"]):
    axes[0].plot(h.history["accuracy"], c=c, ls="--", label=f"{name} train")
    axes[0].plot(h.history["val_accuracy"], c=c, label=f"{name} val")
    axes[1].plot(h.history["loss"], c=c, ls="--", label=f"{name} train")
    axes[1].plot(h.history["val_loss"], c=c, label=f"{name} val")
axes[0].set_title("Accuracy"); axes[1].set_title("Loss")
for ax in axes: ax.set_xlabel("epoch"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

res = pd.DataFrame(rows).set_index("model")
print(res.round(4))
print("\nWithout augmentation train accuracy keeps rising while validation loss rises (overfitting);"
      "\nwith augmentation the train/val gap is much smaller and test accuracy is higher.")
res[["final_train_acc", "final_val_acc", "test_acc"]].plot(kind="bar", rot=0, ylim=(0.5, 1.0), figsize=(8, 4))
plt.title("With vs without augmentation"); plt.show()

