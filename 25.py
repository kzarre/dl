import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
IMG = 128
BATCH = 32
EPOCHS = 20
N_CLASSES = 3
CLASS_NAMES = ["pet", "background", "border"]
AUTOTUNE = tf.data.AUTOTUNE

dataset, info = tfds.load("oxford_iiit_pet", with_info=True)
print("Train:", info.splits["train"].num_examples, "Test:", info.splits["test"].num_examples)

def load(sample):
    img = tf.image.resize(sample["image"], (IMG, IMG)) / 255.0
    mask = tf.image.resize(sample["segmentation_mask"], (IMG, IMG), method="nearest")
    mask = tf.cast(mask, tf.int32) - 1
    return img, mask

def augment(img, mask):
    flip = tf.random.uniform(()) > 0.5
    img = tf.where(flip, tf.image.flip_left_right(img), img)
    mask = tf.where(flip, tf.image.flip_left_right(mask), mask)
    return img, mask

train_ds = (dataset["train"].map(load, AUTOTUNE).cache().shuffle(1000)
            .map(augment, AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE))
test_all = dataset["test"].map(load, AUTOTUNE).cache()
val_ds = test_all.take(1000).batch(BATCH).prefetch(AUTOTUNE)
test_ds = test_all.skip(1000).batch(BATCH).prefetch(AUTOTUNE)

for img, mask in dataset["train"].map(load).take(1):
    print("Image:", img.shape, "Mask:", mask.shape, "mask values:", np.unique(mask.numpy()))
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.imshow(img); plt.title("image"); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(mask[:, :, 0], vmin=0, vmax=2); plt.title("mask"); plt.axis("off")
    plt.show()

def conv_block(x, filters):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    return x

def build_unet():
    inputs = keras.Input(shape=(IMG, IMG, 3))
    skips, x = [], inputs
    for f in [32, 64, 128, 256]:
        x = conv_block(x, f); skips.append(x)
        x = layers.MaxPooling2D()(x)
    x = conv_block(x, 512)
    for f, skip in zip([256, 128, 64, 32], reversed(skips)):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, f)
    outputs = layers.Conv2D(N_CLASSES, 1, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="unet")

model = build_unet()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2,
                    callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, m in zip(axes, ["loss", "accuracy"]):
    ax.plot(history.history[m], label="train"); ax.plot(history.history["val_" + m], label="validation")
    ax.set_title(f"{m} (pixel level)"); ax.set_xlabel("epoch"); ax.legend()
plt.tight_layout(); plt.show()

cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
for imgs, masks in test_ds:
    pred = np.argmax(model.predict(imgs, verbose=0), axis=-1).ravel()
    true = masks.numpy().ravel()
    cm += np.bincount(true * N_CLASSES + pred, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)

iou_per_class = np.diag(cm) / (cm.sum(0) + cm.sum(1) - np.diag(cm))
pixel_acc = np.diag(cm).sum() / cm.sum()
for name, v in zip(CLASS_NAMES, iou_per_class):
    print(f"IoU {name:10s}: {v:.4f}")
print(f"Mean IoU      : {iou_per_class.mean():.4f}")
print(f"Pixel accuracy: {pixel_acc:.4f}")

imgs, masks = next(iter(test_ds))
preds = np.argmax(model.predict(imgs, verbose=0), axis=-1)
plt.figure(figsize=(10, 13))
for i in range(4):
    true_pet, pred_pet = masks[i, :, :, 0].numpy() == 0, preds[i] == 0
    pet_iou = np.logical_and(true_pet, pred_pet).sum() / max(np.logical_or(true_pet, pred_pet).sum(), 1)
    panels = [(imgs[i], "image"), (masks[i, :, :, 0], "true mask"), (preds[i], f"predicted (pet IoU {pet_iou:.2f})")]
    for j, (im, title) in enumerate(panels):
        plt.subplot(4, 3, i * 3 + j + 1)
        if j == 0:
            plt.imshow(im)
        else:
            plt.imshow(im, vmin=0, vmax=2)
        plt.title(title, fontsize=9); plt.axis("off")
plt.tight_layout(); plt.show()

