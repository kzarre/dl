import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)
DATASET = "fashion_mnist"
NOISE_FACTOR = 0.4

(x_train, _), (x_test, _) = getattr(keras.datasets, DATASET).load_data()
x_train = x_train.astype("float32")[..., None] / 255.0
x_test = x_test.astype("float32")[..., None] / 255.0

def add_noise(x):
    return np.clip(x + NOISE_FACTOR * np.random.normal(size=x.shape), 0.0, 1.0).astype("float32")

x_train_noisy, x_test_noisy = add_noise(x_train), add_noise(x_test)
print("Clean:", x_train.shape, "Noisy:", x_train_noisy.shape)

plt.figure(figsize=(10, 2.5))
for i in range(8):
    plt.subplot(2, 8, i + 1); plt.imshow(x_train[i, :, :, 0], cmap="gray"); plt.axis("off")
    plt.subplot(2, 8, 8 + i + 1); plt.imshow(x_train_noisy[i, :, :, 0], cmap="gray"); plt.axis("off")
plt.suptitle(f"Clean (top) vs noisy (bottom), noise factor {NOISE_FACTOR}"); plt.show()

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(2, padding="same"),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(2, padding="same"),
    layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same"),
    layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same"),
    layers.Conv2D(1, 3, activation="sigmoid", padding="same"),
], name="denoising_autoencoder")
model.compile(optimizer="adam", loss="mse")
model.summary()

history = model.fit(x_train_noisy, x_train,
                    epochs=15, batch_size=128, validation_data=(x_test_noisy, x_test), verbose=2)

plt.plot(history.history["loss"], label="train"); plt.plot(history.history["val_loss"], label="validation")
plt.title("Denoising loss (MSE)"); plt.xlabel("epoch"); plt.legend(); plt.show()

denoised = model.predict(x_test_noisy, verbose=0)
mse_noisy = np.mean((x_test_noisy - x_test) ** 2)
mse_denoised = np.mean((denoised - x_test) ** 2)
psnr_noisy = tf.reduce_mean(tf.image.psnr(x_test_noisy, x_test, max_val=1.0)).numpy()
psnr_denoised = tf.reduce_mean(tf.image.psnr(denoised, x_test, max_val=1.0)).numpy()
print(f"{'':10s}{'MSE':>10s}{'PSNR (dB)':>12s}")
print(f"{'Noisy':10s}{mse_noisy:10.5f}{psnr_noisy:12.2f}")
print(f"{'Denoised':10s}{mse_denoised:10.5f}{psnr_denoised:12.2f}")
print(f"PSNR improvement: {psnr_denoised - psnr_noisy:.2f} dB")

n = 10
per_psnr = tf.image.psnr(denoised[:n], x_test[:n], max_val=1.0).numpy()
plt.figure(figsize=(15, 5))
for i in range(n):
    for row, (imgs, title) in enumerate([(x_test, "clean"), (x_test_noisy, "noisy"), (denoised, "denoised")]):
        plt.subplot(3, n, row * n + i + 1); plt.imshow(imgs[i, :, :, 0], cmap="gray"); plt.axis("off")
        if i == 0: plt.title(title, loc="left")
        if row == 2: plt.title(f"{per_psnr[i]:.1f} dB", fontsize=8)
plt.suptitle("Clean / noisy / denoised"); plt.show()

