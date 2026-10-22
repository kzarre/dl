"""
Practical 6: Autoencoder Implementation
Compress MNIST digits to a 32-d code and reconstruct them; show original vs
reconstructed images and report reconstruction error.
Dataset: MNIST
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
LATENT = 32

# ---------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------
(x_train, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
print("Train:", x_train.shape, "Test:", x_test.shape, "| pixel range:", x_train.min(), x_train.max())

# ---------------------------------------------------------------
# Model: encoder + decoder
# ---------------------------------------------------------------
encoder = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(LATENT, activation="relu", name="bottleneck"),
], name="encoder")

decoder = keras.Sequential([
    layers.Input(shape=(LATENT,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(784, activation="sigmoid"),       # pixels in [0, 1]
], name="decoder")

inputs = keras.Input(shape=(784,))
autoencoder = keras.Model(inputs, decoder(encoder(inputs)), name="autoencoder")
encoder.summary(); decoder.summary(); autoencoder.summary()
print(f"Compression: 784 -> {LATENT} values ({784 / LATENT:.1f}x)")

autoencoder.compile(optimizer="adam", loss="mse")
history = autoencoder.fit(x_train, x_train,                 # input = target
                          epochs=20, batch_size=256, validation_data=(x_test, x_test), verbose=2)

# ---------------------------------------------------------------
# Training curve
# ---------------------------------------------------------------
plt.plot(history.history["loss"], label="train"); plt.plot(history.history["val_loss"], label="validation")
plt.title("Reconstruction loss (MSE)"); plt.xlabel("epoch"); plt.legend(); plt.show()

# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------
recon = autoencoder.predict(x_test, verbose=0)
per_image_mse = np.mean((x_test - recon) ** 2, axis=1)
print(f"Mean reconstruction MSE on test set: {per_image_mse.mean():.5f}")
print(f"Min / max per-image MSE: {per_image_mse.min():.5f} / {per_image_mse.max():.5f}")

n = 10
plt.figure(figsize=(15, 3.5))
for i in range(n):
    plt.subplot(2, n, i + 1); plt.imshow(x_test[i].reshape(28, 28), cmap="gray"); plt.axis("off")
    if i == 0: plt.title("original", loc="left")
    plt.subplot(2, n, n + i + 1); plt.imshow(recon[i].reshape(28, 28), cmap="gray"); plt.axis("off")
    plt.title(f"mse={per_image_mse[i]:.4f}", fontsize=8)
plt.suptitle("Original (top) vs reconstructed (bottom)"); plt.show()

plt.hist(per_image_mse, bins=60); plt.title("Distribution of reconstruction error")
plt.xlabel("per-image MSE"); plt.show()

worst = np.argsort(per_image_mse)[-5:]
plt.figure(figsize=(8, 3.5))
for j, i in enumerate(worst):
    plt.subplot(2, 5, j + 1); plt.imshow(x_test[i].reshape(28, 28), cmap="gray"); plt.axis("off")
    plt.subplot(2, 5, 5 + j + 1); plt.imshow(recon[i].reshape(28, 28), cmap="gray"); plt.axis("off")
plt.suptitle("Worst reconstructions"); plt.show()

codes = encoder.predict(x_test, verbose=0)
print("Encoded representation shape:", codes.shape)
