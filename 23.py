"""
Practical 23: Variational Autoencoder (VAE)
Learn a 2-D latent space of MNIST digits, visualise it, and generate new
digits by sampling from it.
Dataset: MNIST
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
LATENT_DIM = 2               # 2-D so the latent space can be plotted directly

# ---------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32")[..., None] / 255.0
x_test = x_test.astype("float32")[..., None] / 255.0
print("Train:", x_train.shape, "Test:", x_test.shape)


# ---------------------------------------------------------------
# Encoder -> (z_mean, z_log_var) -> sampled z -> decoder
# ---------------------------------------------------------------
class Sampling(layers.Layer):
    """Reparameterisation trick: z = mean + sigma * epsilon."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


enc_in = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(enc_in)
x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(enc_in, [z_mean, z_log_var, z], name="encoder")

dec_in = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(7 * 7 * 64, activation="relu")(dec_in)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
dec_out = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)
decoder = keras.Model(dec_in, dec_out, name="decoder")
encoder.summary(); decoder.summary()


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder, self.decoder = encoder, decoder
        self.total_tracker = keras.metrics.Mean(name="loss")
        self.rec_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_tracker, self.rec_tracker, self.kl_tracker]

    def call(self, x):
        return self.decoder(self.encoder(x)[2])

    def _losses(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        recon = self.decoder(z)
        rec = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data, recon), axis=(1, 2)))
        kl = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl = tf.reduce_mean(tf.reduce_sum(kl, axis=1))
        return rec, kl

    def _update(self, rec, kl):
        self.total_tracker.update_state(rec + kl); self.rec_tracker.update_state(rec); self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            rec, kl = self._losses(data)
            total = rec + kl
        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return self._update(rec, kl)

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        return self._update(*self._losses(data))


vae = VAE(encoder, decoder, name="vae")
vae.compile(optimizer=keras.optimizers.Adam(1e-3))
history = vae.fit(x_train, x_train, epochs=30, batch_size=128, validation_data=(x_test, x_test), verbose=2)

# ---------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, m in zip(axes, ["loss", "reconstruction_loss", "kl_loss"]):
    ax.plot(history.history[m], label="train"); ax.plot(history.history["val_" + m], label="validation")
    ax.set_title(m); ax.set_xlabel("epoch"); ax.legend()
plt.tight_layout(); plt.show()

# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------
results = vae.evaluate(x_test, x_test, verbose=0, return_dict=True)
print("Test:", {k: round(float(v), 3) for k, v in results.items()})

# 1. Latent space coloured by digit
z_mean_test, _, _ = encoder.predict(x_test, verbose=0)
plt.figure(figsize=(8, 7))
sc = plt.scatter(z_mean_test[:, 0], z_mean_test[:, 1], c=y_test, cmap="tab10", s=3)
plt.colorbar(sc, label="digit"); plt.xlabel("z[0]"); plt.ylabel("z[1]"); plt.title("2-D latent space (test set)")
plt.show()

# 2. Walk across the latent space
n, size = 15, 28
grid = np.linspace(-3, 3, n)
zs = np.array([[xi, yi] for yi in grid[::-1] for xi in grid])
imgs = decoder.predict(zs, verbose=0)[..., 0]
canvas = imgs.reshape(n, n, size, size).transpose(0, 2, 1, 3).reshape(n * size, n * size)
plt.figure(figsize=(9, 9)); plt.imshow(canvas, cmap="gray"); plt.axis("off")
plt.title("Digits decoded from a grid over the latent space"); plt.show()

# 3. New digits by sampling z ~ N(0, I)
samples = decoder.predict(np.random.normal(size=(16, LATENT_DIM)), verbose=0)
plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1); plt.imshow(samples[i, :, :, 0], cmap="gray"); plt.axis("off")
plt.suptitle("Randomly generated digits"); plt.show()

# 4. Reconstructions
recon = vae.predict(x_test[:10], verbose=0)
plt.figure(figsize=(14, 3))
for i in range(10):
    plt.subplot(2, 10, i + 1); plt.imshow(x_test[i, :, :, 0], cmap="gray"); plt.axis("off")
    plt.subplot(2, 10, 10 + i + 1); plt.imshow(recon[i, :, :, 0], cmap="gray"); plt.axis("off")
plt.suptitle("Original (top) vs VAE reconstruction (bottom)"); plt.show()
