import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
DATASET = "fashion_mnist"
LATENT_DIM = 100
BATCH = 256
EPOCHS = 30
SHOW_EVERY = 5

(x_train, _), _ = getattr(keras.datasets, DATASET).load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = x_train[..., None]
print("Training images:", x_train.shape, "range:", x_train.min(), x_train.max())
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(BATCH, drop_remainder=True)

generator = keras.Sequential([
    layers.Input(shape=(LATENT_DIM,)),
    layers.Dense(7 * 7 * 256, use_bias=False),
    layers.BatchNormalization(), layers.LeakyReLU(0.2),
    layers.Reshape((7, 7, 256)),
    layers.Conv2DTranspose(128, 5, strides=1, padding="same", use_bias=False),
    layers.BatchNormalization(), layers.LeakyReLU(0.2),
    layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False),
    layers.BatchNormalization(), layers.LeakyReLU(0.2),
    layers.Conv2DTranspose(1, 5, strides=2, padding="same", activation="tanh"),
], name="generator")

discriminator = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(64, 5, strides=2, padding="same"), layers.LeakyReLU(0.2), layers.Dropout(0.3),
    layers.Conv2D(128, 5, strides=2, padding="same"), layers.LeakyReLU(0.2), layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(1),
], name="discriminator")
generator.summary(); discriminator.summary()

bce = keras.losses.BinaryCrossentropy(from_logits=True)
g_opt = keras.optimizers.Adam(2e-4, beta_1=0.5)
d_opt = keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(real):
    noise = tf.random.normal([tf.shape(real)[0], LATENT_DIM])
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake = generator(noise, training=True)
        real_logits = discriminator(real, training=True)
        fake_logits = discriminator(fake, training=True)
        d_loss = bce(tf.ones_like(real_logits), real_logits) + bce(tf.zeros_like(fake_logits), fake_logits)
        g_loss = bce(tf.ones_like(fake_logits), fake_logits)
    g_opt.apply_gradients(zip(g_tape.gradient(g_loss, generator.trainable_variables), generator.trainable_variables))
    d_opt.apply_gradients(zip(d_tape.gradient(d_loss, discriminator.trainable_variables), discriminator.trainable_variables))
    return g_loss, d_loss, tf.reduce_mean(tf.sigmoid(real_logits)), tf.reduce_mean(tf.sigmoid(fake_logits))

fixed_noise = tf.random.normal([16, LATENT_DIM])

def show_samples(epoch):
    imgs = generator(fixed_noise, training=False).numpy()
    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1); plt.imshow((imgs[i, :, :, 0] + 1) / 2, cmap="gray"); plt.axis("off")
    plt.suptitle(f"Generated samples - epoch {epoch}")
    plt.savefig(f"gan_epoch_{epoch:03d}.png"); plt.show()

hist = {"g_loss": [], "d_loss": [], "D(x)": [], "D(G(z))": []}
batch_g, batch_d = [], []
show_samples(0)
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    eg, ed, ex, ez = [], [], [], []
    for real in dataset:
        g, d, dx, dz = train_step(real)
        eg.append(float(g)); ed.append(float(d)); ex.append(float(dx)); ez.append(float(dz))
    batch_g += eg; batch_d += ed
    hist["g_loss"].append(np.mean(eg)); hist["d_loss"].append(np.mean(ed))
    hist["D(x)"].append(np.mean(ex)); hist["D(G(z))"].append(np.mean(ez))
    print(f"Epoch {epoch:3d} | G loss {hist['g_loss'][-1]:.4f} | D loss {hist['d_loss'][-1]:.4f} | "
          f"D(x) {hist['D(x)'][-1]:.3f} | D(G(z)) {hist['D(G(z))'][-1]:.3f} | {time.time() - t0:.1f}s")
    if epoch % SHOW_EVERY == 0 or epoch == 1:
        show_samples(epoch)

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
smooth = lambda v, k=50: np.convolve(v, np.ones(k) / k, mode="valid")
axes[0].plot(smooth(batch_g), label="generator"); axes[0].plot(smooth(batch_d), label="discriminator")
axes[0].set_title("Loss per batch (smoothed)"); axes[0].set_xlabel("iteration"); axes[0].legend()
axes[1].plot(hist["g_loss"], label="generator"); axes[1].plot(hist["d_loss"], label="discriminator")
axes[1].set_title("Mean loss per epoch"); axes[1].set_xlabel("epoch"); axes[1].legend()
axes[2].plot(hist["D(x)"], label="D(real)"); axes[2].plot(hist["D(G(z))"], label="D(fake)")
axes[2].axhline(0.5, color="gray", ls="--"); axes[2].set_title("Discriminator outputs"); axes[2].legend()
plt.tight_layout(); plt.show()

gen = generator(tf.random.normal([10, LATENT_DIM]), training=False).numpy()
plt.figure(figsize=(14, 3))
for i in range(10):
    plt.subplot(2, 10, i + 1); plt.imshow((x_train[i, :, :, 0] + 1) / 2, cmap="gray"); plt.axis("off")
    plt.subplot(2, 10, 10 + i + 1); plt.imshow((gen[i, :, :, 0] + 1) / 2, cmap="gray"); plt.axis("off")
plt.suptitle("Real (top) vs generated (bottom)"); plt.show()
generator.save("gan_generator.keras")

