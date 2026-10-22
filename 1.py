"""
Practical 1: TensorFlow Fundamentals and Perceptron
Basic tensor operations, then a single-layer Perceptron vs an MLP on MNIST.
Dataset: MNIST (built into Keras)
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

tf.random.set_seed(42)
np.random.seed(42)


def plot_history(history, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, m in zip(axes, ["loss", "accuracy"]):
        ax.plot(history.history[m], label="train")
        ax.plot(history.history["val_" + m], label="validation")
        ax.set_title(f"{title} - {m}"); ax.set_xlabel("epoch"); ax.legend()
    plt.tight_layout(); plt.show()


# ---------------------------------------------------------------
# PART A: TensorFlow tensor basics
# ---------------------------------------------------------------
a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[5., 6.], [7., 8.]])
print("a =\n", a.numpy())
print("shape:", a.shape, "dtype:", a.dtype, "rank:", tf.rank(a).numpy())
print("a + b =\n", tf.add(a, b).numpy())
print("a * b (element-wise) =\n", tf.multiply(a, b).numpy())
print("a @ b (matrix multiply) =\n", tf.matmul(a, b).numpy())
print("a + 10 (broadcasting) =\n", (a + 10).numpy())
print("transpose(a) =\n", tf.transpose(a).numpy())
print("reshape(a, [4]) =", tf.reshape(a, [4]).numpy())
print("sum / mean / max:", tf.reduce_sum(a).numpy(), tf.reduce_mean(a).numpy(), tf.reduce_max(a).numpy())
print("sum along axis 0:", tf.reduce_sum(a, axis=0).numpy())
print("argmax along axis 1:", tf.argmax(a, axis=1).numpy())
print("cast to int32:\n", tf.cast(a, tf.int32).numpy())
print("zeros:", tf.zeros((2, 3)).numpy().tolist())
print("random normal:", tf.random.normal((3,)).numpy())
print("slicing a[:, 1] =", a[:, 1].numpy())

v = tf.Variable(3.0)
v.assign_add(1.0)
print("Variable after assign_add(1):", v.numpy())
with tf.GradientTape() as tape:
    y = v ** 2                              # dy/dv = 2v
print("Gradient of v^2 at v=4:", tape.gradient(y, v).numpy())

# ---------------------------------------------------------------
# PART B: Data preprocessing
# ---------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Train:", x_train.shape, "Test:", x_test.shape)

plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1); plt.imshow(x_train[i], cmap="gray"); plt.title(y_train[i]); plt.axis("off")
plt.suptitle("Sample MNIST digits"); plt.show()

# flatten 28x28 -> 784 and scale pixels to [0, 1]
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# ---------------------------------------------------------------
# Models
# ---------------------------------------------------------------
# Single-layer perceptron: input connected directly to 10 output neurons (no hidden layer)
perceptron = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(10, activation="softmax"),
], name="single_layer_perceptron")

# Multi-layer perceptron: hidden layers with non-linear activations
mlp = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
], name="multi_layer_perceptron")

EPOCHS, BATCH = 10, 128
results = {}
for model in [perceptron, mlp]:
    model.summary()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_split=0.1, verbose=2)
    plot_history(hist, model.name)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    results[model.name] = (acc, hist)
    print(f"{model.name}: test loss={loss:.4f}  test accuracy={acc:.4f}")

# ---------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------
plt.figure(figsize=(7, 4))
for name, (_, h) in results.items():
    plt.plot(h.history["val_accuracy"], label=name)
plt.title("Validation accuracy: Perceptron vs MLP"); plt.xlabel("epoch"); plt.legend(); plt.show()

plt.bar(results.keys(), [r[0] for r in results.values()], color=["tab:orange", "tab:blue"])
plt.ylim(0.85, 1.0); plt.title("Test accuracy"); plt.show()

for name, (acc, _) in results.items():
    print(f"{name:28s} test accuracy = {acc:.4f}")

# Detailed evaluation of the MLP
y_pred = np.argmax(mlp.predict(x_test, verbose=0), axis=1)
print(classification_report(y_test, y_pred, digits=4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Blues")
plt.title("MLP confusion matrix"); plt.show()

plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1); plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"P:{y_pred[i]}", color="green" if y_pred[i] == y_test[i] else "red"); plt.axis("off")
plt.suptitle("MLP predictions"); plt.show()
