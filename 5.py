"""
Practical 5: Sentiment Analysis Using RNN, LSTM and GRU
Compare accuracy, training time and handling of long reviews.
Dataset: IMDB (built into Keras)
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)
VOCAB = 10000
MAXLEN = 300
EMB = 64
UNITS = 64
EPOCHS = 5
BATCH = 128

# ---------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB)
print("Train:", len(x_train), "Test:", len(x_test), "| positive share:", y_train.mean())

train_lengths = np.array([len(x) for x in x_train])
test_lengths = np.array([len(x) for x in x_test])
print("Review length: mean=%.0f median=%.0f 90th pct=%.0f" %
      (train_lengths.mean(), np.median(train_lengths), np.percentile(train_lengths, 90)))
plt.hist(train_lengths, bins=60); plt.axvline(MAXLEN, color="r", ls="--", label=f"maxlen={MAXLEN}")
plt.title("Review lengths (tokens)"); plt.legend(); plt.show()

word_index = keras.datasets.imdb.get_word_index()
index_word = {v + 3: k for k, v in word_index.items()}
index_word.update({0: "<pad>", 1: "<start>", 2: "<unk>", 3: "<unused>"})
print("Decoded review:", " ".join(index_word.get(i, "?") for i in x_train[0][:60]), "...")
print("Label:", y_train[0])

# pre-padding/pre-truncating keeps the END of each review next to the final RNN state
x_train = keras.utils.pad_sequences(x_train, maxlen=MAXLEN, padding="pre", truncating="pre")
x_test = keras.utils.pad_sequences(x_test, maxlen=MAXLEN, padding="pre", truncating="pre")
print("Padded shape:", x_train.shape)


# ---------------------------------------------------------------
# Models
# ---------------------------------------------------------------
def build(cell_name):
    cell = {"SimpleRNN": layers.SimpleRNN, "LSTM": layers.LSTM, "GRU": layers.GRU}[cell_name]
    model = keras.Sequential([
        layers.Input(shape=(MAXLEN,)),
        layers.Embedding(VOCAB, EMB),
        cell(UNITS),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name=cell_name)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


histories, rows, preds = {}, [], {}
for name in ["SimpleRNN", "LSTM", "GRU"]:
    model = build(name)
    model.summary()
    t0 = time.perf_counter()
    h = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_split=0.2, verbose=2)
    train_time = time.perf_counter() - t0
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    preds[name] = (model.predict(x_test, verbose=0).ravel() > 0.5).astype(int)
    histories[name] = h
    rows.append({"model": name, "test_accuracy": acc, "test_loss": loss,
                 "train_time_s": train_time, "params": model.count_params()})

# ---------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, h in histories.items():
    axes[0].plot(h.history["val_accuracy"], label=f"{name} val")
    axes[0].plot(h.history["accuracy"], ls="--", label=f"{name} train")
    axes[1].plot(h.history["val_loss"], label=f"{name} val")
axes[0].set_title("Accuracy"); axes[1].set_title("Validation loss")
for ax in axes: ax.set_xlabel("epoch"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# ---------------------------------------------------------------
# Evaluation: accuracy, time, long-review handling
# ---------------------------------------------------------------
res = pd.DataFrame(rows).set_index("model")
print(res.round(4))

bins = [0, 100, 200, 300, 10_000]
labels = ["<=100", "101-200", "201-300", ">300 (truncated)"]
bucket = pd.cut(test_lengths, bins=bins, labels=labels)
by_len = pd.DataFrame({name: pd.Series(p == y_test).groupby(bucket, observed=False).mean()
                       for name, p in preds.items()})
print("\nAccuracy by original review length:\n", by_len.round(4))
by_len.plot(kind="bar", figsize=(8, 4), ylim=(0.4, 1.0), rot=0)
plt.title("Accuracy vs review length"); plt.ylabel("accuracy"); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
res["test_accuracy"].plot(kind="bar", ax=axes[0], ylim=(0.4, 1), rot=0, title="Test accuracy")
res["train_time_s"].plot(kind="bar", ax=axes[1], rot=0, title="Training time (s)")
plt.tight_layout(); plt.show()
print("\nObservation: SimpleRNN suffers from vanishing gradients on long sequences; "
      "LSTM/GRU gates keep long-range information, GRU is usually faster than LSTM.")
