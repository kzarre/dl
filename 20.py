"""
Practical 20: Text Generation Using LSTM
Character-level LSTM language model on Shakespeare; experiment with
sequence length, embedding size and sampling temperature.
Dataset: Shakespeare text (downloaded automatically)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
EPOCHS = 10
BATCH = 64
UNITS = 256

# ---------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------
path = keras.utils.get_file("shakespeare.txt",
                            "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
text = open(path, "rb").read().decode("utf-8")
print(f"Characters: {len(text):,}")
print(text[:300])

vocab = sorted(set(text))
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = np.array(vocab)
V = len(vocab)
print("Unique characters:", V)
ids = np.array([char2idx[c] for c in text], dtype="int32")
split = int(0.9 * len(ids))
train_ids, val_ids = ids[:split], ids[split:]


def make_dataset(arr, seq_len, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(arr).batch(seq_len + 1, drop_remainder=True)
    ds = ds.map(lambda s: (s[:-1], s[1:]))            # input "Hell" -> target "ello"
    if shuffle:
        ds = ds.shuffle(10000)
    return ds.batch(BATCH, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def build_model(emb_dim):
    model = keras.Sequential([
        layers.Input(shape=(None,), dtype="int32"),
        layers.Embedding(V, emb_dim),
        layers.LSTM(UNITS, return_sequences=True),       # predict next char at every position
        layers.Dropout(0.2),
        layers.Dense(V),                                 # logits
    ], name=f"char_lstm_emb{emb_dim}")
    model.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    return model


# ---------------------------------------------------------------
# Experiments: sequence length and embedding size
# ---------------------------------------------------------------
configs = [{"seq_len": 50, "emb": 64}, {"seq_len": 100, "emb": 64}, {"seq_len": 100, "emb": 256}]
rows, histories, models = [], {}, {}
for cfg in configs:
    name = f"seq{cfg['seq_len']}_emb{cfg['emb']}"
    model = build_model(cfg["emb"])
    model.summary()
    h = model.fit(make_dataset(train_ids, cfg["seq_len"]), validation_data=make_dataset(val_ids, cfg["seq_len"], False),
                  epochs=EPOCHS, verbose=2)
    vl = min(h.history["val_loss"])
    rows.append({"config": name, "best_val_loss": vl, "val_perplexity": np.exp(vl),
                 "val_accuracy": max(h.history["val_accuracy"]), "params": model.count_params()})
    histories[name], models[name] = h, (model, cfg["seq_len"])

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for name, h in histories.items():
    axes[0].plot(h.history["loss"], ls="--", label=f"{name} train")
    axes[0].plot(h.history["val_loss"], label=f"{name} val")
    axes[1].plot(h.history["val_accuracy"], label=name)
axes[0].set_title("Loss"); axes[1].set_title("Validation next-char accuracy")
for ax in axes: ax.set_xlabel("epoch"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

res = pd.DataFrame(rows).set_index("config")
print(res.round(4))
best = res["best_val_loss"].idxmin()
print("Best configuration:", best)


# ---------------------------------------------------------------
# Generation with different temperatures
# ---------------------------------------------------------------
def generate(model, context_len, start, n_chars=400, temperature=1.0):
    out = [char2idx[c] for c in start]
    for _ in range(n_chars):
        x = np.array([out[-context_len:]], dtype="int32")
        logits = model(x, training=False)[0, -1].numpy() / temperature
        out.append(int(tf.random.categorical(logits[None], 1)[0, 0]))
    return "".join(idx2char[out])


best_model, ctx = models[best]
for t in [0.2, 0.5, 1.0, 1.5]:
    print(f"\n===== temperature {t} =====")
    print(generate(best_model, ctx, "ROMEO:\n", temperature=t))
print("\nLow temperature -> safe, repetitive text; high temperature -> creative but more spelling errors.")
