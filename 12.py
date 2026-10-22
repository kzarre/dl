"""
Practical 12: Transformer-Based Text Classification
Self-attention Transformer vs LSTM for 4-class news classification
(accuracy and training time).
Dataset: AG News (tensorflow_datasets: ag_news_subset)
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

tf.random.set_seed(42)
VOCAB, MAXLEN = 20000, 100
EMBED_DIM, NUM_HEADS, FF_DIM = 64, 4, 128
EPOCHS, BATCH = 3, 128

# ---------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------
(train_raw, test_raw), info = tfds.load("ag_news_subset", split=["train", "test"], with_info=True)
CLASS_NAMES = info.features["label"].names                     # World, Sports, Business, Sci/Tech
print("Classes:", CLASS_NAMES)


def to_arrays(ds):
    texts, labels = [], []
    for b in tfds.as_numpy(ds.batch(10000)):
        texts += [(t + b" " + d).decode("utf-8") for t, d in zip(b["title"], b["description"])]
        labels += b["label"].tolist()
    return np.array(texts), np.array(labels)


x_train_txt, y_train = to_arrays(train_raw)
x_test_txt, y_test = to_arrays(test_raw)
print("Train:", len(x_train_txt), "Test:", len(x_test_txt))
print("Example:", x_train_txt[0][:150], "->", CLASS_NAMES[y_train[0]])
print("Class counts:", np.bincount(y_train))
lengths = [len(t.split()) for t in x_train_txt[:20000]]
print(f"Words per article: mean {np.mean(lengths):.0f}, 95th pct {np.percentile(lengths, 95):.0f}")

vectorizer = layers.TextVectorization(max_tokens=VOCAB, output_sequence_length=MAXLEN)
vectorizer.adapt(x_train_txt)
x_train = vectorizer(x_train_txt).numpy().astype("int32")
x_test = vectorizer(x_test_txt).numpy().astype("int32")
print("Sequence tensor:", x_train.shape)


# ---------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = layers.Embedding(maxlen, embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        return self.token_emb(x) + self.pos_emb(positions)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, x, training=None):
        attn = self.att(x, x, training=training)                       # self-attention: Q = K = V = x
        x = self.norm1(x + self.drop1(attn, training=training))         # residual + layer norm
        ffn = self.ffn(x)
        return self.norm2(x + self.drop2(ffn, training=training))


def build_transformer():
    inputs = keras.Input(shape=(MAXLEN,), dtype="int32")
    x = TokenAndPositionEmbedding(MAXLEN, VOCAB, EMBED_DIM)(inputs)
    x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(4, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="transformer_classifier")


def build_lstm():
    return keras.Sequential([
        layers.Input(shape=(MAXLEN,), dtype="int32"),
        layers.Embedding(VOCAB, EMBED_DIM, mask_zero=True),
        layers.LSTM(128),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(4, activation="softmax"),
    ], name="lstm_classifier")


rows, histories, models = [], {}, {}
for build in [build_transformer, build_lstm]:
    model = build()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    t0 = time.perf_counter()
    h = model.fit(x_train, y_train, batch_size=BATCH, epochs=EPOCHS, validation_split=0.1, verbose=2)
    t = time.perf_counter() - t0
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    rows.append({"model": model.name, "test_accuracy": acc, "test_loss": loss,
                 "train_time_s": t, "time_per_epoch_s": t / EPOCHS, "params": model.count_params()})
    histories[model.name], models[model.name] = h, model

# ---------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, h in histories.items():
    for ax, m in zip(axes, ["accuracy", "loss"]):
        ax.plot(h.history[m], ls="--", label=f"{name} train")
        ax.plot(h.history["val_" + m], label=f"{name} val")
for ax, m in zip(axes, ["accuracy", "loss"]): ax.set_title(m); ax.set_xlabel("epoch"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# ---------------------------------------------------------------
# Evaluation and comparison
# ---------------------------------------------------------------
res = pd.DataFrame(rows).set_index("model")
print(res.round(4))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
res["test_accuracy"].plot(kind="bar", ax=axes[0], ylim=(0.8, 0.95), rot=0, title="Test accuracy")
res["train_time_s"].plot(kind="bar", ax=axes[1], rot=0, title="Training time (s)")
plt.tight_layout(); plt.show()

tr = models["transformer_classifier"]
y_pred = np.argmax(tr.predict(x_test, verbose=0), axis=1)
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=CLASS_NAMES).plot(cmap="Blues")
plt.title("Transformer confusion matrix"); plt.show()

headlines = ["Stocks rally as central bank holds interest rates steady",
             "Champions league final ends in dramatic penalty shootout",
             "New smartphone chip promises faster AI on device",
             "Peace talks resume between neighbouring countries"]
probs = tr.predict(vectorizer(np.array(headlines)).numpy().astype("int32"), verbose=0)
for h, p in zip(headlines, probs):
    print(f"{CLASS_NAMES[np.argmax(p)]:10s} ({p.max():.2f})  {h}")
