"""
Practical 21: Named Entity Recognition Using Deep Learning
BiLSTM sequence-labelling model tagging persons, organisations,
locations (and misc) in sentences; entity-level precision, recall, F1.
Dataset: CoNLL-2003 (downloaded automatically from a GitHub mirror)
pip install seqeval
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score

tf.random.set_seed(42)
MAXLEN = 60
EMB, UNITS = 100, 128

# ---------------------------------------------------------------
# Data loading (CoNLL format: word POS chunk NER-tag)
# If the mirror is unavailable, the same files are on Kaggle / HuggingFace "conll2003".
# ---------------------------------------------------------------
BASE = "https://raw.githubusercontent.com/davidsbatista/NER-datasets/master/CONLL2003/"


def read_conll(fname):
    path = keras.utils.get_file(f"conll2003_{fname}", BASE + fname)
    sents, tags, ws, ts = [], [], [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("-DOCSTART-"):
            if ws:
                sents.append(ws); tags.append(ts); ws, ts = [], []
            continue
        parts = line.split()
        ws.append(parts[0]); ts.append(parts[-1])
    if ws:
        sents.append(ws); tags.append(ts)
    return sents, tags


train_s, train_t = read_conll("train.txt")
val_s, val_t = read_conll("valid.txt")
test_s, test_t = read_conll("test.txt")
print("Sentences train/val/test:", len(train_s), len(val_s), len(test_s))
print("Example:", list(zip(train_s[0], train_t[0])))
print("Tag counts:", Counter(t for s in train_t for t in s))

# ---------------------------------------------------------------
# Preprocessing: vocabularies, integer encoding, padding
# ---------------------------------------------------------------
lengths = [len(s) for s in train_s]
print(f"Sentence length: mean {np.mean(lengths):.1f}, 99th pct {np.percentile(lengths, 99):.0f}, max {max(lengths)}")

words = sorted({w.lower() for s in train_s for w in s})
word2idx = {w: i + 2 for i, w in enumerate(words)}         # 0 = PAD, 1 = UNK
tag_list = ["PAD"] + sorted({t for s in train_t for t in s})
tag2idx = {t: i for i, t in enumerate(tag_list)}
N_TAGS = len(tag_list)
print("Vocab:", len(word2idx) + 2, "| tags:", tag_list)


def encode(sents, tags):
    X = [[word2idx.get(w.lower(), 1) for w in s] for s in sents]
    Y = [[tag2idx.get(t, tag2idx["O"]) for t in ts] for ts in tags]
    X = keras.utils.pad_sequences(X, maxlen=MAXLEN, padding="post", truncating="post")
    Y = keras.utils.pad_sequences(Y, maxlen=MAXLEN, padding="post", truncating="post")
    return X, Y


X_train, y_train = encode(train_s, train_t)
X_val, y_val = encode(val_s, val_t)
X_test, y_test = encode(test_s, test_t)
print("X_train:", X_train.shape, "y_train:", y_train.shape)

# ---------------------------------------------------------------
# BiLSTM tagger
# ---------------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(MAXLEN,)),
    layers.Embedding(len(word2idx) + 2, EMB, mask_zero=True),     # padding positions are masked
    layers.Bidirectional(layers.LSTM(UNITS, return_sequences=True)),
    layers.Dropout(0.3),
    layers.TimeDistributed(layers.Dense(N_TAGS, activation="softmax")),
], name="bilstm_ner")
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=2,
                    callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, m in zip(axes, ["loss", "accuracy"]):
    ax.plot(history.history[m], label="train"); ax.plot(history.history["val_" + m], label="validation")
    ax.set_title(f"{m} (token level)"); ax.set_xlabel("epoch"); ax.legend()
plt.tight_layout(); plt.show()


# ---------------------------------------------------------------
# Evaluation: entity-level precision / recall / F1
# ---------------------------------------------------------------
def decode(pred_ids, ref_tags):
    out = []
    for p, ref in zip(pred_ids, ref_tags):
        seq = [tag_list[i] if tag_list[i] != "PAD" else "O" for i in p[:min(len(ref), MAXLEN)]]
        seq += ["O"] * (len(ref) - len(seq))                     # sentences longer than MAXLEN
        out.append(seq)
    return out


pred = np.argmax(model.predict(X_test, verbose=0), axis=-1)
y_pred_tags = decode(pred, test_t)
print(f"Precision: {precision_score(test_t, y_pred_tags):.4f}")
print(f"Recall   : {recall_score(test_t, y_pred_tags):.4f}")
print(f"F1       : {f1_score(test_t, y_pred_tags):.4f}")
report = classification_report(test_t, y_pred_tags, digits=4, output_dict=True)
print(classification_report(test_t, y_pred_tags, digits=4))

ents = [k for k in report if k in ("PER", "ORG", "LOC", "MISC")]
x = np.arange(len(ents))
for i, m in enumerate(["precision", "recall", "f1-score"]):
    plt.bar(x + i * 0.25, [report[e][m] for e in ents], width=0.25, label=m)
plt.xticks(x + 0.25, ents); plt.ylim(0, 1); plt.legend(); plt.title("Entity-level scores"); plt.show()


def tag_sentence(sentence):
    toks = sentence.split()
    x = keras.utils.pad_sequences([[word2idx.get(w.lower(), 1) for w in toks]], maxlen=MAXLEN, padding="post")
    p = np.argmax(model.predict(x, verbose=0)[0], axis=-1)[:len(toks)]
    return [(w, tag_list[i]) for w, i in zip(toks, p)]


for s in ["Sundar Pichai met officials of the European Union in Brussels on Monday .",
          "Virat Kohli scored a century for India against Australia in Melbourne ."]:
    print(tag_sentence(s))
