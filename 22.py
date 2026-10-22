import os
import glob
import random
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

random.seed(42)
tf.random.set_seed(42)
NUM_PAIRS = 40000
SEQ_LEN = 20
VOCAB = 15000
EMB, UNITS = 256, 512
EPOCHS, BATCH = 15, 64

url = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"
zpath = keras.utils.get_file(fname="fra-eng.zip", origin=url, cache_dir=".", cache_subdir="downloads")
if not os.path.exists("fra_eng"):
    with zipfile.ZipFile(zpath) as z:
        z.extractall("fra_eng")
lines = open(glob.glob("fra_eng/**/fra.txt", recursive=True)[0], encoding="utf-8").read().strip().split("\n")
pairs = []
for line in lines[:NUM_PAIRS]:
    eng, fra = line.split("\t")[:2]
    pairs.append((eng, "[start] " + fra + " [end]"))
random.shuffle(pairs)
for p in pairs[:5]:
    print(p)
n_val = n_test = int(0.1 * len(pairs))
train_pairs = pairs[:-(n_val + n_test)]
val_pairs = pairs[-(n_val + n_test):-n_test]
test_pairs = pairs[-n_test:]
print("Train/val/test pairs:", len(train_pairs), len(val_pairs), len(test_pairs))

def standardize(s):
    s = tf.strings.lower(s)
    s = tf.strings.regex_replace(s, "[  ]", " ")
    s = tf.strings.regex_replace(s, r'[.?!,¿¡"«»:;()\-]', "")
    return tf.strings.strip(s)

eng_vec = layers.TextVectorization(VOCAB, output_sequence_length=SEQ_LEN, standardize=standardize)
fra_vec = layers.TextVectorization(VOCAB, output_sequence_length=SEQ_LEN + 1, standardize=standardize)
eng_vec.adapt([p[0] for p in train_pairs])
fra_vec.adapt([p[1] for p in train_pairs])
EN_V, FR_V = eng_vec.vocabulary_size(), fra_vec.vocabulary_size()
fr_vocab = fra_vec.get_vocabulary()
START, END = fr_vocab.index("[start]"), fr_vocab.index("[end]")
print("English vocab:", EN_V, "| French vocab:", FR_V)

def vectorize(pairs_):
    enc = eng_vec([p[0] for p in pairs_]).numpy()
    fr = fra_vec([p[1] for p in pairs_]).numpy()
    return (enc, fr[:, :-1]), fr[:, 1:]

(tr_enc, tr_dec), tr_y = vectorize(train_pairs)
(va_enc, va_dec), va_y = vectorize(val_pairs)
print("Encoder input:", tr_enc.shape, "Decoder input:", tr_dec.shape, "Target:", tr_y.shape)
print("Example decoder in :", tr_dec[0][:8], "\nExample target     :", tr_y[0][:8])

enc_in = keras.Input(shape=(SEQ_LEN,), dtype="int64", name="english")
x = layers.Embedding(EN_V, EMB, mask_zero=True)(enc_in)
enc_out, h, c = layers.LSTM(UNITS, return_sequences=True, return_state=True, name="encoder_lstm")(x)

dec_in = keras.Input(shape=(SEQ_LEN,), dtype="int64", name="french_shifted")
y = layers.Embedding(FR_V, EMB, mask_zero=True)(dec_in)
dec_out = layers.LSTM(UNITS, return_sequences=True, name="decoder_lstm")(y, initial_state=[h, c])

context, attn_scores = layers.Attention(name="attention")([dec_out, enc_out], return_attention_scores=True)
concat = layers.Concatenate()([dec_out, context])
concat = layers.Dropout(0.3)(concat)
outputs = layers.Dense(FR_V, activation="softmax")(concat)

model = keras.Model([enc_in, dec_in], outputs, name="nmt_attention")
attention_model = keras.Model([enc_in, dec_in], attn_scores)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit([tr_enc, tr_dec], tr_y, validation_data=([va_enc, va_dec], va_y),
                    epochs=EPOCHS, batch_size=BATCH, verbose=2,
                    callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, m in zip(axes, ["loss", "accuracy"]):
    ax.plot(history.history[m], label="train"); ax.plot(history.history["val_" + m], label="validation")
    ax.set_title(m); ax.set_xlabel("epoch"); ax.legend()
plt.tight_layout(); plt.show()

def translate(sentence):
    enc = eng_vec([sentence])
    dec = np.zeros((1, SEQ_LEN), dtype="int64"); dec[0, 0] = START
    words = []
    for i in range(SEQ_LEN - 1):
        probs = model([enc, dec], training=False)[0, i].numpy()
        nid = int(np.argmax(probs))
        if nid == END:
            break
        words.append(fr_vocab[nid]); dec[0, i + 1] = nid
    return " ".join(words), enc, dec

N_EVAL = 300
refs, hyps = [], []
for eng, fra in test_pairs[:N_EVAL]:
    ref = standardize(fra).numpy().decode("utf-8").split()[1:-1]
    refs.append([ref]); hyps.append(translate(eng)[0].split())
sm = SmoothingFunction().method1
print(f"BLEU-1: {corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=sm):.4f}")
print(f"BLEU-4: {corpus_bleu(refs, hyps, smoothing_function=sm):.4f}")

for eng, fra in test_pairs[:8]:
    print(f"EN : {eng}\nREF: {fra[8:-6]}\nOUT: {translate(eng)[0]}\n")

sentence = test_pairs[0][0]
out, enc, dec = translate(sentence)
scores = attention_model([enc, dec], training=False)[0].numpy()
src = standardize(sentence).numpy().decode("utf-8").split()
tgt = out.split()
plt.figure(figsize=(7, 5))
plt.imshow(scores[:len(tgt), :len(src)], cmap="viridis")
plt.xticks(range(len(src)), src, rotation=45); plt.yticks(range(len(tgt)), tgt)
plt.xlabel("English (source)"); plt.ylabel("French (generated)"); plt.title("Attention weights"); plt.colorbar()
plt.tight_layout(); plt.show()

