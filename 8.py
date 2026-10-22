"""
Practical 8: ResNet50 Transfer Learning and Image Captioning
Extract image features with pretrained ResNet50, then train a CNN + LSTM
caption generator (merge architecture). Evaluate with BLEU.
Dataset: Flickr8k (downloaded automatically, ~1 GB)
pip install nltk
"""
import os
import re
import glob
import random
import pickle
import zipfile
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

random.seed(42)
tf.random.set_seed(42)
AUTOTUNE = tf.data.AUTOTUNE


def download_and_unzip(url, out_dir):
    zpath = keras.utils.get_file(fname=os.path.basename(url), origin=url,
                                 cache_dir=".", cache_subdir="downloads")
    if not os.path.exists(out_dir):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out_dir)
    return out_dir


BASE = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/"
download_and_unzip(BASE + "Flickr8k_Dataset.zip", "flickr8k/images")
download_and_unzip(BASE + "Flickr8k_text.zip", "flickr8k/text")

img_paths = sorted(glob.glob("flickr8k/images/**/*.jpg", recursive=True))
find = lambda name: glob.glob(f"flickr8k/text/**/{name}", recursive=True)[0]
print("Images found:", len(img_paths))

# ---------------------------------------------------------------
# Step 1: ResNet50 feature extraction (transfer learning)
# ---------------------------------------------------------------
FEATURE_FILE = "flickr8k_resnet50_features.pkl"
resnet = keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg")
print("ResNet50 output shape:", resnet.output_shape)            # (None, 2048)

if os.path.exists(FEATURE_FILE):
    features = pickle.load(open(FEATURE_FILE, "rb"))
else:
    def load_img(path):
        img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (224, 224))
        return keras.applications.resnet50.preprocess_input(img)
    ds = tf.data.Dataset.from_tensor_slices(img_paths).map(load_img, AUTOTUNE).batch(64).prefetch(AUTOTUNE)
    feats = resnet.predict(ds, verbose=1)
    features = {os.path.basename(p): f for p, f in zip(img_paths, feats)}
    pickle.dump(features, open(FEATURE_FILE, "wb"))
print("Feature vectors:", len(features))

# ---------------------------------------------------------------
# Step 2: Caption preprocessing
# ---------------------------------------------------------------
def clean(caption):
    caption = re.sub(r"[^a-z ]", "", caption.lower())
    words = [w for w in caption.split() if len(w) > 1 or w == "a"]
    return "startseq " + " ".join(words) + " endseq"


captions = defaultdict(list)
for line in open(find("Flickr8k.token.txt"), encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    img_id, cap = line.split("\t")
    img = img_id.split("#")[0]
    if img in features:
        captions[img].append(clean(cap))

read_ids = lambda f: [l.strip() for l in open(find(f)) if l.strip() in captions]
train_imgs = read_ids("Flickr_8k.trainImages.txt")
val_imgs = read_ids("Flickr_8k.devImages.txt")
test_imgs = read_ids("Flickr_8k.testImages.txt")
print("Train/Val/Test images:", len(train_imgs), len(val_imgs), len(test_imgs))
print("Example captions:", captions[train_imgs[0]][:2])

counts = Counter(w for img in train_imgs for c in captions[img] for w in c.split())
vocab = sorted(w for w, c in counts.items() if c >= 5)          # drop rare words
word2idx = {w: i + 1 for i, w in enumerate(vocab)}               # 0 = padding
idx2word = {i: w for w, i in word2idx.items()}
VOCAB_SIZE = len(word2idx) + 1
encode = lambda cap: [word2idx[w] for w in cap.split() if w in word2idx]
MAX_LEN = max(len(encode(c)) for img in train_imgs for c in captions[img])
print("Vocabulary size:", VOCAB_SIZE, "| max caption length:", MAX_LEN)


def data_generator(img_list, images_per_batch=32):
    """Each caption 'a dog runs' becomes pairs: (img, [startseq]) -> a, (img, [startseq a]) -> dog ..."""
    while True:
        random.shuffle(img_list)
        for i in range(0, len(img_list), images_per_batch):
            X1, X2, Y = [], [], []
            for img in img_list[i:i + images_per_batch]:
                for cap in captions[img]:
                    seq = encode(cap)
                    for t in range(1, len(seq)):
                        X1.append(features[img]); X2.append(seq[:t]); Y.append(seq[t])
            X2 = keras.utils.pad_sequences(X2, maxlen=MAX_LEN, padding="post")
            yield (np.array(X1, dtype="float32"), X2), np.array(Y)


# ---------------------------------------------------------------
# Step 3: CNN features + LSTM decoder (merge model)
# ---------------------------------------------------------------
img_in = keras.Input(shape=(2048,), name="image_features")
fe = layers.Dropout(0.4)(img_in)
fe = layers.Dense(256, activation="relu")(fe)

txt_in = keras.Input(shape=(MAX_LEN,), name="partial_caption")
se = layers.Embedding(VOCAB_SIZE, 256, mask_zero=True)(txt_in)
se = layers.Dropout(0.4)(se)
se = layers.LSTM(256)(se)

dec = layers.add([fe, se])
dec = layers.Dense(256, activation="relu")(dec)
out = layers.Dense(VOCAB_SIZE, activation="softmax")(dec)
model = keras.Model([img_in, txt_in], out, name="caption_generator")
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

EPOCHS, IPB = 15, 32
history = model.fit(data_generator(list(train_imgs), IPB), steps_per_epoch=len(train_imgs) // IPB,
                    validation_data=data_generator(list(val_imgs), IPB), validation_steps=len(val_imgs) // IPB,
                    epochs=EPOCHS, verbose=2,
                    callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, m in zip(axes, ["loss", "accuracy"]):
    ax.plot(history.history[m], label="train"); ax.plot(history.history["val_" + m], label="validation")
    ax.set_title(m); ax.set_xlabel("epoch"); ax.legend()
plt.tight_layout(); plt.show()


# ---------------------------------------------------------------
# Step 4: Caption generation + BLEU evaluation
# ---------------------------------------------------------------
def generate_caption(feature):
    words = ["startseq"]
    for _ in range(MAX_LEN):
        seq = keras.utils.pad_sequences([encode(" ".join(words))], maxlen=MAX_LEN, padding="post")
        probs = model([feature[None], seq], training=False).numpy()[0]
        word = idx2word.get(int(np.argmax(probs)))
        if word is None or word == "endseq":
            break
        words.append(word)
    return " ".join(words[1:])


N_EVAL = 200
refs, hyps = [], []
for img in test_imgs[:N_EVAL]:
    refs.append([c.split()[1:-1] for c in captions[img]])
    hyps.append(generate_caption(features[img]).split())
sm = SmoothingFunction().method1
for n, w in enumerate([(1, 0, 0, 0), (.5, .5, 0, 0), (1/3, 1/3, 1/3, 0), (.25, .25, .25, .25)], 1):
    print(f"BLEU-{n}: {corpus_bleu(refs, hyps, weights=w, smoothing_function=sm):.4f}")

img_dir = os.path.dirname(img_paths[0])
plt.figure(figsize=(14, 8))
for i, img in enumerate(test_imgs[:6]):
    plt.subplot(2, 3, i + 1); plt.imshow(plt.imread(os.path.join(img_dir, img))); plt.axis("off")
    plt.title(f"PRED: {generate_caption(features[img])}\nTRUE: {captions[img][0][9:-7]}", fontsize=9)
plt.tight_layout(); plt.show()
