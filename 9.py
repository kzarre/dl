"""
Practical 9: Deep Learning-Based Recommendation System
Embedding-based (matrix-factorization style) model on implicit user-item
interactions, evaluated with Precision@K and Recall@K.
Dataset: Retailrocket e-commerce (Kaggle: retailrocket/ecommerce-dataset).
Download events.csv and put it next to this script (or set EVENTS_CSV).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

tf.random.set_seed(42)
rng = np.random.default_rng(42)
EVENTS_CSV = "events.csv"
EMB_DIM = 32
NEG_PER_POS = 4
K_VALUES = [5, 10, 20]

# ---------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------
df = pd.read_csv(EVENTS_CSV)
print(df.head())
print(df["event"].value_counts())
print("Raw users:", df.visitorid.nunique(), "items:", df.itemid.nunique())

# implicit feedback: any view / addtocart / transaction = interaction
df = df[["visitorid", "itemid"]].drop_duplicates()

# keep active users and popular items (repeat to converge)
for _ in range(3):
    ic = df.itemid.value_counts(); df = df[df.itemid.isin(ic[ic >= 5].index)]
    uc = df.visitorid.value_counts(); df = df[df.visitorid.isin(uc[uc >= 5].index)]

df["user"] = df.visitorid.astype("category").cat.codes
df["item"] = df.itemid.astype("category").cat.codes
n_users, n_items = df.user.nunique(), df.item.nunique()
print(f"After filtering: {len(df)} interactions, {n_users} users, {n_items} items, "
      f"sparsity={1 - len(df) / (n_users * n_items):.5f}")

# per-user split: 80% of each user's items for training, 20% held out for testing
df["r"] = rng.random(len(df))
df["pct"] = df.groupby("user")["r"].rank(pct=True)
train_df, test_df = df[df.pct <= 0.8], df[df.pct > 0.8]
print("Train interactions:", len(train_df), "Test interactions:", len(test_df))

# negative sampling: pair each positive with random (probably unseen) items
pos_u, pos_i = train_df.user.values, train_df.item.values
neg_u = np.repeat(pos_u, NEG_PER_POS)
neg_i = rng.integers(0, n_items, size=len(neg_u))
users = np.concatenate([pos_u, neg_u])
items = np.concatenate([pos_i, neg_i])
labels = np.concatenate([np.ones(len(pos_u)), np.zeros(len(neg_u))]).astype("float32")
perm = rng.permutation(len(users))
users, items, labels = users[perm], items[perm], labels[perm]

# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------
u_in = keras.Input(shape=(1,), name="user")
i_in = keras.Input(shape=(1,), name="item")
u_vec = layers.Flatten()(layers.Embedding(n_users, EMB_DIM, name="user_embedding",
                                          embeddings_regularizer=regularizers.l2(1e-6))(u_in))
i_vec = layers.Flatten()(layers.Embedding(n_items, EMB_DIM, name="item_embedding",
                                          embeddings_regularizer=regularizers.l2(1e-6))(i_in))
i_bias = layers.Flatten()(layers.Embedding(n_items, 1, name="item_bias")(i_in))
score = layers.Add()([layers.Dot(axes=1)([u_vec, i_vec]), i_bias])
out = layers.Activation("sigmoid")(score)
model = keras.Model([u_in, i_in], out, name="embedding_recommender")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
model.summary()

history = model.fit({"user": users, "item": items}, labels, batch_size=1024, epochs=10,
                    validation_split=0.1, verbose=2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, m in zip(axes, ["loss", "auc"]):
    ax.plot(history.history[m], label="train"); ax.plot(history.history["val_" + m], label="validation")
    ax.set_title(m); ax.set_xlabel("epoch"); ax.legend()
plt.tight_layout(); plt.show()

# ---------------------------------------------------------------
# Evaluation: Precision@K and Recall@K
# ---------------------------------------------------------------
U = model.get_layer("user_embedding").get_weights()[0]
I = model.get_layer("item_embedding").get_weights()[0]
B = model.get_layer("item_bias").get_weights()[0].ravel()

train_items = train_df.groupby("user")["item"].apply(set).to_dict()
test_items = test_df.groupby("user")["item"].apply(set).to_dict()
popularity = train_df.item.value_counts().reindex(range(n_items), fill_value=0).values.astype(float)
eval_users = rng.choice(list(test_items.keys()), size=min(2000, len(test_items)), replace=False)


def precision_recall_at_k(score_fn, k):
    p, r = [], []
    for u in eval_users:
        s = score_fn(u).copy()
        s[list(train_items.get(u, []))] = -np.inf            # don't recommend already-seen items
        top = np.argpartition(-s, k)[:k]
        hits = len(set(top) & test_items[u])
        p.append(hits / k); r.append(hits / len(test_items[u]))
    return np.mean(p), np.mean(r)


rows = []
for k in K_VALUES:
    pm, rm = precision_recall_at_k(lambda u: I @ U[u] + B, k)
    pp, rp = precision_recall_at_k(lambda u: popularity, k)
    rows.append({"K": k, "Precision@K (model)": pm, "Recall@K (model)": rm,
                 "Precision@K (popularity)": pp, "Recall@K (popularity)": rp})
res = pd.DataFrame(rows).set_index("K")
print(res.round(4))
res.plot(kind="bar", figsize=(9, 4), rot=0, title="Model vs popularity baseline")
plt.tight_layout(); plt.show()

u = eval_users[0]
s = I @ U[u] + B
s[list(train_items[u])] = -np.inf
print(f"User {u}: top-10 recommended items {np.argsort(-s)[:10].tolist()}")
print(f"User {u}: actually interacted (held-out) {sorted(test_items[u])}")
