"""
Practical 10: Sentiment Analysis on Network Graphs Using RNN
LSTM tweet sentiment classifier + user @mention graph coloured by sentiment.
Dataset: Sentiment140 (Kaggle: kazanova/sentiment140) ->
         training.1600000.processed.noemoticon.csv next to this script.
pip install networkx
"""
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

tf.random.set_seed(42)
CSV = "training.1600000.processed.noemoticon.csv"
SAMPLE = 200_000
VOCAB, MAXLEN = 20000, 40

# ---------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------
cols = ["target", "id", "date", "flag", "user", "text"]
df = pd.read_csv(CSV, encoding="latin-1", names=cols)
df = df.sample(SAMPLE, random_state=42).reset_index(drop=True)
df["label"] = (df.target == 4).astype(int)                       # 0 = negative, 4 = positive
print(df[["user", "text", "label"]].head())
print("Class balance:", df.label.value_counts().to_dict())


def clean(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " @user ", t)
    t = re.sub(r"[^a-z@\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


df["clean"] = df.text.apply(clean)
df["mentions"] = df.text.apply(lambda t: [m.lower() for m in re.findall(r"@(\w+)", t)])

X_tr, X_te, y_tr, y_te = train_test_split(df.clean.values, df.label.values, test_size=0.2,
                                          random_state=42, stratify=df.label.values)
vectorizer = layers.TextVectorization(max_tokens=VOCAB, output_sequence_length=MAXLEN)
vectorizer.adapt(X_tr)
X_tr_seq, X_te_seq = vectorizer(X_tr).numpy(), vectorizer(X_te).numpy()
print("Vectorized:", X_tr_seq.shape, "| example:", X_tr[0], "->", X_tr_seq[0][:12])

# ---------------------------------------------------------------
# LSTM model
# ---------------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(MAXLEN,)),
    layers.Embedding(VOCAB, 64, mask_zero=True),
    layers.LSTM(64),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
], name="tweet_lstm")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
history = model.fit(X_tr_seq, y_tr, epochs=4, batch_size=256, validation_split=0.1, verbose=2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, m in zip(axes, ["loss", "accuracy"]):
    ax.plot(history.history[m], label="train"); ax.plot(history.history["val_" + m], label="validation")
    ax.set_title(m); ax.set_xlabel("epoch"); ax.legend()
plt.tight_layout(); plt.show()

y_pred = (model.predict(X_te_seq, verbose=0).ravel() > 0.5).astype(int)
print(classification_report(y_te, y_pred, target_names=["negative", "positive"], digits=4))
ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred), display_labels=["neg", "pos"]).plot(cmap="Blues")
plt.show()

# ---------------------------------------------------------------
# User interaction graph (@mentions)
# ---------------------------------------------------------------
df["sentiment"] = model.predict(vectorizer(df.clean.values).numpy(), batch_size=1024, verbose=0).ravel()

G = nx.DiGraph()
for user, mentions, s in zip(df.user.str.lower(), df.mentions, df.sentiment):
    for m in mentions:
        if m == user:
            continue
        if G.has_edge(user, m):
            G[user][m]["sent"].append(s)
        else:
            G.add_edge(user, m, sent=[s])
print(f"Graph: {G.number_of_nodes()} users, {G.number_of_edges()} mention edges")

# node sentiment = mean sentiment of tweets the user wrote; for users who only get
# mentioned, use the mean sentiment of tweets mentioning them
authored = df.groupby(df.user.str.lower())["sentiment"].mean().to_dict()
for n in G.nodes:
    if n in authored:
        G.nodes[n]["sentiment"] = authored[n]
    else:
        G.nodes[n]["sentiment"] = float(np.mean([s for _, _, d in G.in_edges(n, data=True) for s in d["sent"]]))
for u, v, d in G.edges(data=True):
    d["mean_sent"] = float(np.mean(d["sent"]))

pos_edges = sum(d["mean_sent"] > 0.5 for _, _, d in G.edges(data=True))
print(f"Positive mention edges: {pos_edges / G.number_of_edges():.2%}")
top = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
print("Most connected users:", [(u, d, round(G.nodes[u]["sentiment"], 2)) for u, d in top])

# visualise the neighbourhoods of the 25 most-mentioned users
hubs = [u for u, _ in sorted(G.in_degree, key=lambda x: x[1], reverse=True)[:25]]
nodes = set(hubs)
for h in hubs:
    nodes.update(list(G.predecessors(h))[:15])
H = G.subgraph(nodes)
pos = nx.spring_layout(H, seed=42, k=0.3)
node_colors = [H.nodes[n]["sentiment"] for n in H.nodes]
sizes = [30 + 25 * H.in_degree(n) for n in H.nodes]
plt.figure(figsize=(13, 10))
nx.draw_networkx_edges(H, pos, alpha=0.25, arrows=False,
                       edge_color=[d["mean_sent"] for _, _, d in H.edges(data=True)], edge_cmap=plt.cm.RdYlGn)
nc = nx.draw_networkx_nodes(H, pos, node_color=node_colors, cmap=plt.cm.RdYlGn, vmin=0, vmax=1, node_size=sizes)
nx.draw_networkx_labels(H, pos, labels={h: h for h in hubs[:12]}, font_size=8)
plt.colorbar(nc, label="predicted sentiment (0 = negative, 1 = positive)")
plt.title("@mention network coloured by sentiment"); plt.axis("off"); plt.show()

plt.hist([G.nodes[n]["sentiment"] for n in G.nodes], bins=30, color="tab:purple")
plt.title("Sentiment distribution across users"); plt.xlabel("mean predicted sentiment"); plt.show()
