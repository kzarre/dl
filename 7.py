import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
EPOCHS = 10
BATCH = 128

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()
print("Train:", x_train.shape, "Test:", x_test.shape)
STEPS_PER_EPOCH = int(np.ceil(len(x_train) * 0.9 / BATCH))

def build_cnn(init="he_normal"):
    return keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu", kernel_initializer=init),
        layers.Conv2D(32, 3, padding="same", activation="relu", kernel_initializer=init),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer=init),
        layers.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer=init),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu", kernel_initializer=init),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ], name=f"cnn_{init}")

build_cnn().summary()

for init in ["glorot_normal", "he_normal"]:
    w = build_cnn(init).layers[2].get_weights()[0]
    print(f"{init:14s} initial std of conv layer 3 weights = {w.std():.4f}")

results, histories = [], {}

def run(name, init, make_optimizer, callbacks=None):
    tf.keras.backend.clear_session()
    model = build_cnn(init)
    model.compile(optimizer=make_optimizer(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_split=0.1,
                  callbacks=callbacks or [], verbose=2)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    val = np.array(h.history["val_accuracy"])
    reached = int(np.argmax(val >= 0.60)) + 1 if (val >= 0.60).any() else None
    results.append({"experiment": name, "test_acc": acc, "best_val_acc": val.max(),
                    "final_train_loss": h.history["loss"][-1], "epochs_to_60%_val": reached})
    histories[name] = h
    print(f"==> {name}: test accuracy {acc:.4f}")

run("Xavier + Adam", "glorot_normal", lambda: keras.optimizers.Adam(1e-3))
run("He + Adam", "he_normal", lambda: keras.optimizers.Adam(1e-3))

run("He + SGD", "he_normal", lambda: keras.optimizers.SGD(0.01, momentum=0.9))
run("He + RMSprop", "he_normal", lambda: keras.optimizers.RMSprop(1e-3))

exp_decay = keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=STEPS_PER_EPOCH, decay_rate=0.8)
cos_decay = keras.optimizers.schedules.CosineDecay(0.01, decay_steps=EPOCHS * STEPS_PER_EPOCH)
run("SGD + exponential decay", "he_normal", lambda: keras.optimizers.SGD(exp_decay, momentum=0.9))
run("SGD + cosine decay", "he_normal", lambda: keras.optimizers.SGD(cos_decay, momentum=0.9))
step_cb = keras.callbacks.LearningRateScheduler(lambda epoch, lr: 0.01 * (0.5 ** (epoch // 3)))
run("SGD + step decay", "he_normal", lambda: keras.optimizers.SGD(0.01, momentum=0.9), [step_cb])

epochs = np.arange(EPOCHS)
plt.figure(figsize=(7, 4))
plt.plot(epochs, [0.01] * EPOCHS, label="constant")
plt.plot(epochs, [float(exp_decay(e * STEPS_PER_EPOCH)) for e in epochs], label="exponential")
plt.plot(epochs, [float(cos_decay(e * STEPS_PER_EPOCH)) for e in epochs], label="cosine")
plt.plot(epochs, [0.01 * 0.5 ** (e // 3) for e in epochs], label="step")
plt.title("Learning-rate schedules"); plt.xlabel("epoch"); plt.ylabel("LR"); plt.legend(); plt.show()

groups = {
    "Initialization": ["Xavier + Adam", "He + Adam"],
    "Optimizer": ["He + SGD", "He + Adam", "He + RMSprop"],
    "LR schedule": ["He + SGD", "SGD + exponential decay", "SGD + step decay", "SGD + cosine decay"],
}
fig, axes = plt.subplots(2, 3, figsize=(17, 8))
for col, (g, names) in enumerate(groups.items()):
    for n in names:
        axes[0, col].plot(histories[n].history["loss"], label=n)
        axes[1, col].plot(histories[n].history["val_accuracy"], label=n)
    axes[0, col].set_title(f"{g}: training loss (convergence)")
    axes[1, col].set_title(f"{g}: validation accuracy")
    for r in range(2): axes[r, col].legend(fontsize=8); axes[r, col].set_xlabel("epoch")
plt.tight_layout(); plt.show()

res = pd.DataFrame(results).set_index("experiment")
print(res.round(4))
res["test_acc"].plot(kind="barh", figsize=(8, 4), xlim=(0.3, 0.85), title="Test accuracy")
plt.tight_layout(); plt.show()

