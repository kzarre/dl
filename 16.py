"""
Practical 16: Time-Series Forecasting Using LSTM
Forecast the next hour's temperature from the past 24 hours;
compare predicted vs actual with MAE and RMSE.
Dataset: Jena Climate 2009-2016 (downloaded automatically)
"""
import os
import glob
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
PAST = 24           # hours of history per sample
BATCH = 256

# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zpath = keras.utils.get_file(fname="jena_climate_2009_2016.csv.zip", origin=url, cache_dir=".", cache_subdir="downloads")
if not os.path.exists("jena"):
    with zipfile.ZipFile(zpath) as z:
        z.extractall("jena")
df = pd.read_csv(glob.glob("jena/**/*.csv", recursive=True)[0])
print(df.head()); print("Raw shape (10-minute readings):", df.shape)

# ---------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------
df = df[5::6].copy()                                             # one reading per hour
df.index = pd.to_datetime(df.pop("Date Time"), format="%d.%m.%Y %H:%M:%S")
df.loc[df["wv (m/s)"] < 0, "wv (m/s)"] = 0.0                     # fix -9999 sensor errors
hour = df.index.hour + df.index.minute / 60
df["day_sin"], df["day_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
doy = df.index.dayofyear
df["year_sin"], df["year_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)

FEATURES = ["T (degC)", "p (mbar)", "rh (%)", "VPmax (mbar)", "wv (m/s)",
            "day_sin", "day_cos", "year_sin", "year_cos"]
print("Hourly samples:", len(df))
df["T (degC)"].plot(figsize=(12, 3), title="Temperature (degC)"); plt.show()

data = df[FEATURES].values.astype("float32")
n = len(data)
train_end, val_end = int(0.7 * n), int(0.9 * n)                  # chronological split, no shuffling
mean, std = data[:train_end].mean(axis=0), data[:train_end].std(axis=0)
data_n = (data - mean) / std                                     # normalise with TRAIN statistics only


def make_ds(start, end, shuffle):
    # inputs: rows [i, i+PAST) ; target: temperature at row i+PAST (the next hour)
    return keras.utils.timeseries_dataset_from_array(
        data_n[start:end - PAST], targets=data_n[start + PAST:end, 0],
        sequence_length=PAST, batch_size=BATCH, shuffle=shuffle)


train_ds = make_ds(0, train_end, True)
val_ds = make_ds(train_end, val_end, False)
test_ds = make_ds(val_end, n, False)
for xb, yb in train_ds.take(1):
    print("Input batch:", xb.shape, "Target batch:", yb.shape)

# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(PAST, len(FEATURES))),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1),
], name="jena_lstm")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
model.summary()

history = model.fit(train_ds, validation_data=val_ds, epochs=15, verbose=2,
                    callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, m in zip(axes, ["loss", "mae"]):
    ax.plot(history.history[m], label="train"); ax.plot(history.history["val_" + m], label="validation")
    ax.set_title(f"{m} (normalised units)"); ax.set_xlabel("epoch"); ax.legend()
plt.tight_layout(); plt.show()

# ---------------------------------------------------------------
# Evaluation in degrees Celsius
# ---------------------------------------------------------------
denorm = lambda v: v * std[0] + mean[0]
y_true = denorm(np.concatenate([y.numpy() for _, y in test_ds]))
y_pred = denorm(model.predict(test_ds, verbose=0).ravel())
y_naive = denorm(np.concatenate([x[:, -1, 0].numpy() for x, _ in test_ds]))   # "same as last hour"

mae = lambda a, b: np.mean(np.abs(a - b))
rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
print(f"LSTM          MAE = {mae(y_true, y_pred):.3f} degC   RMSE = {rmse(y_true, y_pred):.3f} degC")
print(f"Naive (last)  MAE = {mae(y_true, y_naive):.3f} degC   RMSE = {rmse(y_true, y_naive):.3f} degC")

H = 24 * 7
plt.figure(figsize=(13, 4))
plt.plot(y_true[:H], label="actual"); plt.plot(y_pred[:H], label="LSTM prediction", alpha=0.8)
plt.title("Next-hour temperature: first week of test set"); plt.xlabel("hour"); plt.ylabel("degC"); plt.legend()
plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(y_true, y_pred, s=2, alpha=0.3)
lim = [y_true.min(), y_true.max()]; plt.plot(lim, lim, "r--")
plt.xlabel("actual (degC)"); plt.ylabel("predicted (degC)"); plt.title("Predicted vs actual"); plt.show()
