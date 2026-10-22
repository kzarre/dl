# Deep Learning Practicals 1–25 (TensorFlow / Keras)

Every script is standalone and follows the same exam order:
**data preprocessing → `model.summary()` → training curves → evaluation**.

## Running
Best on **Google Colab with a GPU** (Runtime → Change runtime type → T4 GPU).
Paste a script into a cell, or upload it and run `!python pXX_name.py`
(plots show inline when pasted into a cell).

Extra packages (most are preinstalled on Colab):
```
pip install tensorflow tensorflow-datasets tensorflow-hub scikit-learn pandas matplotlib nltk seqeval networkx
```

## Datasets
| # | Dataset | How it loads |
|---|---------|--------------|
| 1, 6, 23 | MNIST | `keras.datasets` (auto) |
| 2 | California Housing | `sklearn` (auto) |
| 3, 7 | CIFAR-10 | `keras.datasets` (auto) |
| 4, 18 | Cats vs Dogs | `tensorflow_datasets` (auto, ~800 MB) |
| 5 | IMDB | `keras.datasets` (auto) |
| 8 | Flickr8k | auto-download from GitHub (~1 GB) |
| **9** | **Retailrocket** | **manual: download `events.csv` from Kaggle (retailrocket/ecommerce-dataset)** |
| **10** | **Sentiment140** | **manual: download `training.1600000.processed.noemoticon.csv` from Kaggle (kazanova/sentiment140)** |
| 11, 17 | Fashion-MNIST | `keras.datasets` (auto) |
| 12 | AG News | `tensorflow_datasets` (auto) |
| 13 | Fashion-MNIST | `keras.datasets` (auto) |
| 14, 15 | Breast Cancer / Wine | `sklearn` (auto) |
| 16 | Jena Climate | auto-download |
| 19 | Pascal VOC 2007 | `tensorflow_datasets` (auto, ~900 MB) + TF-Hub model |
| 20 | Shakespeare | auto-download |
| 21 | CoNLL-2003 | auto-download from GitHub mirror |
| 22 | English–French | auto-download |
| 24 | Credit Card Fraud | auto-download |
| 25 | Oxford-IIIT Pet | `tensorflow_datasets` (auto) |

## Exam tips
- Epoch counts are set for practice speed. If the exam is time-limited, lower `EPOCHS` at the top of the script.
- The slow ones: 7 (seven training runs), 8 (feature extraction + training), 19 (VOC download), 22 (translation decoding).
- Be ready to explain: why normalise, why dropout/regularisation, train vs validation curve gap (overfitting),
  and what each evaluation metric means.
