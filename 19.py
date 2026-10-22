"""
Practical 19: Object Detection Using Transfer Learning
A detector pretrained on COCO (SSD MobileNet V2 from TensorFlow Hub) is
transferred to Pascal VOC 2007: COCO classes are mapped to VOC classes,
boxes are drawn, and performance is measured with IoU and mAP@0.5.
Dataset: voc/2007 (tensorflow_datasets)
pip install tensorflow-hub tensorflow-datasets
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from collections import defaultdict

N_EVAL = 500             # test images used for mAP (increase for a more precise score)
DRAW_THRESHOLD = 0.5

# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
test_ds, info = tfds.load("voc/2007", split="test", with_info=True)
VOC = info.features["objects"]["label"].names
print("VOC classes:", VOC)
print("Test images:", info.splits["test"].num_examples)

# COCO category id -> VOC class name (the transfer mapping)
COCO_TO_VOC = {5: "aeroplane", 2: "bicycle", 16: "bird", 9: "boat", 44: "bottle", 6: "bus", 3: "car",
               17: "cat", 62: "chair", 21: "cow", 67: "diningtable", 18: "dog", 19: "horse",
               4: "motorbike", 1: "person", 64: "pottedplant", 20: "sheep", 63: "sofa", 7: "train", 72: "tvmonitor"}

# ---------------------------------------------------------------
# Pretrained detector
# ---------------------------------------------------------------
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
print("Architecture: SSD (single-shot detector) with MobileNetV2 backbone, trained on COCO (90 classes).")
print("Input: uint8 image [1, H, W, 3]; output: boxes [ymin, xmin, ymax, xmax] (normalised), classes, scores")


def detect(image):
    out = detector(tf.expand_dims(image, 0))
    boxes = out["detection_boxes"][0].numpy()
    classes = out["detection_classes"][0].numpy().astype(int)
    scores = out["detection_scores"][0].numpy()
    keep = [i for i, c in enumerate(classes) if c in COCO_TO_VOC]
    return boxes[keep], [COCO_TO_VOC[classes[i]] for i in keep], scores[keep]


def iou(a, b):
    y1, x1 = max(a[0], b[0]), max(a[1], b[1])
    y2, x2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, y2 - y1) * max(0, x2 - x1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------
# Run detection on test images
# ---------------------------------------------------------------
detections = defaultdict(list)          # class -> [(img_id, score, box)]
ground_truth = {}                        # (img_id, class) -> {"boxes", "difficult", "matched"}
samples = []
for img_id, ex in enumerate(tfds.as_numpy(test_ds.take(N_EVAL))):
    image = ex["image"]
    gt_boxes, gt_labels, gt_diff = ex["objects"]["bbox"], ex["objects"]["label"], ex["objects"]["is_difficult"]
    for box, lbl, diff in zip(gt_boxes, gt_labels, gt_diff):
        g = ground_truth.setdefault((img_id, VOC[lbl]), {"boxes": [], "difficult": [], "matched": []})
        g["boxes"].append(box); g["difficult"].append(bool(diff)); g["matched"].append(False)
    boxes, labels, scores = detect(image)
    for b, l, s in zip(boxes, labels, scores):
        detections[l].append((img_id, s, b))
    if img_id < 6:
        samples.append((image, gt_boxes, [VOC[l] for l in gt_labels], boxes, labels, scores))
    if (img_id + 1) % 100 == 0:
        print(f"processed {img_id + 1}/{N_EVAL}")


# ---------------------------------------------------------------
# Evaluation: per-class AP at IoU 0.5 (VOC protocol) and mean IoU
# ---------------------------------------------------------------
def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0])); mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


aps, matched_ious = {}, []
for cls in VOC:
    npos = sum(sum(not d for d in g["difficult"]) for (i, c), g in ground_truth.items() if c == cls)
    if npos == 0:
        continue
    dets = sorted(detections[cls], key=lambda d: -d[1])
    tp, fp = [], []
    for img_id, score, box in dets:
        g = ground_truth.get((img_id, cls))
        best, best_j = 0.0, -1
        if g:
            for j, gb in enumerate(g["boxes"]):
                o = iou(box, gb)
                if o > best: best, best_j = o, j
        if best >= 0.5:
            if g["difficult"][best_j]:
                continue                                  # ignored, as in the VOC protocol
            if not g["matched"][best_j]:
                g["matched"][best_j] = True; tp.append(1); fp.append(0); matched_ious.append(best)
            else:
                tp.append(0); fp.append(1)                # duplicate detection
        else:
            tp.append(0); fp.append(1)
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / npos if len(tp) else np.array([0.0])
    prec = tp / np.maximum(tp + fp, 1e-9) if len(tp) else np.array([0.0])
    aps[cls] = voc_ap(rec, prec)

print("\nPer-class AP@0.5:")
for c, ap in sorted(aps.items(), key=lambda x: -x[1]):
    print(f"  {c:12s} {ap:.3f}")
print(f"\nmAP@0.5 over {len(aps)} classes: {np.mean(list(aps.values())):.4f}")
print(f"Mean IoU of correctly matched detections: {np.mean(matched_ious):.4f}")

plt.figure(figsize=(10, 4))
plt.bar(list(aps.keys()), list(aps.values())); plt.xticks(rotation=60); plt.ylabel("AP@0.5")
plt.title(f"Per-class AP (mAP = {np.mean(list(aps.values())):.3f})"); plt.tight_layout(); plt.show()

# ---------------------------------------------------------------
# Visualise: ground truth (green) vs predictions (red)
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, (image, gtb, gtl, boxes, labels, scores) in zip(axes.ravel(), samples):
    h, w = image.shape[:2]
    ax.imshow(image); ax.axis("off")
    for b, l in zip(gtb, gtl):
        ax.add_patch(patches.Rectangle((b[1] * w, b[0] * h), (b[3] - b[1]) * w, (b[2] - b[0]) * h,
                                       fill=False, ec="lime", lw=2))
    for b, l, s in zip(boxes, labels, scores):
        if s < DRAW_THRESHOLD:
            continue
        ax.add_patch(patches.Rectangle((b[1] * w, b[0] * h), (b[3] - b[1]) * w, (b[2] - b[0]) * h,
                                       fill=False, ec="red", lw=2))
        ax.text(b[1] * w, b[0] * h - 3, f"{l} {s:.2f}", color="white", fontsize=8, backgroundcolor="red")
plt.suptitle("Green = ground truth, red = detections"); plt.tight_layout(); plt.show()
