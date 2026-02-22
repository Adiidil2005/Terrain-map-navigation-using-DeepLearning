# Terrain-map-navigation-using-DeepLearning

#  Drone Visual Geolocalization
### CNN Siamese Network · University-1652 · TensorFlow · CPU Optimized

> A deep learning system that predicts GPS coordinates from drone downward-looking images by matching them against a satellite image database — no GPS hardware required.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Metrics](#metrics)
- [Results](#results)
- [Tips for Better Accuracy](#tips-for-better-accuracy)
- [Known Issues](#known-issues)

---

## Overview

This project implements a **visual geolocalization pipeline** for drones operating in GPS-denied environments. Given a nadir (downward-looking) drone image, the system:

1. Embeds the image into a compact feature vector using a CNN
2. Searches a pre-built satellite image database for the closest match
3. Returns the predicted **latitude and longitude** of the drone

The model is trained on the [University-1652](https://github.com/layumi/University1652-Baseline) dataset using a **Siamese network with Triplet Loss**, and is exported to **TFLite** for edge deployment on drone companion computers.

---

## How It Works

```
Drone Camera (nadir image)
        │
        ▼
┌───────────────────┐
│  MobileNetV3Small │  ← Shared CNN Backbone (ImageNet pretrained)
│  (feature extractor)│
└────────┬──────────┘
         │  576-dim features
         ▼
┌───────────────────┐
│  Embedding Head   │  ← Dense(1024) → Dropout → Dense(256) → L2Norm
└────────┬──────────┘
         │  256-dim unit vector
         ▼
┌───────────────────┐
│  kNN Search       │  ← Cosine similarity against satellite DB
│  (MapVectorDB)    │
└────────┬──────────┘
         │
         ▼
  Predicted (lat, lon)
```

**Training:** The network learns to pull drone and satellite images of the **same location** closer together, while pushing images of **different locations** apart — using Triplet Loss with hard negative mining.

---

## Project Structure

```
MiniProj/
├── drone_geo_fast.py          ← Training + evaluation (single file)
├── predict_and_evaluate.py    ← Prediction + metrics script
├── README.md                  ← This file
│
├── university1652-kml/        ← Real GPS coordinates
│   ├── _MACOSX/               (auto-generated, ignored)
│   └── first-key/
│       ├── 0001.kml
│       ├── 0002.kml
│       └── ...                (1652 KML files, one per location)
│
├── U1652/
│   └── University-Release/    ← Main dataset
│       ├── train/
│       │   ├── drone/         ← Drone-view training images
│       │   ├── satellite/     ← Satellite-view training images
│       │   ├── street/        (not used)
│       │   └── google/        (not used)
│       └── test/
│           ├── query_drone/       ← Test queries
│           ├── gallery_satellite/ ← Test gallery
│           └── ...
│
├── outputs_fast/              ← Auto-created during training
│   ├── best_model.weights.h5
│   ├── drone_geo.tflite
│   ├── map_db.pkl
│   ├── training_log.csv
│   ├── training_loss.png
│   └── retrieval_results.png
│
└── eval_outputs/              ← Auto-created during evaluation
    ├── evaluation_report.png
    ├── per_query_results.csv
    └── sample_XX_locYYYY.png
```

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| TensorFlow (CPU) | 2.15.0 |
| NumPy | 1.26.4 |
| scikit-learn | 1.4.2 |
| Matplotlib | 3.8.4 |
| Pillow | 10.3.0 |
| tqdm | 4.66.4 |

**Hardware:** CPU only — optimized for **AMD Ryzen 5 5500U** (6 cores, 16 GB RAM).
No GPU required.

---

## Setup

### 1. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install tensorflow-cpu numpy scikit-learn matplotlib Pillow tqdm
```

### 3. Set dataset path

Open `drone_geo_fast.py` and update **line 47**:
```python
DATASET_PATH = r"C:\Users\adity\MiniProj\U1652\University-Release"
```

Open `predict_and_evaluate.py` and update **line 22**:
```python
DATASET_PATH = r"C:\Users\adity\MiniProj\U1652\University-Release"
```

---

## Usage

### Training

```bash
python drone_geo_fast.py
```

This runs all 6 steps automatically:
1. Loads drone + satellite images
2. Builds triplets (anchor / positive / negative)
3. Trains Siamese CNN for N epochs
4. Saves best model checkpoint
5. Builds the satellite map vector database
6. Exports to TFLite and evaluates Recall@K

> ⏱ Expected time on Ryzen 5 5500U: **~3–55 minutes** depending on config

---

### Full Evaluation

```bash
python predict_and_evaluate.py
```

Evaluates on all test query images and saves:
- Accuracy, Precision, Recall, F1 Score
- Recall@1, @5, @10
- GPS error metrics (mean, median, % within 500m)
- 6-panel evaluation report plot
- Sample prediction visualization grids

---

### Single Image Prediction

```bash
# Predict only
python predict_and_evaluate.py --image path\to\drone.jpg

# Predict with known ground truth location
python predict_and_evaluate.py --image path\to\drone.jpg --true_loc 0042
```

---

### Control Number of Sample Visualizations

```bash
python predict_and_evaluate.py --samples 10
```

---

## Model Architecture

### SharedCNNBackbone
- Base model: `MobileNetV3Small` (ImageNet pretrained)
- Input shape: `(128, 128, 3)` or `(160, 160, 3)`
- Output: 576-dim feature vector via Global Average Pooling
- Last N layers fine-tuned (N configurable)

### EmbeddingHead
```
Dense(1024, ReLU)
      ↓
Dropout(0.2)
      ↓
Dense(256, Linear)
      ↓
L2 Normalize  →  256-dim unit vector
```

### Training: Siamese with Triplet Loss
```
Triplet Loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)

where:
  anchor   = drone image of location X
  positive = satellite image of location X
  negative = satellite image of location Y ≠ X
  margin   = 0.3 (or 0.5 for stronger separation)
```

### Inference
```
Query drone image → Embedding → kNN (cosine) → Top-K satellite matches → GPS coordinates
```

---

## Configuration

All settings are inside the `Config` class at the top of each script.

| Parameter | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | `(128, 128)` | Input resolution. Larger = more accurate, slower |
| `EMBEDDING_DIM` | `256` | Output embedding size. `512` for richer features |
| `EPOCHS` | `8` | Training epochs. `20` for better convergence |
| `BATCH_SIZE` | `16` | Keep at 16 for CPU RAM limits |
| `LEARNING_RATE` | `2e-4` | Adam optimizer LR |
| `MARGIN` | `0.3` | Triplet loss margin. `0.5` for stronger separation |
| `MAX_LOCATIONS` | `400` | Locations used for training (max 1652) |
| `TRIPLETS_PER_LOC` | `4` | Triplets generated per location per epoch |
| `UNFREEZE_LAST_N` | `10` | Backbone layers fine-tuned |
| `TOP_K` | `5` | Top-K matches returned at inference |
| `MAX_EVAL_IMAGES` | `300` | Query images evaluated (set `None` for all) |
| `DISTANCE_THRESHOLD_KM` | `0.5` | GPS threshold for "correct" prediction |

### ⚡ Fast Config (~3–10 min)
```python
IMAGE_SIZE = (128, 128), EPOCHS = 8, MAX_LOCATIONS = 400, EMBEDDING_DIM = 256
```

### 🎯 Accurate Config (~45–55 min)
```python
IMAGE_SIZE = (160, 160), EPOCHS = 20, MAX_LOCATIONS = 1200, EMBEDDING_DIM = 512,
MARGIN = 0.5, UNFREEZE_LAST_N = 30
```

---

## Output Files

| File | Description |
|---|---|
| `outputs_fast/best_model.weights.h5` | Best Keras model weights (lowest training loss) |
| `outputs_fast/drone_geo.tflite` | Quantized TFLite model (~2–5 MB) for edge deployment |
| `outputs_fast/map_db.pkl` | Satellite embedding database (`vectors` + `ids`) |
| `outputs_fast/training_log.csv` | Epoch-by-epoch loss and training time |
| `outputs_fast/training_loss.png` | Loss curve plot |
| `outputs_fast/retrieval_results.png` | Sample retrieval visualization |
| `eval_outputs/evaluation_report.png` | 6-panel metrics report |
| `eval_outputs/per_query_results.csv` | Per-image: true loc, pred loc, correct, GPS error |
| `eval_outputs/sample_XX_locYYYY.png` | Prediction grids (drone → predicted → true → rank 2–4) |

### Prediction Visualization Layout

Each `sample_XX.png` shows a 6-column grid:

```
┌─────────────┬──────────────┬──────────────┬───────┬───────┬───────┐
│   QUERY     │  PREDICTED   │    TRUE      │ Rank  │ Rank  │ Rank  │
│   (Drone)   │  Satellite   │  Satellite   │   2   │   3   │   4   │
│ Blue border │ 🟢 Correct   │ Gold border  │       │       │       │
│             │ 🔴 Wrong     │              │       │       │       │
└─────────────┴──────────────┴──────────────┴───────┴───────┴───────┘
```

---

## Metrics

| Metric | What it measures |
|---|---|
| **Accuracy** | Fraction of queries with correct top-1 predicted location |
| **Precision** | Of all predictions for a class, how many were correct (weighted avg) |
| **Recall** | Of all true instances of a class, how many were retrieved (weighted avg) |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **Recall@K** | Is the correct location in the top-K results? (K = 1, 5, 10) |
| **Mean GPS Error** | Average haversine distance (km) between predicted and true coordinates |
| **Median GPS Error** | Median haversine distance — less sensitive to outliers |
| **Within 500m %** | Fraction of predictions within 500 metres of ground truth |

---

## Results

### Fast Config (3.5 minutes · 128×128 · 8 epochs · 400 locations)

| Metric | Score |
|---|---|
| Recall@1 | 0.0600 |
| Recall@5 | 0.1500 |
| Recall@10 | 0.1900 |

> These are baseline scores from the initial prototype run. Use the accurate config for significantly better results.

---

## Tips for Better Accuracy

- **Slow training?** Reduce `EPOCHS` to 5 and `MAX_LOCATIONS` to 200
- **Out of memory?** Reduce `BATCH_SIZE` to 8
- **Want better accuracy?** Use the accurate config (see Configuration section)
- **Fine-tuning after initial training?** Set `UNFREEZE_LAST_N = 60` and retrain with `LEARNING_RATE = 1e-5`
- **Deployment?** Use `drone_geo.tflite` with the `TFLiteEngine` class in `predict_and_evaluate.py`

---

## Known Issues

- **Low Recall@1 on first run (~0.06)** is expected with the fast config. The model needs more epochs and data to converge — use the accurate config for real results.
- **Satellite images not found** for some locations during visualization means that location only appears in training, not the test gallery. The prediction still works correctly.
- **GPS coordinates not in KML** — all 1652 locations have KML files so this should not occur. If it does, synthetic coordinates are generated automatically as a fallback.
- **Path errors on Windows** — always use raw strings `r"C:\path\to\folder"` to avoid backslash issues.

---

## Dataset

**University-1652** — A multi-view dataset of 1,652 university buildings captured from drone, satellite, street, and Google view.

| Split | Images | Locations |
|---|---|---|
| Train (drone) | 37,855 | 1,652 |
| Train (satellite) | 54,754 | 1,652 |
| Test query (drone) | 54,754 | 701 |
| Test gallery (satellite) | 51,355 | 701 |

> Repository: [https://github.com/layumi/University1652-Baseline](https://github.com/layumi/University1652-Baseline)

---

*Project: Drone Visual Geolocalization — Mini Project*
*Model: MobileNetV3Small + Siamese Triplet Loss (TensorFlow 2.x)*
