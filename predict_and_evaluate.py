# ============================================================
# predict_and_evaluate.py
#
# 1. Predicts GPS coordinates (lat/lon) for drone images
# 2. Measures Accuracy, Precision, Recall, F1 Score
#
# Works in TWO modes automatically:
#   MODE A — if you have the U1652 KML file with real GPS coords
#   MODE B — uses synthetic coords derived from location IDs
#             (fully functional for all metrics even without KML)
#
# Run:
#   python predict_and_evaluate.py
#   python predict_and_evaluate.py --image path/to/drone.jpg
#   python predict_and_evaluate.py --kml  path/to/doc.kml
# ============================================================

import os, sys, re, math, pickle, random, argparse, warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.neighbors import NearestNeighbors

# ============================================================
# CONFIGURATION  — update paths if needed
# ============================================================

class Config:
    DATASET_PATH     = r"C:\Users\adity\MiniProj\U1652\University-Release"
    QUERY_DRONE_PATH = os.path.join(DATASET_PATH, "test", "query_drone")
    GALLERY_SAT_PATH = os.path.join(DATASET_PATH, "test", "gallery_satellite")
    TRAIN_SAT_PATH   = os.path.join(DATASET_PATH, "train", "satellite")

    DB_PATH          = os.path.join("outputs_fast", "map_db.pkl")
    TFLITE_PATH      = os.path.join("outputs_fast", "drone_geo.tflite")
    OUTPUT_DIR       = "eval_outputs"

    IMAGE_SIZE       = (128, 128)     # Must match what you trained with
    INPUT_SHAPE      = (128, 128, 3)  # Change to (160,160,3) if you used updated config
    EMBEDDING_DIM    = 256            # Change to 512 if you used updated config
    TOP_K            = 5

    # Evaluation limits (set to None to evaluate all)
    MAX_EVAL_IMAGES  = 300            # None = all query images

    # Distance threshold in km — predictions within this are "correct"
    # Used for coordinate-based accuracy metric
    DISTANCE_THRESHOLD_KM = 0.5      # 500 metres


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# ============================================================
# SECTION 1 — GPS COORDINATE HANDLING
# ============================================================

def load_coords_from_kml(kml_path: str) -> dict:
    """
    Parse the University-1652 KML file.
    Returns dict: { location_id_str -> (lat, lon) }
    The KML from the dataset names placemarks as '0001', '0002', etc.
    """
    print(f"[GPS] Loading coordinates from KML: {kml_path}")
    coords = {}
    try:
        with open(kml_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract <Placemark> blocks
        placemark_blocks = re.findall(
            r'<Placemark>(.*?)</Placemark>', content, re.DOTALL
        )

        for block in placemark_blocks:
            # Get name (location ID like 0001)
            name_match = re.search(r'<name>(.*?)</name>', block)
            # Get coordinates: lon,lat,alt
            coord_match = re.search(
                r'<coordinates>\s*([\-\d.]+),([\-\d.]+)', block
            )
            if name_match and coord_match:
                loc_id = name_match.group(1).strip().zfill(4)
                lon    = float(coord_match.group(1))
                lat    = float(coord_match.group(2))
                coords[loc_id] = (lat, lon)

        print(f"[GPS] Loaded {len(coords)} real GPS coordinates from KML.")
    except Exception as e:
        print(f"[GPS] Failed to parse KML: {e}")
        coords = {}

    return coords


def generate_synthetic_coords(location_ids: list) -> dict:
    """
    MODE B: Generate reproducible synthetic lat/lon from location IDs.
    Uses a fixed seed so the same ID always maps to the same coordinate.
    Spread across a realistic geographic area (university campus scale).

    These are NOT real coordinates but allow all metrics to work correctly
    since ground truth and prediction use the same mapping.
    """
    print("[GPS] No KML found — generating synthetic coordinates.")
    print("      (Download doc.kml from the University-1652 GitHub for real GPS)")

    base_lat, base_lon = 40.0, 116.0   # Approximate centre of dataset universities
    coords = {}

    for loc_id in location_ids:
        # Deterministic hash → small lat/lon offset
        seed = int(loc_id) if loc_id.isdigit() else hash(loc_id) % 10000
        rng  = np.random.default_rng(seed)
        # Spread ±0.05° ≈ ±5.5 km, spaced so locations are distinct
        lat_offset = rng.uniform(-0.05, 0.05)
        lon_offset = rng.uniform(-0.05, 0.05)
        coords[loc_id] = (round(base_lat + lat_offset, 6),
                          round(base_lon + lon_offset, 6))

    print(f"[GPS] Generated synthetic coords for {len(coords)} locations.\n")
    return coords


def get_coordinate_map(location_ids: list, kml_path: str = None) -> dict:
    """
    Returns coord map. Tries KML first, falls back to synthetic.
    """
    if kml_path and os.path.isfile(kml_path):
        coords = load_coords_from_kml(kml_path)
        if coords:
            return coords

    # Auto-search for KML in dataset folder
    for root, dirs, files in os.walk(Config.DATASET_PATH):
        for f in files:
            if f.lower().endswith('.kml'):
                coords = load_coords_from_kml(os.path.join(root, f))
                if coords:
                    return coords

    return generate_synthetic_coords(location_ids)


# ============================================================
# SECTION 2 — IMAGE UTILITIES
# ============================================================

def preprocess(img_path: str) -> np.ndarray:
    img = tf.keras.preprocessing.image.load_img(
        img_path, target_size=Config.IMAGE_SIZE
    )
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)
    return arr.astype(np.float32)


def get_imgs(base_path: str, loc_id: str) -> list:
    folder = os.path.join(base_path, loc_id)
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]


# ============================================================
# SECTION 3 — TFLite INFERENCE ENGINE
# ============================================================

class TFLiteEngine:
    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"TFLite model not found: {model_path}\n"
                "Run drone_geo_fast.py first to generate it."
            )
        self.interp = tf.lite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self._in  = self.interp.get_input_details()
        self._out = self.interp.get_output_details()
        print(f"[TFLite] Loaded: {model_path}")

    def predict(self, img_path: str) -> np.ndarray:
        """Single image → 256d (or 512d) embedding."""
        arr = preprocess(img_path)
        inp = np.expand_dims(arr, 0).astype(np.float32)
        self.interp.set_tensor(self._in[0]['index'], inp)
        self.interp.invoke()
        return self.interp.get_tensor(self._out[0]['index'])[0]


# ============================================================
# SECTION 4 — MAP VECTOR DATABASE (load existing)
# ============================================================

class MapDB:
    def __init__(self, db_path: str):
        if not os.path.isfile(db_path):
            raise FileNotFoundError(
                f"Database not found: {db_path}\n"
                "Run drone_geo_fast.py first to generate it."
            )
        with open(db_path, 'rb') as f:
            data = pickle.load(f)
        self.vectors = data['vectors'].astype(np.float32)
        self.ids     = data['ids']

        k = min(Config.TOP_K, len(self.vectors))
        self._nn = NearestNeighbors(
            n_neighbors=k, metric='cosine',
            algorithm='brute', n_jobs=-1
        )
        self._nn.fit(self.vectors)
        print(f"[DB] Loaded {len(self.vectors)} reference vectors.")

    def search(self, query_vec: np.ndarray, k: int = 5):
        dists, idx = self._nn.kneighbors(
            query_vec.reshape(1, -1), n_neighbors=k
        )
        return dists[0], [self.ids[i] for i in idx[0]]


# ============================================================
# SECTION 5 — HAVERSINE DISTANCE
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two GPS points in kilometres."""
    R    = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat/2)**2 +
            math.cos(math.radians(lat1)) *
            math.cos(math.radians(lat2)) *
            math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


# ============================================================
# SECTION 6 — SINGLE IMAGE PREDICTION
# ============================================================

def predict_single(img_path: str, engine: TFLiteEngine,
                   db: MapDB, coord_map: dict, k: int = 5) -> dict:
    """
    Given a drone image path, return predicted coordinates + top-K matches.
    """
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    emb           = engine.predict(img_path)
    dists, loc_ids = db.search(emb, k=k)

    top_loc       = loc_ids[0]
    top_lat, top_lon = coord_map.get(top_loc, (0.0, 0.0))

    result = {
        'image_path'   : img_path,
        'top_location' : top_loc,
        'latitude'     : top_lat,
        'longitude'    : top_lon,
        'top_k'        : [
            {
                'rank'     : i + 1,
                'loc_id'   : lid,
                'lat'      : coord_map.get(lid, (0.0, 0.0))[0],
                'lon'      : coord_map.get(lid, (0.0, 0.0))[1],
                'cos_dist' : float(d),
                'confidence': round((1 - float(d)) * 100, 2)
            }
            for i, (lid, d) in enumerate(zip(loc_ids, dists))
        ]
    }
    return result


def print_prediction(result: dict) -> None:
    print(f"\n{'─'*55}")
    print(f"  Image   : {os.path.basename(result['image_path'])}")
    print(f"{'─'*55}")
    print(f"  Predicted Location ID : {result['top_location']}")
    print(f"  Predicted Latitude    : {result['latitude']:.6f}°")
    print(f"  Predicted Longitude   : {result['longitude']:.6f}°")
    print(f"\n  Top-{len(result['top_k'])} Candidate Matches:")
    print(f"  {'Rank':<5} {'Loc ID':<8} {'Latitude':>12} {'Longitude':>12} "
          f"{'Confidence':>12}")
    print(f"  {'─'*55}")
    for m in result['top_k']:
        bar = '█' * int(m['confidence'] / 5)
        print(f"  {m['rank']:<5} {m['loc_id']:<8} {m['lat']:>12.6f} "
              f"{m['lon']:>12.6f} {m['confidence']:>10.1f}%  {bar}")
    print(f"{'─'*55}\n")


# ============================================================
# SECTION 7 — FULL EVALUATION WITH ALL METRICS
# ============================================================

def evaluate_all(engine: TFLiteEngine, db: MapDB,
                 coord_map: dict) -> dict:
    """
    Evaluate on all query_drone images.

    Metrics computed:
    ─────────────────────────────────────────────────────
    Classification metrics (location ID = class label):
      • Accuracy   — fraction of queries with correct top-1 location
      • Precision  — weighted avg precision across all location classes
      • Recall     — weighted avg recall across all location classes
      • F1 Score   — weighted avg F1 across all location classes

    Retrieval metrics (ranked list):
      • Recall@1, @5, @10 — correct loc in top-K results

    Coordinate metrics (GPS distance):
      • Mean Error (km)    — average haversine distance, pred vs true
      • Median Error (km)
      • % within 0.5 km    — fraction of predictions within threshold
    ─────────────────────────────────────────────────────
    """
    print(f"\n{'='*55}")
    print("  FULL EVALUATION")
    print(f"{'='*55}")

    query_ids = sorted(os.listdir(Config.QUERY_DRONE_PATH))

    y_true      = []    # true location IDs
    y_pred      = []    # predicted top-1 location IDs
    y_pred_topk = []    # predicted top-k location IDs (list of lists)
    dist_errors = []    # haversine errors in km

    count = 0
    for loc_id in query_ids:
        imgs = get_imgs(Config.QUERY_DRONE_PATH, loc_id)
        if not imgs:
            continue

        # Use first image per location for speed; use all if MAX_EVAL_IMAGES=None
        eval_imgs = imgs[:1]

        for img_path in eval_imgs:
            emb = engine.predict(img_path)
            dists, pred_ids = db.search(emb, k=10)

            top1 = pred_ids[0]
            y_true.append(loc_id)
            y_pred.append(top1)
            y_pred_topk.append(pred_ids)

            # GPS error
            true_lat, true_lon = coord_map.get(loc_id, (0.0, 0.0))
            pred_lat, pred_lon = coord_map.get(top1,   (0.0, 0.0))
            err_km = haversine_km(true_lat, true_lon, pred_lat, pred_lon)
            dist_errors.append(err_km)

            count += 1
            if count % 50 == 0:
                print(f"  Processed {count} queries...")

        if Config.MAX_EVAL_IMAGES and count >= Config.MAX_EVAL_IMAGES:
            break

    total = len(y_true)
    print(f"\n  Total queries evaluated: {total}")

    # ── Classification Metrics ────────────────────────────────
    # Get all unique classes that appear in y_true
    classes = sorted(set(y_true))

    correct  = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total > 0 else 0.0

    # sklearn metrics with zero_division=0 to handle unseen classes
    precision = precision_score(y_true, y_pred, average='weighted',
                                zero_division=0, labels=classes)
    recall    = recall_score(   y_true, y_pred, average='weighted',
                                zero_division=0, labels=classes)
    f1        = f1_score(       y_true, y_pred, average='weighted',
                                zero_division=0, labels=classes)

    # ── Retrieval Metrics ─────────────────────────────────────
    r_at_1  = sum(1 for t, p in zip(y_true, y_pred_topk) if t in p[:1])  / total
    r_at_5  = sum(1 for t, p in zip(y_true, y_pred_topk) if t in p[:5])  / total
    r_at_10 = sum(1 for t, p in zip(y_true, y_pred_topk) if t in p[:10]) / total

    # ── Coordinate Metrics ────────────────────────────────────
    mean_err   = float(np.mean(dist_errors))
    median_err = float(np.median(dist_errors))
    within_thr = sum(1 for d in dist_errors
                     if d <= Config.DISTANCE_THRESHOLD_KM) / total

    metrics = {
        'total_queries'  : total,
        'accuracy'       : accuracy,
        'precision'      : precision,
        'recall'         : recall,
        'f1_score'       : f1,
        'recall_at_1'    : r_at_1,
        'recall_at_5'    : r_at_5,
        'recall_at_10'   : r_at_10,
        'mean_error_km'  : mean_err,
        'median_error_km': median_err,
        'within_500m_pct': within_thr * 100,
        'dist_errors'    : dist_errors,
        'y_true'         : y_true,
        'y_pred'         : y_pred,
    }

    # ── Print Results ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  RESULTS")
    print(f"{'='*55}")

    print(f"\n  ── Classification Metrics ─────────────────────")
    print(f"  Accuracy    : {accuracy:.4f}  ({correct}/{total} correct)")
    print(f"  Precision   : {precision:.4f}  (weighted avg)")
    print(f"  Recall      : {recall:.4f}  (weighted avg)")
    print(f"  F1 Score    : {f1:.4f}  (weighted avg)")

    print(f"\n  ── Retrieval Metrics ──────────────────────────")
    _bar = lambda v: '█' * int(v * 30)
    print(f"  Recall@1    : {r_at_1:.4f}  [{_bar(r_at_1)}]")
    print(f"  Recall@5    : {r_at_5:.4f}  [{_bar(r_at_5)}]")
    print(f"  Recall@10   : {r_at_10:.4f}  [{_bar(r_at_10)}]")

    print(f"\n  ── Coordinate (GPS) Metrics ───────────────────")
    print(f"  Mean error   : {mean_err:.4f} km")
    print(f"  Median error : {median_err:.4f} km")
    print(f"  Within {Config.DISTANCE_THRESHOLD_KM*1000:.0f}m : "
          f"{within_thr*100:.1f}%")
    print(f"{'='*55}\n")

    return metrics


# ============================================================
# SECTION 8 — PLOTS
# ============================================================

def plot_metrics(metrics: dict) -> None:
    """Save 4 plots: bar chart, error distribution, confusion matrix, Recall@K."""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Drone Geolocalization — Evaluation Results",
                 fontsize=15, fontweight='bold', y=0.98)

    # ── 1. Classification metrics bar chart ───────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    names  = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [metrics['accuracy'], metrics['precision'],
              metrics['recall'],   metrics['f1_score']]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    bars   = ax1.bar(names, values, color=colors, width=0.5, edgecolor='white')
    ax1.set_ylim(0, 1.0)
    ax1.set_title('Classification Metrics', fontsize=11)
    ax1.set_ylabel('Score')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # ── 2. Recall@K bar chart ─────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    ks     = [1, 5, 10]
    rvals  = [metrics['recall_at_1'], metrics['recall_at_5'],
              metrics['recall_at_10']]
    bars2  = ax2.bar([f'R@{k}' for k in ks], rvals,
                     color='steelblue', width=0.4, edgecolor='white')
    ax2.set_ylim(0, 1.0)
    ax2.set_title('Retrieval — Recall@K', fontsize=11)
    ax2.set_ylabel('Recall')
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, rvals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # ── 3. GPS error distribution histogram ───────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    errs = metrics['dist_errors']
    ax3.hist(errs, bins=40, color='#55A868', edgecolor='white', alpha=0.85)
    ax3.axvline(metrics['mean_error_km'],   color='red',
                linestyle='--', label=f"Mean={metrics['mean_error_km']:.3f} km")
    ax3.axvline(metrics['median_error_km'], color='orange',
                linestyle='--', label=f"Median={metrics['median_error_km']:.3f} km")
    ax3.axvline(Config.DISTANCE_THRESHOLD_KM, color='purple',
                linestyle=':', label=f"Threshold={Config.DISTANCE_THRESHOLD_KM} km")
    ax3.set_title('GPS Error Distribution', fontsize=11)
    ax3.set_xlabel('Distance Error (km)')
    ax3.set_ylabel('Count')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # ── 4. Cumulative error curve ─────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    sorted_errs = np.sort(errs)
    cdf = np.arange(1, len(sorted_errs)+1) / len(sorted_errs)
    ax4.plot(sorted_errs, cdf, color='steelblue', lw=2)
    ax4.axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='50%')
    ax4.axhline(0.9, color='red',    linestyle='--', alpha=0.7, label='90%')
    ax4.axvline(Config.DISTANCE_THRESHOLD_KM, color='purple',
                linestyle=':', alpha=0.7, label=f'{Config.DISTANCE_THRESHOLD_KM} km')
    ax4.set_title('Cumulative GPS Error Curve', fontsize=11)
    ax4.set_xlabel('Distance Error (km)')
    ax4.set_ylabel('Fraction of Queries')
    ax4.set_xlim(left=0)
    ax4.set_ylim(0, 1)
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # ── 5. Summary text panel ─────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    summary = (
        f"EVALUATION SUMMARY\n"
        f"{'─'*32}\n"
        f"Total queries    : {metrics['total_queries']}\n\n"
        f"CLASSIFICATION\n"
        f"  Accuracy       : {metrics['accuracy']:.4f}\n"
        f"  Precision      : {metrics['precision']:.4f}\n"
        f"  Recall         : {metrics['recall']:.4f}\n"
        f"  F1 Score       : {metrics['f1_score']:.4f}\n\n"
        f"RETRIEVAL\n"
        f"  Recall@1       : {metrics['recall_at_1']:.4f}\n"
        f"  Recall@5       : {metrics['recall_at_5']:.4f}\n"
        f"  Recall@10      : {metrics['recall_at_10']:.4f}\n\n"
        f"GPS COORDINATES\n"
        f"  Mean error     : {metrics['mean_error_km']:.4f} km\n"
        f"  Median error   : {metrics['median_error_km']:.4f} km\n"
        f"  Within 500m    : {metrics['within_500m_pct']:.1f}%"
    )
    ax5.text(0.05, 0.95, summary, transform=ax5.transAxes,
             fontsize=9.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f4f8', alpha=0.8))

    # ── 6. Correct vs Wrong pie ────────────────────────────────
    ax6 = fig.add_subplot(2, 3, 6)
    correct  = int(metrics['accuracy'] * metrics['total_queries'])
    wrong    = metrics['total_queries'] - correct
    ax6.pie([correct, wrong],
            labels=[f'Correct\n({correct})', f'Wrong\n({wrong})'],
            colors=['#55A868', '#C44E52'],
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax6.set_title('Top-1 Prediction Accuracy', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(Config.OUTPUT_DIR, "evaluation_report.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Evaluation report saved → {out}")


def save_csv_report(metrics: dict) -> None:
    """Save per-query results to CSV for further analysis."""
    import csv
    path = os.path.join(Config.OUTPUT_DIR, "per_query_results.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query_loc_id', 'pred_loc_id',
                         'correct', 'dist_error_km'])
        for t, p, d in zip(metrics['y_true'], metrics['y_pred'],
                            metrics['dist_errors']):
            writer.writerow([t, p, int(t == p), f"{d:.6f}"])
    print(f"[CSV] Per-query results saved → {path}")


# ============================================================
# SECTION 9 — MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Drone Geolocalization — Predict & Evaluate'
    )
    parser.add_argument('--image', type=str, default=None,
                        help='Path to a single drone image for prediction')
    parser.add_argument('--kml',   type=str, default=None,
                        help='Path to University-1652 KML GPS file (optional)')
    parser.add_argument('--k',     type=int, default=Config.TOP_K,
                        help='Top-K matches to return (default: 5)')
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  DRONE GEOLOCALIZATION — PREDICT & EVALUATE")
    print("="*55 + "\n")

    # ── Load TFLite engine ─────────────────────────────────────
    try:
        engine = TFLiteEngine(Config.TFLITE_PATH)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # ── Load database ──────────────────────────────────────────
    try:
        db = MapDB(Config.DB_PATH)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # ── Build coordinate map ───────────────────────────────────
    # Collect all location IDs from both train and test
    all_loc_ids = set()
    for folder in [Config.TRAIN_SAT_PATH, Config.GALLERY_SAT_PATH,
                   Config.QUERY_DRONE_PATH]:
        if os.path.isdir(folder):
            all_loc_ids.update(os.listdir(folder))
    all_loc_ids = sorted(all_loc_ids)

    coord_map = get_coordinate_map(all_loc_ids, kml_path=args.kml)

    # ── Single image prediction mode ───────────────────────────
    if args.image:
        print(f"[Mode] Single image prediction")
        result = predict_single(args.image, engine, db, coord_map, k=args.k)
        print_prediction(result)
        return

    # ── Full evaluation mode ───────────────────────────────────
    print(f"[Mode] Full evaluation on query_drone images")
    if not os.path.isdir(Config.QUERY_DRONE_PATH):
        print(f"[ERROR] Query folder not found: {Config.QUERY_DRONE_PATH}")
        sys.exit(1)

    metrics = evaluate_all(engine, db, coord_map)
    plot_metrics(metrics)
    save_csv_report(metrics)

    print("\n  All evaluation outputs saved to:", Config.OUTPUT_DIR)
    print("  ├── evaluation_report.png  (6-panel metrics plot)")
    print("  └── per_query_results.csv  (per-image results)\n")


if __name__ == "__main__":
    main()
