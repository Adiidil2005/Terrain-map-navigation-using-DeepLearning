# ============================================================
# predict_and_evaluate.py
#
# 1. Parses real GPS coords from university1652-kml/first-key/*.kml
# 2. Predicts coordinates for drone images
# 3. Shows: drone query | predicted satellite | true satellite | rank 2-4
# 4. Measures Accuracy, Precision, Recall, F1 + GPS metrics
#
# Run:
#   python predict_and_evaluate.py                    ← full evaluation
#   python predict_and_evaluate.py --image path/to/drone.jpg
#   python predict_and_evaluate.py --image x.jpg --true_loc 0042
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
import matplotlib.patches as mpatches
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    DATASET_PATH     = r"C:\Users\adity\MiniProj\U1652\University-Release"
    QUERY_DRONE_PATH = os.path.join(DATASET_PATH, "test", "query_drone")
    GALLERY_SAT_PATH = os.path.join(DATASET_PATH, "test", "gallery_satellite")
    TRAIN_SAT_PATH   = os.path.join(DATASET_PATH, "train", "satellite")

    # KML folder structure:
    #   university1652-kml/
    #     _MACOSX/        ← ignored
    #     first-key/      ← 0001.kml, 0002.kml ... are here
    KML_DIR          = os.path.join("university1652-kml", "first-key")

    DB_PATH          = os.path.join("outputs_fast", "map_db.pkl")
    TFLITE_PATH      = os.path.join("outputs_fast", "drone_geo.tflite")
    OUTPUT_DIR       = "eval_outputs"

    # ── Must match what you trained with ──────────────────────
    IMAGE_SIZE       = (128, 128)    # change to (160,160) if you retrained
    INPUT_SHAPE      = (128, 128, 3) # change to (160,160,3) if you retrained
    EMBEDDING_DIM    = 256           # change to 512 if you retrained

    TOP_K            = 5
    MAX_EVAL_IMAGES  = 300           # set to None to evaluate all
    DISTANCE_THRESHOLD_KM = 0.5     # 500 m = "correct" GPS prediction


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# ============================================================
# SECTION 1 — KML PARSING
# Each file: 0001.kml contains coords for location 0001
# Format: <coordinates>lon,lat,alt</coordinates>
# ============================================================

def parse_single_kml(kml_path: str):
    """
    Parse one KML file. Returns (lat, lon) or None.
    Coordinates in KML are lon,lat,alt — we swap to return (lat, lon).
    """
    try:
        with open(kml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        match = re.search(
            r'<coordinates>\s*([\-\d.]+)\s*,\s*([\-\d.]+)', content
        )
        if match:
            lon = float(match.group(1))
            lat = float(match.group(2))
            return (lat, lon)
    except Exception:
        pass
    return None


def load_coords_from_kml_folder(kml_dir: str) -> dict:
    """
    Scan kml_dir for XXXX.kml files.
    Returns { '0001': (lat, lon), '0002': (lat, lon), ... }
    Ignores _MACOSX and non-kml files automatically.
    """
    coords = {}

    if not os.path.isdir(kml_dir):
        print(f"[KML] Folder not found: {kml_dir}")
        return {}

    kml_files = [f for f in os.listdir(kml_dir) if f.endswith('.kml')]
    if not kml_files:
        print(f"[KML] No .kml files found in: {kml_dir}")
        return {}

    failed = 0
    for fname in sorted(kml_files):
        loc_id = os.path.splitext(fname)[0].zfill(4)   # '1' → '0001'
        result = parse_single_kml(os.path.join(kml_dir, fname))
        if result:
            coords[loc_id] = result
        else:
            failed += 1

    print(f"[KML] Loaded real GPS for {len(coords)} locations "
          f"({failed} parse failures)  ←  {kml_dir}")
    return coords


def get_coordinate_map(location_ids: list) -> dict:
    """
    Returns coord map for all location IDs.
    Uses real KML coords where available, synthetic for the rest.
    """
    coords = load_coords_from_kml_folder(Config.KML_DIR)

    # Fill any missing IDs with reproducible synthetic coords
    missing = [lid for lid in location_ids if lid not in coords]
    if missing:
        print(f"[KML] Generating synthetic coords for {len(missing)} "
              f"locations not found in KML.")
        base_lat, base_lon = 40.0, 116.0
        for loc_id in missing:
            seed = int(loc_id) if loc_id.isdigit() else hash(loc_id) % 10000
            rng  = np.random.default_rng(seed)
            coords[loc_id] = (
                round(base_lat + rng.uniform(-0.05, 0.05), 6),
                round(base_lon + rng.uniform(-0.05, 0.05), 6)
            )

    print(f"[KML] Coordinate map ready: {len(coords)} locations.\n")
    return coords


# ============================================================
# SECTION 2 — IMAGE UTILITIES
# ============================================================

def preprocess(img_path: str) -> np.ndarray:
    """Preprocess image for model inference."""
    img = tf.keras.preprocessing.image.load_img(
        img_path, target_size=Config.IMAGE_SIZE
    )
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)
    return arr.astype(np.float32)


def load_display_img(img_path: str, size=(256, 256)) -> np.ndarray:
    """Load image as RGB uint8 array for matplotlib display."""
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    return np.array(img)


def get_imgs(base_path: str, loc_id: str) -> list:
    """Return sorted list of image paths for a location folder."""
    folder = os.path.join(base_path, loc_id)
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]


def get_sat_img(loc_id: str) -> str | None:
    """Get a satellite image path for loc_id (gallery first, then train)."""
    for folder in [Config.GALLERY_SAT_PATH, Config.TRAIN_SAT_PATH]:
        imgs = get_imgs(folder, loc_id)
        if imgs:
            return imgs[0]
    return None


# ============================================================
# SECTION 3 — TFLite INFERENCE ENGINE
# ============================================================

class TFLiteEngine:
    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"TFLite model not found: {model_path}\n"
                "  → Run drone_geo_fast.py first."
            )
        self.interp = tf.lite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self._in  = self.interp.get_input_details()
        self._out = self.interp.get_output_details()
        print(f"[TFLite] Loaded  : {model_path}")

    def embed(self, img_path: str) -> np.ndarray:
        arr = preprocess(img_path)
        inp = np.expand_dims(arr, 0).astype(np.float32)
        self.interp.set_tensor(self._in[0]['index'], inp)
        self.interp.invoke()
        return self.interp.get_tensor(self._out[0]['index'])[0]


# ============================================================
# SECTION 4 — MAP VECTOR DATABASE
# ============================================================

class MapDB:
    def __init__(self, db_path: str):
        if not os.path.isfile(db_path):
            raise FileNotFoundError(
                f"Database not found: {db_path}\n"
                "  → Run drone_geo_fast.py first."
            )
        with open(db_path, 'rb') as f:
            data = pickle.load(f)
        self.vectors = data['vectors'].astype(np.float32)
        self.ids     = data['ids']

        k = min(10, len(self.vectors))
        self._nn = NearestNeighbors(
            n_neighbors=k, metric='cosine',
            algorithm='brute', n_jobs=-1
        )
        self._nn.fit(self.vectors)
        print(f"[DB]     Loaded  : {len(self.vectors)} reference vectors.")

    def search(self, query_vec: np.ndarray, k: int = 5):
        dists, idx = self._nn.kneighbors(
            query_vec.reshape(1, -1),
            n_neighbors=min(k, len(self.vectors))
        )
        return dists[0], [self.ids[i] for i in idx[0]]


# ============================================================
# SECTION 5 — HAVERSINE DISTANCE
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R    = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat/2)**2 +
            math.cos(math.radians(lat1)) *
            math.cos(math.radians(lat2)) *
            math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


# ============================================================
# SECTION 6 — SINGLE IMAGE PREDICTION + VISUALIZATION
#
# Layout (1 row × 6 columns):
# ┌──────────────┬──────────────┬──────────────┬──────┬──────┬──────┐
# │  QUERY       │  PREDICTED   │  TRUE        │ Rank │ Rank │ Rank │
# │  (Drone)     │  Satellite   │  Satellite   │  2   │  3   │  4   │
# │  blue border │ green/red    │ gold border  │      │      │      │
# └──────────────┴──────────────┴──────────────┴──────┴──────┴──────┘
# ============================================================

_BLANK = np.full((256, 256, 3), 40, dtype=np.uint8)   # dark placeholder


def _styled_ax(ax, img, title, subtitle='', border_color=None, title_bg=None):
    """Helper: display image on axis with styled border and title."""
    ax.imshow(img)
    ax.set_facecolor('#1a1a2e')
    for sp in ax.spines.values():
        if border_color:
            sp.set_edgecolor(border_color)
            sp.set_linewidth(5)
        else:
            sp.set_visible(False)
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    ax.set_title(title, fontsize=8.5, color='white',
                 fontweight='bold', pad=4,
                 backgroundcolor=title_bg or 'none')
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=7, color='#cccccc', labelpad=4)


def predict_and_visualize(img_path: str,
                           engine: TFLiteEngine,
                           db: MapDB,
                           coord_map: dict,
                           true_loc_id: str = None,
                           k: int = 5,
                           save_path: str = None) -> dict:
    """
    Predict location for a drone image and save a visualization.

    Returns result dict with predicted lat/lon and top-K matches.
    """
    # ── Run model ──────────────────────────────────────────────
    emb             = engine.embed(img_path)
    dists, loc_ids  = db.search(emb, k=k)

    top_loc          = loc_ids[0]
    top_lat, top_lon = coord_map.get(top_loc, (0.0, 0.0))

    is_correct = (true_loc_id == top_loc) if true_loc_id else None

    result = {
        'image_path' : img_path,
        'true_loc'   : true_loc_id,
        'pred_loc'   : top_loc,
        'latitude'   : top_lat,
        'longitude'  : top_lon,
        'correct'    : is_correct,
        'top_k'      : [
            {
                'rank'      : i + 1,
                'loc_id'    : lid,
                'lat'       : coord_map.get(lid, (0.0, 0.0))[0],
                'lon'       : coord_map.get(lid, (0.0, 0.0))[1],
                'cos_dist'  : float(d),
                'confidence': round((1 - float(d)) * 100, 2),
            }
            for i, (lid, d) in enumerate(zip(loc_ids, dists))
        ],
    }

    # ── Figure ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 6, figsize=(20, 4.2))
    fig.patch.set_facecolor('#1a1a2e')

    # --- Col 0 : Query drone image ---
    drone_img = load_display_img(img_path)
    _styled_ax(
        axes[0], drone_img,
        title     = "QUERY  (Drone)",
        subtitle  = f"True loc: {true_loc_id or 'unknown'}",
        border_color = '#4fc3f7',
        title_bg  = '#0d47a1'
    )

    # --- Col 1 : PREDICTED satellite (shown FIRST as requested) ---
    pred_img_path = get_sat_img(top_loc)
    pred_img      = load_display_img(pred_img_path) if pred_img_path else _BLANK
    pred_color    = '#66bb6a' if is_correct else '#ef5350'
    mark          = '✓ CORRECT' if is_correct else '✗ WRONG'
    conf          = result['top_k'][0]['confidence']

    _styled_ax(
        axes[1], pred_img,
        title    = f"PREDICTED  {mark if is_correct is not None else ''}",
        subtitle = (f"Loc: {top_loc}   Conf: {conf:.1f}%\n"
                    f"Lat: {top_lat:.5f}°   Lon: {top_lon:.5f}°"),
        border_color = pred_color,
        title_bg     = '#1b5e20' if is_correct else '#b71c1c'
    )

    # --- Col 2 : TRUE satellite (ground truth, shown SECOND) ---
    if true_loc_id:
        true_img_path    = get_sat_img(true_loc_id)
        true_img         = load_display_img(true_img_path) \
                           if true_img_path else _BLANK
        true_lat, true_lon = coord_map.get(true_loc_id, (0.0, 0.0))
        gps_err            = haversine_km(true_lat, true_lon, top_lat, top_lon)
        _styled_ax(
            axes[2], true_img,
            title    = "TRUE LOCATION",
            subtitle = (f"Loc: {true_loc_id}\n"
                        f"Lat: {true_lat:.5f}°   Lon: {true_lon:.5f}°\n"
                        f"GPS error: {gps_err:.3f} km"),
            border_color = '#ffd54f',
            title_bg     = '#e65100'
        )
    else:
        axes[2].set_facecolor('#1a1a2e')
        axes[2].text(0.5, 0.5, 'True location\nnot provided',
                     ha='center', va='center', color='#888',
                     transform=axes[2].transAxes, fontsize=9)
        axes[2].axis('off')

    # --- Cols 3-5 : Rank 2, 3, 4 candidates ---
    for col_i, rank_i in enumerate(range(1, 4), start=3):
        if rank_i >= len(result['top_k']):
            axes[col_i].axis('off')
            continue
        m          = result['top_k'][rank_i]
        r_img_path = get_sat_img(m['loc_id'])
        r_img      = load_display_img(r_img_path) if r_img_path else _BLANK
        r_correct  = (true_loc_id == m['loc_id']) if true_loc_id else None
        _styled_ax(
            axes[col_i], r_img,
            title        = f"Rank {m['rank']}  {'✓' if r_correct else ''}",
            subtitle     = f"Loc: {m['loc_id']}   Conf: {m['confidence']:.1f}%",
            border_color = '#66bb6a' if r_correct else None
        )

    # Legend
    legend = [
        mpatches.Patch(color='#4fc3f7', label='Query (Drone)'),
        mpatches.Patch(color='#66bb6a', label='Correct match'),
        mpatches.Patch(color='#ef5350', label='Wrong match'),
        mpatches.Patch(color='#ffd54f', label='Ground truth'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=4,
               fontsize=8, facecolor='#1a1a2e', labelcolor='white',
               framealpha=0.6, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out = save_path or os.path.join(
        Config.OUTPUT_DIR,
        f"pred_loc{true_loc_id or 'unknown'}.png"
    )
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    return result


def print_prediction(result: dict) -> None:
    is_c = result.get('correct')
    mark = ('✓ CORRECT' if is_c else '✗ WRONG') if is_c is not None else ''
    print(f"\n{'─'*60}")
    print(f"  Image      : {os.path.basename(result['image_path'])}")
    print(f"  True loc   : {result['true_loc'] or 'unknown'}")
    print(f"  Pred loc   : {result['pred_loc']}  {mark}")
    print(f"  Latitude   : {result['latitude']:.6f}°")
    print(f"  Longitude  : {result['longitude']:.6f}°")
    print(f"\n  {'Rk':<4} {'Loc':<8} {'Lat':>11} {'Lon':>12} {'Conf':>8}")
    print(f"  {'─'*50}")
    for m in result['top_k']:
        print(f"  {m['rank']:<4} {m['loc_id']:<8} "
              f"{m['lat']:>11.5f} {m['lon']:>12.5f} "
              f"{m['confidence']:>7.1f}%  "
              f"{'█' * int(m['confidence']/5)}")
    print(f"{'─'*60}\n")


# ============================================================
# SECTION 7 — FULL EVALUATION
# ============================================================

def evaluate_all(engine: TFLiteEngine, db: MapDB,
                 coord_map: dict) -> dict:
    print(f"\n{'='*60}")
    print("  FULL EVALUATION — Accuracy · Precision · Recall · F1")
    print(f"{'='*60}")

    query_ids   = sorted(os.listdir(Config.QUERY_DRONE_PATH))
    y_true, y_pred, y_topk, dist_errors = [], [], [], []

    count = 0
    for loc_id in query_ids:
        imgs = get_imgs(Config.QUERY_DRONE_PATH, loc_id)
        if not imgs:
            continue
        emb = engine.embed(imgs[0])
        dists, pred_ids = db.search(emb, k=10)

        y_true.append(loc_id)
        y_pred.append(pred_ids[0])
        y_topk.append(pred_ids)

        t_lat, t_lon = coord_map.get(loc_id,      (0.0, 0.0))
        p_lat, p_lon = coord_map.get(pred_ids[0], (0.0, 0.0))
        dist_errors.append(haversine_km(t_lat, t_lon, p_lat, p_lon))

        count += 1
        if count % 50 == 0:
            print(f"  Processed {count} queries...")
        if Config.MAX_EVAL_IMAGES and count >= Config.MAX_EVAL_IMAGES:
            break

    total   = len(y_true)
    classes = sorted(set(y_true))
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    accuracy  = correct / total if total > 0 else 0.0
    precision = precision_score(y_true, y_pred, average='weighted',
                                zero_division=0, labels=classes)
    recall    = recall_score(   y_true, y_pred, average='weighted',
                                zero_division=0, labels=classes)
    f1        = f1_score(       y_true, y_pred, average='weighted',
                                zero_division=0, labels=classes)

    r1  = sum(1 for t,p in zip(y_true,y_topk) if t in p[:1])  / total
    r5  = sum(1 for t,p in zip(y_true,y_topk) if t in p[:5])  / total
    r10 = sum(1 for t,p in zip(y_true,y_topk) if t in p[:10]) / total

    mean_err   = float(np.mean(dist_errors))
    median_err = float(np.median(dist_errors))
    within_thr = sum(1 for d in dist_errors
                     if d <= Config.DISTANCE_THRESHOLD_KM) / total

    metrics = dict(
        total=total, correct=correct,
        accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        r1=r1, r5=r5, r10=r10,
        mean_err=mean_err, median_err=median_err,
        within_thr_pct=within_thr * 100,
        dist_errors=dist_errors, y_true=y_true, y_pred=y_pred,
    )

    _b = lambda v: '█' * int(v * 30)
    print(f"\n  Queries : {total}")
    print(f"\n  ── Classification ──────────────────────────────────")
    print(f"  Accuracy   : {accuracy:.4f}  ({correct}/{total})")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"\n  ── Retrieval ───────────────────────────────────────")
    print(f"  Recall@1   : {r1:.4f}  [{_b(r1)}]")
    print(f"  Recall@5   : {r5:.4f}  [{_b(r5)}]")
    print(f"  Recall@10  : {r10:.4f}  [{_b(r10)}]")
    print(f"\n  ── GPS Coordinates ─────────────────────────────────")
    print(f"  Mean error   : {mean_err:.4f} km")
    print(f"  Median error : {median_err:.4f} km")
    print(f"  Within 500m  : {within_thr*100:.1f}%")
    print(f"{'='*60}\n")
    return metrics


# ============================================================
# SECTION 8 — PLOTS & CSV
# ============================================================

def plot_metrics(metrics: dict) -> None:
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Drone Geolocalization — Evaluation Report",
                 fontsize=14, fontweight='bold')

    ax1 = fig.add_subplot(2, 3, 1)
    names  = ['Accuracy', 'Precision', 'Recall', 'F1']
    vals   = [metrics['accuracy'], metrics['precision'],
              metrics['recall'],   metrics['f1']]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    bars   = ax1.bar(names, vals, color=colors, edgecolor='white', width=0.5)
    ax1.set_ylim(0, 1.15); ax1.set_title('Classification Metrics', fontsize=11)
    ax1.set_ylabel('Score'); ax1.grid(axis='y', alpha=0.3)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x()+b.get_width()/2, v+0.03,
                 f'{v:.3f}', ha='center', fontsize=10)

    ax2 = fig.add_subplot(2, 3, 2)
    rvals = [metrics['r1'], metrics['r5'], metrics['r10']]
    bars2 = ax2.bar(['R@1', 'R@5', 'R@10'], rvals,
                    color='steelblue', edgecolor='white', width=0.4)
    ax2.set_ylim(0, 1.15); ax2.set_title('Retrieval — Recall@K', fontsize=11)
    ax2.set_ylabel('Recall'); ax2.grid(axis='y', alpha=0.3)
    for b, v in zip(bars2, rvals):
        ax2.text(b.get_x()+b.get_width()/2, v+0.03,
                 f'{v:.3f}', ha='center', fontsize=10)

    ax3 = fig.add_subplot(2, 3, 3)
    errs = metrics['dist_errors']
    ax3.hist(errs, bins=40, color='#55A868', edgecolor='white', alpha=0.85)
    ax3.axvline(metrics['mean_err'],   color='red',    linestyle='--',
                label=f"Mean={metrics['mean_err']:.3f} km")
    ax3.axvline(metrics['median_err'], color='orange', linestyle='--',
                label=f"Median={metrics['median_err']:.3f} km")
    ax3.axvline(Config.DISTANCE_THRESHOLD_KM, color='purple', linestyle=':',
                label=f"{Config.DISTANCE_THRESHOLD_KM} km threshold")
    ax3.set_title('GPS Error Distribution', fontsize=11)
    ax3.set_xlabel('Distance Error (km)'); ax3.set_ylabel('Count')
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(2, 3, 4)
    se  = np.sort(errs)
    cdf = np.arange(1, len(se)+1) / len(se)
    ax4.plot(se, cdf, color='steelblue', lw=2)
    ax4.axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='50%')
    ax4.axhline(0.9, color='red',    linestyle='--', alpha=0.7, label='90%')
    ax4.axvline(Config.DISTANCE_THRESHOLD_KM, color='purple',
                linestyle=':', alpha=0.7)
    ax4.set_title('Cumulative GPS Error', fontsize=11)
    ax4.set_xlabel('Distance Error (km)'); ax4.set_ylabel('Fraction of Queries')
    ax4.set_xlim(left=0); ax4.set_ylim(0, 1)
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    txt = (
        f"EVALUATION SUMMARY\n{'─'*34}\n"
        f"Total queries    : {metrics['total']}\n\n"
        f"CLASSIFICATION\n"
        f"  Accuracy       : {metrics['accuracy']:.4f}\n"
        f"  Precision      : {metrics['precision']:.4f}\n"
        f"  Recall         : {metrics['recall']:.4f}\n"
        f"  F1 Score       : {metrics['f1']:.4f}\n\n"
        f"RETRIEVAL\n"
        f"  Recall@1       : {metrics['r1']:.4f}\n"
        f"  Recall@5       : {metrics['r5']:.4f}\n"
        f"  Recall@10      : {metrics['r10']:.4f}\n\n"
        f"GPS COORDINATES (real KML)\n"
        f"  Mean error     : {metrics['mean_err']:.4f} km\n"
        f"  Median error   : {metrics['median_err']:.4f} km\n"
        f"  Within 500m    : {metrics['within_thr_pct']:.1f}%"
    )
    ax5.text(0.05, 0.95, txt, transform=ax5.transAxes, fontsize=9.5,
             va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f4f8', alpha=0.9))

    ax6 = fig.add_subplot(2, 3, 6)
    wrong = metrics['total'] - metrics['correct']
    ax6.pie(
        [metrics['correct'], wrong],
        labels=[f"Correct\n({metrics['correct']})",
                f"Wrong\n({wrong})"],
        colors=['#55A868', '#C44E52'], autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    ax6.set_title('Top-1 Accuracy', fontsize=11)

    plt.tight_layout()
    out = os.path.join(Config.OUTPUT_DIR, "evaluation_report.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Evaluation report   → {out}")


def save_csv(metrics: dict) -> None:
    import csv
    path = os.path.join(Config.OUTPUT_DIR, "per_query_results.csv")
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['query_loc_id', 'pred_loc_id', 'correct', 'dist_error_km'])
        for t, p, d in zip(metrics['y_true'], metrics['y_pred'],
                           metrics['dist_errors']):
            w.writerow([t, p, int(t == p), f"{d:.6f}"])
    print(f"[CSV] Per-query results    → {path}")


def save_sample_predictions(engine, db, coord_map, n=6):
    """Save n sample prediction visualizations from random query locations."""
    print(f"\n[Viz] Saving {n} sample prediction images...")
    query_ids = sorted(os.listdir(Config.QUERY_DRONE_PATH))
    samples   = random.sample(query_ids, min(n, len(query_ids)))

    for i, loc_id in enumerate(samples):
        imgs = get_imgs(Config.QUERY_DRONE_PATH, loc_id)
        if not imgs:
            continue
        out    = os.path.join(Config.OUTPUT_DIR,
                              f"sample_{i+1:02d}_loc{loc_id}.png")
        result = predict_and_visualize(
            img_path    = imgs[0],
            engine      = engine,
            db          = db,
            coord_map   = coord_map,
            true_loc_id = loc_id,
            k           = Config.TOP_K,
            save_path   = out,
        )
        print_prediction(result)
        print(f"  Saved → {out}")


# ============================================================
# SECTION 9 — MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Drone Geolocalization — Predict & Evaluate'
    )
    parser.add_argument('--image',    type=str, default=None,
                        help='Path to a single drone image')
    parser.add_argument('--true_loc', type=str, default=None,
                        help='True location ID for --image (e.g. 0042)')
    parser.add_argument('--samples',  type=int, default=6,
                        help='Number of sample prediction images (default: 6)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  DRONE GEOLOCALIZATION — PREDICT & EVALUATE")
    print("="*60 + "\n")

    # Load model + DB
    try:
        engine = TFLiteEngine(Config.TFLITE_PATH)
        db     = MapDB(Config.DB_PATH)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}"); sys.exit(1)

    # Build coord map from KML files
    all_loc_ids = set()
    for folder in [Config.TRAIN_SAT_PATH,
                   Config.GALLERY_SAT_PATH,
                   Config.QUERY_DRONE_PATH]:
        if os.path.isdir(folder):
            all_loc_ids.update(
                f for f in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, f))
            )
    coord_map = get_coordinate_map(sorted(all_loc_ids))

    # ── Single image mode ──────────────────────────────────────
    if args.image:
        out = os.path.join(Config.OUTPUT_DIR, "single_prediction.png")
        result = predict_and_visualize(
            img_path    = args.image,
            engine      = engine,
            db          = db,
            coord_map   = coord_map,
            true_loc_id = args.true_loc,
            k           = Config.TOP_K,
            save_path   = out,
        )
        print_prediction(result)
        print(f"  Visualization saved → {out}")
        return

    # ── Full evaluation mode ───────────────────────────────────
    metrics = evaluate_all(engine, db, coord_map)
    plot_metrics(metrics)
    save_csv(metrics)
    save_sample_predictions(engine, db, coord_map, n=args.samples)

    print(f"\n  All outputs → {Config.OUTPUT_DIR}/")
    print(f"  ├── evaluation_report.png")
    print(f"  ├── per_query_results.csv")
    print(f"  └── sample_XX_locYYYY.png  ({args.samples} prediction grids)\n")


if __name__ == "__main__":
    main()