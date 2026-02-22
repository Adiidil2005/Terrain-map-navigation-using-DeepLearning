# ============================================================
# drone_geo_fast.py  -  Single-file Visual Geolocalization
# Target: Complete training + evaluation in < 1 hour
# Hardware: AMD Ryzen 5 5500U (CPU only, ~16 GB RAM assumed)
#
# Run:   python drone_geo_fast.py
# ============================================================

import os, sys, csv, time, random, pickle, warnings
warnings.filterwarnings('ignore')

# ── Must set BEFORE importing TensorFlow ────────────────────
os.environ['CUDA_VISIBLE_DEVICES']  = '-1'   # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'    # Suppress TF logs

import numpy as np
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)   # Avoids contention
tf.config.threading.set_intra_op_parallelism_threads(6)   # Ryzen 5 5500U: 6 cores

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print(f"TensorFlow : {tf.__version__}")
print(f"Python     : {sys.version.split()[0]}")
print(f"Device     : CPU — AMD Ryzen 5 5500U")


# ============================================================
# SECTION 1 — CONFIGURATION
# Speed-tuned for < 1 hour total on Ryzen 5 5500U
# ============================================================

class Config:
    # ── Dataset path  ← CHANGE THIS ─────────────────────────
    DATASET_PATH     = r"C:\Users\adity\MiniProj\U1652\University-Release"

    TRAIN_DRONE_PATH = os.path.join(DATASET_PATH, "train", "drone")
    TRAIN_SAT_PATH   = os.path.join(DATASET_PATH, "train", "satellite")
    QUERY_DRONE_PATH = os.path.join(DATASET_PATH, "test",  "query_drone")
    GALLERY_SAT_PATH = os.path.join(DATASET_PATH, "test",  "gallery_satellite")

    OUTPUT_DIR       = "outputs_fast"
    CHECKPOINT_PATH  = os.path.join(OUTPUT_DIR, "best_model.weights.h5")
    TFLITE_PATH      = os.path.join(OUTPUT_DIR, "drone_geo.tflite")
    DB_PATH          = os.path.join(OUTPUT_DIR, "map_db.pkl")

    # ── Model ───────────────────────────────────────────────
    IMAGE_SIZE       = (128, 128)     # ← 128 vs 224: ~3× faster per batch
    INPUT_SHAPE      = (128, 128, 3)
    EMBEDDING_DIM    = 256            # ← 256 vs 512: faster head + smaller DB

    # ── Training  (tuned for ~50 min total) ─────────────────
    EPOCHS           = 8             # Enough for a good prototype
    BATCH_SIZE       = 16            # Sweet spot for CPU RAM vs speed
    LEARNING_RATE    = 2e-4          # Slightly higher LR for fewer epochs
    MARGIN           = 0.3

    # ── Data sampling (controls how much data is used) ──────
    MAX_LOCATIONS    = 400           # Use 400/1652 locations (covers variety)
    TRIPLETS_PER_LOC = 4             # 4 × 400 = 1600 triplets per epoch
    NUM_WORKERS      = 4             # Parallel image loading threads
    PREFETCH         = 2

    # ── Backbone ─────────────────────────────────────────────
    UNFREEZE_LAST_N  = 10            # Fine-tune only last 10 layers

    # ── Retrieval ────────────────────────────────────────────
    TOP_K            = 5
    MAX_QUERY_IMGS   = 200           # Limit test queries for fast eval

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# ============================================================
# SECTION 2 — DATA UTILITIES
# ============================================================

def get_location_ids():
    drone_ids = set(os.listdir(Config.TRAIN_DRONE_PATH))
    sat_ids   = set(os.listdir(Config.TRAIN_SAT_PATH))
    common    = sorted(drone_ids & sat_ids)
    # Sample a subset for speed
    if len(common) > Config.MAX_LOCATIONS:
        common = random.sample(common, Config.MAX_LOCATIONS)
    print(f"[Data] Using {len(common)} locations (from "
          f"{len(drone_ids & sat_ids)} available)")
    return common


def get_imgs(base_path, loc_id):
    folder = os.path.join(base_path, loc_id)
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]


def load_img_fast(path):
    """
    Load + preprocess image. Uses 128×128 to cut load time ~3×
    compared to 224×224 while preserving enough spatial detail.
    """
    img = tf.keras.preprocessing.image.load_img(path, target_size=Config.IMAGE_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)
    return arr.astype(np.float32)


def augment(img):
    """Lightweight augmentation — only ops that don't cost much on CPU."""
    if random.random() > 0.5:
        img = np.fliplr(img)                      # Horizontal flip
    img = np.rot90(img, random.randint(0, 3))     # 90° rotation (yaw)
    return img.astype(np.float32)


def build_triplets(location_ids):
    triplets = []
    loc_list = list(location_ids)
    for loc_id in loc_list:
        d_imgs = get_imgs(Config.TRAIN_DRONE_PATH, loc_id)
        s_imgs = get_imgs(Config.TRAIN_SAT_PATH,   loc_id)
        if not d_imgs or not s_imgs:
            continue
        for _ in range(Config.TRIPLETS_PER_LOC):
            neg_id   = random.choice([x for x in loc_list if x != loc_id])
            neg_imgs = get_imgs(Config.TRAIN_SAT_PATH, neg_id)
            if not neg_imgs:
                continue
            triplets.append((
                random.choice(d_imgs),
                random.choice(s_imgs),
                random.choice(neg_imgs)
            ))
    random.shuffle(triplets)
    print(f"[Data] {len(triplets)} triplets built.")
    return triplets


def make_tf_dataset(triplets, augment_anchors=True):
    """
    tf.data pipeline with parallel loading + prefetch.
    Keeps CPU saturated between training steps.
    """
    a_paths = [t[0] for t in triplets]
    p_paths = [t[1] for t in triplets]
    n_paths = [t[2] for t in triplets]

    ds = tf.data.Dataset.from_tensor_slices((a_paths, p_paths, n_paths))
    ds = ds.shuffle(min(len(triplets), 1000), reshuffle_each_iteration=True)

    def _load(a, p, n):
        def _py(a, p, n):
            ai = load_img_fast(a.numpy().decode())
            pi = load_img_fast(p.numpy().decode())
            ni = load_img_fast(n.numpy().decode())
            if augment_anchors:
                ai = augment(ai)
            return ai, pi, ni
        ai, pi, ni = tf.py_function(_py, [a, p, n],
                                    [tf.float32, tf.float32, tf.float32])
        ai.set_shape(Config.INPUT_SHAPE)
        pi.set_shape(Config.INPUT_SHAPE)
        ni.set_shape(Config.INPUT_SHAPE)
        return (ai, pi, ni), tf.constant(0.0)

    ds = ds.map(_load, num_parallel_calls=Config.NUM_WORKERS)
    ds = ds.batch(Config.BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(Config.PREFETCH)
    return ds


# ============================================================
# SECTION 3 — MODEL
# ============================================================

def build_model():
    """
    SharedCNNBackbone: MobileNetV3Small at 128×128
    EmbeddingHead:     Dense(512, ReLU) → Dropout → Dense(256) → L2Norm

    Using 128×128 input + 256-dim embedding instead of 224×224 + 512-dim
    cuts compute by ~4× with minimal retrieval accuracy drop for a prototype.
    """
    # Backbone
    base = MobileNetV3Small(
        input_shape=Config.INPUT_SHAPE,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-Config.UNFREEZE_LAST_N:]:
        layer.trainable = True

    feat_dim = base.output_shape[-1]
    trainable = sum(1 for l in base.layers if l.trainable)
    print(f"[Model] Backbone output dim : {feat_dim}")
    print(f"[Model] Trainable layers    : {trainable}/{len(base.layers)}")

    # Embedding head
    inp = layers.Input(shape=(feat_dim,))
    x   = layers.Dense(512, activation='relu')(inp)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(Config.EMBEDDING_DIM, activation=None)(x)
    x   = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1),
                        name='l2_norm')(x)
    head = Model(inp, x, name='EmbeddingHead')

    # Siamese graph
    a_in = layers.Input(Config.INPUT_SHAPE, name='anchor')
    p_in = layers.Input(Config.INPUT_SHAPE, name='positive')
    n_in = layers.Input(Config.INPUT_SHAPE, name='negative')

    a_emb = head(base(a_in, training=True))
    p_emb = head(base(p_in, training=True))
    n_emb = head(base(n_in, training=True))

    out = layers.Concatenate()([a_emb, p_emb, n_emb])
    siamese   = Model([a_in, p_in, n_in], out, name='Siamese')
    inference = Model(a_in, a_emb, name='Inference')

    total_params = siamese.count_params()
    print(f"[Model] Total parameters    : {total_params:,}")
    return siamese, inference


# ============================================================
# SECTION 4 — TRIPLET LOSS
# ============================================================

class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, margin=0.3, **kw):
        super().__init__(**kw)
        self.margin = margin

    def call(self, y_true, y_pred):
        d   = Config.EMBEDDING_DIM
        a   = y_pred[:, :d]
        p   = y_pred[:, d:2*d]
        n   = y_pred[:, 2*d:]
        dp  = tf.reduce_sum(tf.square(a - p), axis=1)
        dn  = tf.reduce_sum(tf.square(a - n), axis=1)
        return tf.reduce_mean(tf.maximum(dp - dn + self.margin, 0.0))


# ============================================================
# SECTION 5 — TRAINING LOOP
# ============================================================

def train(siamese, inference):
    optimizer = Adam(Config.LEARNING_RATE, epsilon=1e-7)
    loss_fn   = TripletLoss(Config.MARGIN)

    @tf.function
    def train_step(a, p, n):
        with tf.GradientTape() as tape:
            pred  = siamese([a, p, n], training=True)
            dummy = tf.zeros(tf.shape(pred)[0])
            loss  = loss_fn(dummy, pred)
        grads = tape.gradient(loss, siamese.trainable_variables)
        optimizer.apply_gradients(zip(grads, siamese.trainable_variables))
        return loss

    location_ids = get_location_ids()
    triplets     = build_triplets(location_ids)
    dataset      = make_tf_dataset(triplets)

    all_losses = []
    best_loss  = float('inf')

    log_path = os.path.join(Config.OUTPUT_DIR, "training_log.csv")
    logf     = open(log_path, 'w', newline='')
    logger   = csv.writer(logf)
    logger.writerow(['epoch', 'loss', 'time_s', 'steps'])

    print(f"\n{'='*55}")
    print(f"  TRAINING  —  {Config.EPOCHS} epochs")
    print(f"  Image size : {Config.IMAGE_SIZE}  "
          f"Embedding : {Config.EMBEDDING_DIM}d")
    print(f"  Batch size : {Config.BATCH_SIZE}  "
          f"Locations : {Config.MAX_LOCATIONS}")
    print(f"{'='*55}\n")

    total_start = time.time()

    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start  = time.time()
        step_losses  = []
        step         = 0

        for (a_batch, p_batch, n_batch), _ in dataset:
            loss = train_step(a_batch, p_batch, n_batch)
            step_losses.append(float(loss.numpy()))
            step += 1

            if step % 20 == 0:
                elapsed = time.time() - epoch_start
                print(f"  Ep {epoch}/{Config.EPOCHS}  "
                      f"Step {step:3d}  "
                      f"Loss {np.mean(step_losses[-20:]):.4f}  "
                      f"[{elapsed:.0f}s]")

        epoch_loss = float(np.mean(step_losses)) if step_losses else 0.0
        epoch_time = time.time() - epoch_start
        all_losses.append(epoch_loss)

        logger.writerow([epoch, f"{epoch_loss:.6f}",
                         f"{epoch_time:.1f}", step])
        logf.flush()

        elapsed_total = time.time() - total_start
        eta_per_epoch = epoch_time
        remaining     = (Config.EPOCHS - epoch) * eta_per_epoch

        print(f"\n  ► Epoch {epoch}/{Config.EPOCHS} complete")
        print(f"    Loss     : {epoch_loss:.4f}")
        print(f"    Time     : {epoch_time:.0f}s this epoch")
        print(f"    Elapsed  : {elapsed_total/60:.1f} min total")
        print(f"    ETA      : ~{remaining/60:.0f} min remaining\n")

        # Save best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            siamese.save_weights(Config.CHECKPOINT_PATH)
            print(f"    ✓ Best model saved (loss={best_loss:.4f})\n")

        # Rebuild triplets halfway through to refresh negatives
        if epoch == Config.EPOCHS // 2:
            print("  [Data] Refreshing triplets with new negatives...\n")
            triplets = build_triplets(location_ids)
            dataset  = make_tf_dataset(triplets)

    logf.close()

    total_time = time.time() - total_start
    print(f"{'='*55}")
    print(f"  Training finished in {total_time/60:.1f} minutes")
    print(f"  Best loss : {best_loss:.4f}")
    print(f"{'='*55}\n")

    # Load best weights before returning
    siamese.load_weights(Config.CHECKPOINT_PATH)
    return all_losses


# ============================================================
# SECTION 6 — MAP DATABASE
# ============================================================

class MapVectorDatabase:
    def __init__(self):
        self.vectors = np.empty((0, Config.EMBEDDING_DIM), dtype=np.float32)
        self.ids     = []
        self._nn     = None

    def build(self, sat_folder, inference_model, label="satellite"):
        print(f"[DB] Building {label} database...")
        vecs, ids = [], []
        BATCH = 32

        all_items = []
        for loc_id in sorted(os.listdir(sat_folder)):
            for p in get_imgs(sat_folder, loc_id):
                all_items.append((loc_id, p))

        print(f"[DB] Embedding {len(all_items)} images...")
        t0 = time.time()

        for i in range(0, len(all_items), BATCH):
            batch = all_items[i:i+BATCH]
            imgs  = np.stack([load_img_fast(p) for _, p in batch])
            embs  = inference_model.predict(imgs, verbose=0)
            vecs.extend(embs)
            ids.extend(loc for loc, _ in batch)

            if (i // BATCH) % 10 == 0:
                pct = (i+len(batch)) / len(all_items) * 100
                print(f"  {pct:5.1f}%  ({i+len(batch)}/{len(all_items)})  "
                      f"[{time.time()-t0:.0f}s]")

        self.vectors = np.array(vecs, dtype=np.float32)
        self.ids     = ids
        self._build_index()
        print(f"[DB] Done — {len(self.vectors)} vectors in "
              f"{time.time()-t0:.0f}s\n")

    def _build_index(self):
        k = min(Config.TOP_K, len(self.vectors))
        self._nn = NearestNeighbors(
            n_neighbors=k, metric='cosine',
            algorithm='brute', n_jobs=-1
        )
        self._nn.fit(self.vectors)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'vectors': self.vectors, 'ids': self.ids}, f, protocol=4)
        mb = os.path.getsize(path) / 1024 / 1024
        print(f"[DB] Saved → {path}  ({mb:.1f} MB)")

    def load(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.vectors = d['vectors']
        self.ids     = d['ids']
        self._build_index()
        print(f"[DB] Loaded {len(self.vectors)} vectors from {path}")

    def search(self, query_vec, k=5):
        dists, idx = self._nn.kneighbors(
            query_vec.reshape(1, -1).astype(np.float32),
            n_neighbors=k
        )
        return dists[0], [self.ids[i] for i in idx[0]]


# ============================================================
# SECTION 7 — TFLITE EXPORT
# ============================================================

def export_tflite(inference_model):
    print("[TFLite] Converting model...")
    conv = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    conv.optimizations         = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    model_bytes = conv.convert()
    with open(Config.TFLITE_PATH, 'wb') as f:
        f.write(model_bytes)
    mb = os.path.getsize(Config.TFLITE_PATH) / 1024 / 1024
    print(f"[TFLite] Saved → {Config.TFLITE_PATH}  ({mb:.2f} MB)\n")


# ============================================================
# SECTION 8 — EVALUATION
# ============================================================

def evaluate(inference_model, test_db):
    print(f"{'='*55}")
    print("  EVALUATION — Recall@K")
    print(f"{'='*55}")

    query_ids = sorted(os.listdir(Config.QUERY_DRONE_PATH))
    results   = []
    t0        = time.time()

    count = 0
    for loc_id in query_ids:
        imgs = get_imgs(Config.QUERY_DRONE_PATH, loc_id)
        if not imgs:
            continue
        # Use only first image per location for speed
        img  = load_img_fast(imgs[0])
        emb  = inference_model.predict(np.expand_dims(img, 0), verbose=0)[0]
        _, pred_ids = test_db.search(emb, k=10)
        results.append((loc_id, pred_ids))
        count += 1
        if count >= Config.MAX_QUERY_IMGS:
            break

    total   = len(results)
    recalls = {}
    for k in [1, 5, 10]:
        c = sum(1 for (tid, pids) in results if tid in pids[:k])
        recalls[k] = c / total if total > 0 else 0.0

    print(f"\n  Queries evaluated : {total}")
    print(f"  Eval time         : {time.time()-t0:.0f}s")
    for k, r in recalls.items():
        bar = '█' * int(r * 30)
        print(f"  Recall@{k:<3} : {r:.4f}  [{bar}]")
    print(f"{'='*55}\n")
    return recalls


# ============================================================
# SECTION 9 — VISUALIZATION
# ============================================================

def visualize(inference_model, test_db, n=3):
    """Save a retrieval result grid (no display window needed)."""
    query_ids  = sorted(os.listdir(Config.QUERY_DRONE_PATH))
    sample_ids = random.sample(query_ids, min(n, len(query_ids)))

    fig, axes = plt.subplots(n, 6, figsize=(18, n * 3.5))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Drone Geolocalization — Retrieval Results\n"
        "Column 1: Query (Drone)   |   Columns 2–6: Top-5 Retrieved (Satellite)\n"
        "Green = Correct   Red = Wrong",
        fontsize=10
    )

    for row, loc_id in enumerate(sample_ids):
        imgs = get_imgs(Config.QUERY_DRONE_PATH, loc_id)
        if not imgs:
            continue

        # Embed query
        img = load_img_fast(random.choice(imgs))
        emb = inference_model.predict(np.expand_dims(img, 0), verbose=0)[0]
        dists, pred_ids = test_db.search(emb, k=5)

        # Denormalize for display
        def to_display(arr):
            arr = (arr + 1.0) / 2.0  # [-1,1] → [0,1]
            return np.clip(arr, 0, 1)

        axes[row, 0].imshow(to_display(img))
        axes[row, 0].set_title(f"QUERY\n{loc_id}", fontsize=8,
                                color='white', backgroundcolor='steelblue')
        axes[row, 0].axis('off')

        for ci, (pid, dist) in enumerate(zip(pred_ids, dists)):
            sat_imgs = get_imgs(Config.GALLERY_SAT_PATH, pid)
            ax = axes[row, ci + 1]
            if not sat_imgs:
                ax.axis('off')
                continue
            s_img = load_img_fast(random.choice(sat_imgs))
            ax.imshow(to_display(s_img))
            ok    = pid == loc_id
            color = 'limegreen' if ok else 'tomato'
            ax.set_title(f"#{ci+1} {'✓' if ok else '✗'}\n{pid}\n{dist:.3f}",
                         fontsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(color)
                sp.set_linewidth(4)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

    plt.tight_layout()
    out = os.path.join(Config.OUTPUT_DIR, "retrieval_results.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Retrieval grid saved → {out}")


def plot_loss(losses):
    plt.figure(figsize=(9, 4))
    plt.plot(range(1, len(losses)+1), losses,
             marker='o', lw=2, color='steelblue', ms=6)
    plt.title(f"Training Loss — {Config.EPOCHS} Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Triplet Loss")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    out = os.path.join(Config.OUTPUT_DIR, "training_loss.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Plot] Loss curve saved → {out}")


# ============================================================
# SECTION 10 — MAIN
# ============================================================

def main():
    wall_start = time.time()

    print("\n" + "="*55)
    print("  DRONE VISUAL GEOLOCALIZATION — FAST PROTOTYPE")
    print(f"  Target: < 1 hour on Ryzen 5 5500U")
    print("="*55 + "\n")

    # Validate paths
    for name, path in [("TRAIN_DRONE", Config.TRAIN_DRONE_PATH),
                        ("TRAIN_SAT",   Config.TRAIN_SAT_PATH)]:
        if not os.path.isdir(path):
            print(f"[ERROR] {name} not found:\n  {path}")
            print("\n  → Open drone_geo_fast.py and update:")
            print("    Config.DATASET_PATH = r'C:\\path\\to\\University-1652'")
            sys.exit(1)

    # ── Step 1: Build model ────────────────────────────────────
    print("[1/6] Building model...")
    siamese, inference = build_model()
    print()

    # ── Step 2: Train ──────────────────────────────────────────
    print("[2/6] Training...\n")
    losses = train(siamese, inference)
    plot_loss(losses)

    # ── Step 3: Build training satellite DB ───────────────────
    print("[3/6] Building training map database...")
    train_db = MapVectorDatabase()
    train_db.build(Config.TRAIN_SAT_PATH, inference, label="train-satellite")
    train_db.save(Config.DB_PATH)

    # ── Step 4: Export TFLite ─────────────────────────────────
    print("[4/6] Exporting TFLite model...")
    export_tflite(inference)

    # ── Step 5: Build test gallery DB + evaluate ──────────────
    print("[5/6] Building test gallery database...")
    test_db = MapVectorDatabase()
    test_db.build(Config.GALLERY_SAT_PATH, inference, label="test-satellite")

    print("[6/6] Evaluating...")
    recalls = evaluate(inference, test_db)

    # ── Visualization ──────────────────────────────────────────
    print("[Viz] Generating retrieval visualization...")
    visualize(inference, test_db, n=3)

    # ── Final summary ──────────────────────────────────────────
    wall_min = (time.time() - wall_start) / 60

    print("\n" + "="*55)
    print("  ALL DONE")
    print(f"  Total wall time  : {wall_min:.1f} minutes")
    print(f"  Recall@1         : {recalls.get(1,  0):.4f}")
    print(f"  Recall@5         : {recalls.get(5,  0):.4f}")
    print(f"  Recall@10        : {recalls.get(10, 0):.4f}")
    print(f"  Outputs saved to : {Config.OUTPUT_DIR}/")
    print("="*55 + "\n")

    if wall_min > 60:
        print("  [NOTE] Ran over 1 hour. To speed up further, reduce:")
        print("    Config.MAX_LOCATIONS  (currently", Config.MAX_LOCATIONS, ")")
        print("    Config.EPOCHS         (currently", Config.EPOCHS, ")")
        print("    Config.TRIPLETS_PER_LOC (currently", Config.TRIPLETS_PER_LOC, ")")


if __name__ == "__main__":
    main()
