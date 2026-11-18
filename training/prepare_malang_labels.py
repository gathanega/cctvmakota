"""
CCTVMAKOTA - Pre-labeling frame Malang untuk fine-tune
=======================================================
Alur:
 1. Pilih frame terpilih (stratified per kamera) dari dataset_malang_live
 2. Deteksi pakai YOLO11x COCO (full 1280 + tiles) sebagai PRE-LABEL
 3. Tulis label format YOLO (0=car, 1=motorcycle, 2=bus, 3=truck)
 4. Frame yang gagal terdeteksi diulang dengan peningkatan kontras (CLAHE) +
    threshold lebih rendah; kalau tetap kosong -> TIDAK dijadikan background,
    tapi dipindah ke needs_manual/ untuk dicek manusia
 5. Susun dataset: dataset_malang_labeled/{images,labels}/{train,val} + data.yaml
 6. Buat preview berkotak + galeri HTML + daftar frame yang perlu dikoreksi

Label ini masih "pre-label" -> harus dikoreksi manual sebelum fine-tune final.

Pemakaian:
    python prepare_malang_labels.py            # default 400 gambar
    python prepare_malang_labels.py 600        # jumlah gambar lain
"""

import sys
import csv
import time
import shutil
import random
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # training/ -> root proyek
sys.path.insert(0, str(ROOT))
LIVE_DIR = ROOT / "dataset_malang_live"
OUT_DS = ROOT / "dataset_malang_labeled"
REPORT_DIR = ROOT / "Output" / "reports"
REVIEW_DIR = OUT_DS / "review"

MODEL_PATH = ROOT / "models" / "yolo11x.pt"

TARGET_IMAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 400
TRAIN_RATIO = 0.8
SEED = 42

# parameter deteksi untuk pre-label (lebih konservatif = label lebih bersih)
FULL_IMGSZ = 1280
FULL_CONF = 0.20
TILE_IMGSZ = 640
TILE_CONF = 0.20
TILE_ROWS = 2
TILE_COLS = 3
TILE_OVERLAP = 0.25
NMS_IOU = 0.45

CLASS_CONF = {"car": 0.22, "motorcycle": 0.20, "bus": 0.35, "truck": 0.50}
MIN_BOX_AREA_RATIO = 0.00020
MAX_ASPECT = 7.0

# --- pass kedua untuk frame yang gagal terdeteksi (hujan/blur/kontras rendah) ---
RETRY_CONF = 0.08
RETRY_CLASS_CONF = {k: 0.10 for k in CLASS_CONF}
RETRY_IMGSZ = 1536

# --- filter kerucut/pembatas lalu lintas oranye yang sering dilabeli truck/car ---
BARRIER_ORANGE_RATIO = 0.45
BARRIER_MAX_ASPECT = 1.00      # hanya kotak yang lebih tinggi daripada lebar
BARRIER_CLASSES = {0, 3}       # car, truck

NAME_TO_ID = {"car": 0, "motorcycle": 1, "bus": 2, "truck": 3}
COLORS = {0: (60, 200, 60), 1: (0, 160, 255), 2: (255, 120, 0), 3: (60, 60, 235)}
ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}


def gen_tiles(h, w):
    th = int(h / (TILE_ROWS - (TILE_ROWS - 1) * TILE_OVERLAP))
    tw = int(w / (TILE_COLS - (TILE_COLS - 1) * TILE_OVERLAP))
    th, tw = min(th, h), min(tw, w)
    ys = np.linspace(0, max(0, h - th), TILE_ROWS).astype(int)
    xs = np.linspace(0, max(0, w - tw), TILE_COLS).astype(int)
    for y in ys:
        for x in xs:
            yield int(x), int(y), tw, th


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)
    bb = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
    return inter / (aa + bb - inter + 1e-9)


def containment(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)
    bb = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
    return inter / min(aa, bb)


def dedupe(boxes, scores, ids):
    """NMS per kelas + suppression antar kelas + containment."""
    order = np.argsort(-scores)
    keep = []
    for i in order:
        drop = False
        for j in keep:
            if iou(boxes[i], boxes[j]) > (0.45 if ids[i] == ids[j] else 0.55):
                drop = True
                break
            if containment(boxes[i], boxes[j]) > 0.75:
                drop = True
                break
        if not drop:
            keep.append(i)
    keep = np.array(keep, dtype=int)
    return boxes[keep], scores[keep], ids[keep]


def enhance(img):
    """CLAHE pada kanal L -> bantu frame hujan/mendung yang kontrasnya rendah."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def is_orange_barrier(img, box):
    """True jika kotak didominasi warna oranye dan lebih tinggi daripada lebar."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return False
    if (x2 - x1) / max(y2 - y1, 1) > BARRIER_MAX_ASPECT:
        return False
    patch = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    orange = (((h <= 22) | (h >= 165)) & (s >= 110) & (v >= 90))
    return float(orange.mean()) >= BARRIER_ORANGE_RATIO


def detect(model, img, device, conf=None, class_conf=None, imgsz=None):
    h, w = img.shape[:2]
    names = model.names
    parts = []

    full_conf = FULL_CONF if conf is None else conf
    tile_conf = TILE_CONF if conf is None else conf
    full_sz = FULL_IMGSZ if imgsz is None else imgsz
    thr = CLASS_CONF if class_conf is None else class_conf

    r = model.predict(img, conf=full_conf, iou=NMS_IOU, imgsz=full_sz,
                      device=device, verbose=False)[0]
    if len(r.boxes):
        parts.append((r.boxes.xyxy.cpu().numpy().copy(),
                      r.boxes.conf.cpu().numpy(),
                      r.boxes.cls.cpu().numpy().astype(int)))

    for x, y, tw, th in gen_tiles(h, w):
        crop = img[y:y + th, x:x + tw]
        if crop.size == 0:
            continue
        rt = model.predict(crop, conf=tile_conf, iou=NMS_IOU, imgsz=TILE_IMGSZ,
                           device=device, verbose=False)[0]
        if not len(rt.boxes):
            continue
        b = rt.boxes.xyxy.cpu().numpy().copy()
        b[:, [0, 2]] += x
        b[:, [1, 3]] += y
        cx = (b[:, 0] + b[:, 2]) / 2
        cy = (b[:, 1] + b[:, 3]) / 2
        inside = (cx >= x) & (cx <= x + tw) & (cy >= y) & (cy <= y + th)
        if inside.any():
            parts.append((b[inside],
                          rt.boxes.conf.cpu().numpy()[inside],
                          rt.boxes.cls.cpu().numpy().astype(int)[inside]))

    if not parts:
        return np.empty((0, 4)), np.array([]), np.array([], dtype=int)

    boxes = np.vstack([p[0] for p in parts])
    scores = np.concatenate([p[1] for p in parts])
    coco_ids = np.concatenate([p[2] for p in parts])

    labels = np.array([names[int(c)] for c in coco_ids])
    keep_mask = np.isin(labels, list(NAME_TO_ID.keys()))
    boxes, scores, coco_ids, labels = boxes[keep_mask], scores[keep_mask], coco_ids[keep_mask], labels[keep_mask]
    if len(boxes) == 0:
        return boxes, scores, np.array([], dtype=int)

    ids = np.array([NAME_TO_ID[l] for l in labels], dtype=int)

    # filter ukuran
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    area_ratio = (bw * bh) / float(w * h)
    aspect = bw / np.maximum(bh, 1)
    ok = ((area_ratio >= MIN_BOX_AREA_RATIO) & (bw > 6) & (bh > 6)
          & (aspect <= MAX_ASPECT) & (aspect >= 1 / MAX_ASPECT))
    boxes, scores, ids = boxes[ok], scores[ok], ids[ok]
    if len(boxes) == 0:
        return boxes, scores, ids

    boxes, scores, ids = dedupe(boxes, scores, ids)

    # buang pembatas/kerucut oranye yang mirip truk
    ok = np.array([not (int(c) in BARRIER_CLASSES and is_orange_barrier(img, b))
                   for b, c in zip(boxes, ids)], dtype=bool)
    boxes, scores, ids = boxes[ok], scores[ok], ids[ok]
    if len(boxes) == 0:
        return boxes, scores, ids

    # threshold per kelas
    keep = [i for i, (s, c) in enumerate(zip(scores, ids))
            if s >= thr.get(ID_TO_NAME[int(c)], 0.25)]
    keep = np.array(keep, dtype=int)
    if len(keep) == 0:
        return boxes[:0], scores[:0], ids[:0]
    return boxes[keep], scores[keep], ids[keep]


def draw(img, boxes, ids, scores):
    out = img.copy()
    for (x1, y1, x2, y2), c, s in zip(boxes, ids, scores):
        col = COLORS[int(c)]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        t = f"{ID_TO_NAME[int(c)]} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, t, (x1 + 2, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"{len(boxes)}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, f"{len(boxes)}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    import torch
    from ultralytics import YOLO

    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 80)
    print(" PRE-LABELING FRAME CCTV MALANG")
    print("=" * 80)
    print(f" Sumber      : {LIVE_DIR}")
    print(f" Output      : {OUT_DS}")
    print(f" Model       : {MODEL_PATH.name}")
    print(f" Target      : {TARGET_IMAGES} gambar (train {int(TRAIN_RATIO*100)}% / val {100-int(TRAIN_RATIO*100)}%)")
    print(f" Conf        : full {FULL_CONF} | tile {TILE_CONF} | per kelas {CLASS_CONF}")
    print("=" * 80)

    if not MODEL_PATH.exists():
        print(f"[ERROR] model tidak ada: {MODEL_PATH}")
        sys.exit(1)

    device = 0 if torch.cuda.is_available() else "cpu"

    # bersihkan output lama
    if OUT_DS.exists():
        shutil.rmtree(OUT_DS)
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "review", "needs_manual"):
        (OUT_DS / sub).mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --- pilih frame stratified per kamera ---
    cams = {}
    for cam_dir in sorted(p for p in LIVE_DIR.glob("*CCTV") if p.is_dir()):
        fs = sorted(cam_dir.glob("*.jpg"))
        if fs:
            cams[cam_dir.name] = fs

    total_avail = sum(len(v) for v in cams.values())
    picks = []
    for cam, fs in cams.items():
        quota = max(1, int(round(TARGET_IMAGES * len(fs) / total_avail)))
        quota = min(quota, len(fs))
        idx = np.linspace(0, len(fs) - 1, quota).astype(int)
        picks.extend([fs[i] for i in sorted(set(idx.tolist()))])

    random.shuffle(picks)
    picks = picks[:TARGET_IMAGES]
    random.shuffle(picks)

    n_train = int(len(picks) * TRAIN_RATIO)
    assigned = [("train", f) for f in picks[:n_train]] + [("val", f) for f in picks[n_train:]]

    print(f"\nKamera      : {len(cams)}")
    print(f"Frame dipilih: {len(assigned)}  (train {n_train}, val {len(assigned)-n_train})\n")

    model = YOLO(str(MODEL_PATH))
    t0 = time.time()
    rows = []
    stats = {0: 0, 1: 0, 2: 0, 3: 0}
    total_boxes = 0
    empty = 0
    low_conf = 0
    retried = 0
    recovered = 0
    needs_manual = OUT_DS / "needs_manual"

    for i, (split, fpath) in enumerate(assigned, 1):
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        h, w = img.shape[:2]

        boxes, scores, ids = detect(model, img, device)
        via = "normal"

        # pass kedua: kontras dinaikkan + threshold lebih rendah
        if len(boxes) == 0:
            retried += 1
            enh = enhance(img)
            boxes, scores, ids = detect(model, enh, device, conf=RETRY_CONF,
                                        class_conf=RETRY_CLASS_CONF, imgsz=RETRY_IMGSZ)
            if len(boxes):
                recovered += 1
                via = "retry-klap"

        img_name = f"{fpath.parent.name}_{fpath.stem}.jpg"

        # tetap tidak ada kendaraan -> jangan jadikan label background,
        # pindahkan ke needs_manual supaya dicek manusia
        if len(boxes) == 0:
            shutil.copy2(fpath, needs_manual / img_name)
            prev = draw(img, boxes, ids, scores)
            cv2.imwrite(str(REVIEW_DIR / img_name.replace(".jpg", "_prev.jpg")), prev,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            empty += 1
            rows.append({
                "image": img_name,
                "split": "needs_manual",
                "camera": fpath.parent.name,
                "source": str(fpath),
                "boxes": 0,
                "car": 0, "motorcycle": 0, "bus": 0, "truck": 0,
                "conf_mean": 0.0,
                "conf_min": 0.0,
                "detection_pass": "gagal",
            })
            if i % 50 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(assigned)} ({el:.0f}s, {el/i*1000:.0f} ms/gambar)")
            continue

        # tulis label YOLO
        lbl_name = f"{fpath.parent.name}_{fpath.stem}.txt"
        with open(OUT_DS / "labels" / split / lbl_name, "w", encoding="utf-8") as fh:
            for (x1, y1, x2, y2), c, s in zip(boxes, ids, scores):
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                fh.write(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        # copy gambar
        shutil.copy2(fpath, OUT_DS / "images" / split / img_name)

        # preview review
        prev = draw(img, boxes, ids, scores)
        cv2.imwrite(str(REVIEW_DIR / img_name.replace(".jpg", "_prev.jpg")), prev,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

        n = len(boxes)
        total_boxes += n
        for c in ids:
            stats[int(c)] = stats.get(int(c), 0) + 1
        if n > 0 and float(np.max(scores)) < 0.35:
            low_conf += 1

        rows.append({
            "image": img_name,
            "split": split,
            "camera": fpath.parent.name,
            "source": str(fpath),
            "boxes": n,
            "car": int((ids == 0).sum()),
            "motorcycle": int((ids == 1).sum()),
            "bus": int((ids == 2).sum()),
            "truck": int((ids == 3).sum()),
            "conf_mean": float(scores.mean()) if n else 0.0,
            "conf_min": float(scores.min()) if n else 0.0,
            "detection_pass": via,
        })

        if i % 50 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(assigned)} ({el:.0f}s, {el/i*1000:.0f} ms/gambar)")

    dur = time.time() - t0

    # --- data.yaml ---
    yaml_path = OUT_DS / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write("# Pre-label CCTV Malang (perlu koreksi manual sebelum training final)\n")
        fh.write(f"path: {str(OUT_DS).replace(chr(92), '/')}\n")
        fh.write("train: images/train\nval: images/val\n\nnc: 4\nnames:\n")
        fh.write("  0: car\n  1: motorcycle\n  2: bus\n  3: truck\n")

    # --- CSV ---
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORT_DIR / f"prelabel_malang_{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        fh.write("\nSUMMARY\n")
        fh.write(f"total_images,{len(rows)}\n")
        fh.write(f"total_boxes,{total_boxes}\n")
        fh.write(f"images_without_detection,{empty}\n")
        fh.write(f"images_retried,{retried}\n")
        fh.write(f"images_recovered_by_retry,{recovered}\n")
        fh.write(f"images_low_confidence,{low_conf}\n")
        for k in sorted(stats):
            fh.write(f"total_{ID_TO_NAME[k]},{stats[k]}\n")
        fh.write(f"duration_seconds,{dur:.1f}\n")

    # --- galeri HTML untuk koreksi manual ---
    html = OUT_DS / "review" / "index.html"
    with open(html, "w", encoding="utf-8") as fh:
        fh.write("<!doctype html><meta charset='utf-8'><title>Review Pre-label Malang</title>")
        fh.write("<style>body{font-family:system-ui;background:#0f172a;color:#e2e8f0;margin:20px}"
                 "h1{font-size:20px}table{border-collapse:collapse;font-size:13px}"
                 "td,th{border:1px solid #334155;padding:4px 8px}img{width:320px;border-radius:6px}"
                 "tr:nth-child(even){background:#16213a}"
                 ".warn{color:#fbbf24;font-weight:600}</style>")
        fh.write(f"<h1>Review Pre-label — {len(rows)} gambar</h1>")
        fh.write(f"<p>Total kotak: {total_boxes} · tanpa deteksi: {empty} "
                 f"(dipindah ke <code>needs_manual/</code>, tidak dipakai training) · "
                 f"confidence rendah: {low_conf} · pulih lewat retry: {recovered}/{retried}</p>")
        fh.write("<p>Koreksi file .txt di <code>labels/train</code> atau <code>labels/val</code>. "
                 "Format YOLO: <code>kelas cx cy w h</code> (nilai 0–1).</p>")
        fh.write("<table><tr><th>Preview</th><th>File</th><th>Split</th><th>Jumlah</th><th>Detail</th></tr>")
        for r in rows:
            warn = " class='warn'" if r["boxes"] == 0 or r["conf_mean"] < 0.35 else ""
            fh.write(
                f"<tr{warn}><td><a href='{r['image'].replace('.jpg','_prev.jpg')}'>"
                f"<img src='{r['image'].replace('.jpg','_prev.jpg')}'></a></td>"
                f"<td>{r['image']}</td><td>{r['split']}</td><td>{r['boxes']}</td>"
                f"<td>car {r['car']} · motor {r['motorcycle']} · bus {r['bus']} · truk {r['truck']}<br>"
                f"conf avg {r['conf_mean']:.2f} · {r['detection_pass']}</td></tr>"
            )
        fh.write("</table>")

    print("\n" + "=" * 80)
    print(" RINGKASAN PRE-LABEL")
    print("=" * 80)
    print(f" Gambar diproses    : {len(rows)}")
    print(f" Total kotak        : {total_boxes}")
    print(f" Retry pass kedua   : {retried} gambar  (pulih: {recovered})")
    print(f" Tetap kosong       : {empty}  -> needs_manual/ (TIDAK dipakai training)")
    print(f" Conf rendah (<.35) : {low_conf}")
    print(f"   car        : {stats.get(0,0)}")
    print(f"   motorcycle : {stats.get(1,0)}")
    print(f"   bus        : {stats.get(2,0)}")
    print(f"   truck      : {stats.get(3,0)}")
    print(f" Durasi             : {dur:.0f}s ({dur/max(len(rows),1)*1000:.0f} ms/gambar)")
    print("-" * 80)
    print(f" Dataset   : {OUT_DS}")
    print(f" data.yaml : {yaml_path}")
    print(f" CSV       : {csv_path}")
    print(f" Review    : {html}")
    print("=" * 80)
    print(" LANJUTAN: buka review/index.html, koreksi label, lalu fine-tune:")
    print("   python train_yolo11x_malang_live.py")


if __name__ == "__main__":
    main()
