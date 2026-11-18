"""
Benchmark checkpoint COCO pada frame CCTV Malang -> pilih model untuk dashboard.

Referensi: dataset_malang_labeled/labels/val (80 gambar pre-label).
Bias: pre-label dibuat oleh yolo11x (full+tiles+retry), jadi skor yolo11x
sedikit diuntungkan. Model lain tetap dibandingkan secara adil satu sama lain.

    python benchmark_coco.py
Output: Output/reports/benchmark_coco_<stamp>.csv
"""
import sys
import csv
import time
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # training/ -> root proyek
sys.path.insert(0, str(ROOT))
VAL_IMG = ROOT / "dataset_malang_labeled" / "images" / "val"
VAL_LBL = ROOT / "dataset_malang_labeled" / "labels" / "val"
MODELS = ["models/yolov8n.pt", "models/yolo11s.pt", "models/yolo11x.pt", "runs/yolo11s_malang_live/weights/best.pt"]
IMGSZ = 1280
CONF_AP = 0.05     # konf rendah untuk kurva AP
CONF_OP = 0.25     # titik operasi untuk P/R
IOU_MATCH = 0.5
COCO_TO_ID = {"car": 0, "motorcycle": 1, "bus": 2, "truck": 3}
NAMES = ["car", "motorcycle", "bus", "truck"]


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match(pred_boxes, pred_cls, pred_conf, gt_boxes, gt_cls):
    """Greedy per kelas, skor tertinggi dulu. Return (tp per prediksi, gt yang tertangkap)."""
    tp = np.zeros(len(pred_boxes), dtype=bool)
    used = np.zeros(len(gt_boxes), dtype=bool)
    ious = iou_matrix(pred_boxes, gt_boxes)
    for i in np.argsort(-pred_conf):
        cand = np.where((gt_cls == pred_cls[i]) & ~used)[0]
        if len(cand) == 0:
            continue
        j = cand[np.argmax(ious[i, cand])]
        if ious[i, j] >= IOU_MATCH:
            tp[i] = used[j] = True
    return tp, used


def size_bucket(boxes, frame_h):
    """Tinggi kotak relatif frame: kecil (<4%) = kendaraan jauh, sumber utama false negative."""
    rel = (boxes[:, 3] - boxes[:, 1]) / frame_h
    return np.where(rel < 0.04, "kecil", np.where(rel < 0.10, "sedang", "besar"))


def load_gt(lbl, w, h):
    rows = [l.split() for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()] if lbl.exists() else []
    if not rows:
        return np.empty((0, 4)), np.array([], dtype=int)
    a = np.array(rows, dtype=float)
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1), a[:, 0].astype(int)


def self_check():
    a = np.array([[0, 0, 10, 10]], float)
    assert abs(iou_matrix(a, a)[0, 0] - 1) < 1e-6
    assert iou_matrix(a, np.array([[20, 20, 30, 30]], float))[0, 0] == 0
    tp, used = match(a, np.array([0]), np.array([0.9]), a, np.array([0]))
    assert tp.tolist() == [True] and used.tolist() == [True]
    tp, used = match(a, np.array([1]), np.array([0.9]), a, np.array([0]))   # kelas beda -> FP + FN
    assert tp.tolist() == [False] and used.tolist() == [False]
    assert size_bucket(np.array([[0, 0, 5, 20], [0, 0, 5, 200]], float), 1000).tolist() == ["kecil", "besar"]


def main():
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import ap_per_class

    self_check()
    device = 0 if torch.cuda.is_available() else "cpu"
    imgs = sorted(VAL_IMG.glob("*.jpg"))
    if not imgs:
        sys.exit(f"[ERROR] tidak ada gambar di {VAL_IMG}. Jalankan prepare_malang_labels.py dulu.")
    frames = [cv2.imread(str(p)) for p in imgs]
    gts = [load_gt(VAL_LBL / (p.stem + ".txt"), f.shape[1], f.shape[0]) for p, f in zip(imgs, frames)]
    n_gt = sum(len(g[1]) for g in gts)
    print(f"Gambar val: {len(imgs)} | kotak referensi: {n_gt} | imgsz {IMGSZ} | device {device}\n")

    results = []
    for name in MODELS:
        if not (ROOT / name).exists():
            print(f"  lewati {name} (tidak ada)")
            continue
        model = YOLO(str(ROOT / name))
        veh = {i: COCO_TO_ID[n] for i, n in model.names.items() if n in COCO_TO_ID}
        model.predict(frames[0], imgsz=IMGSZ, device=device, verbose=False)   # warm-up
        all_tp, all_conf, all_cls, all_tcls, times = [], [], [], [], []
        fn = {}   # kondisi -> [terlewat, total]
        worst = []
        for p_img, f, (gb, gc) in zip(imgs, frames, gts):
            t = time.perf_counter()
            r = model.predict(f, conf=CONF_AP, iou=0.5, imgsz=IMGSZ, device=device,
                              classes=list(veh), verbose=False)[0]
            times.append(time.perf_counter() - t)
            pb = r.boxes.xyxy.cpu().numpy()
            pc = np.array([veh[int(c)] for c in r.boxes.cls.cpu().numpy()], dtype=int)
            pf = r.boxes.conf.cpu().numpy()
            all_tp.append(match(pb, pc, pf, gb, gc)[0]); all_conf.append(pf); all_cls.append(pc); all_tcls.append(gc)
            op_i = pf >= CONF_OP
            hit = match(pb[op_i], pc[op_i], pf[op_i], gb, gc)[1]
            light = "malam" if cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() < 70 else "siang"
            for key in [light, *size_bucket(gb, f.shape[0]).tolist(), *[NAMES[c] for c in gc]]:
                fn.setdefault(key, [0, 0])
            for k_size, k_cls, h in zip(size_bucket(gb, f.shape[0]), gc, hit):
                for key in (light, k_size, NAMES[k_cls]):
                    fn[key][0] += int(not h); fn[key][1] += 1
            worst.append((int((~hit).sum()), p_img.name))
        tp = np.concatenate(all_tp); conf = np.concatenate(all_conf)
        pcls = np.concatenate(all_cls); tcls = np.concatenate(all_tcls)
        ap = ap_per_class(tp[:, None], conf, pcls, tcls)[5][:, 0]   # AP@0.5 per kelas yang ada
        op = conf >= CONF_OP
        p = tp[op].sum() / max(op.sum(), 1)
        rc = tp[op].sum() / max(len(tcls), 1)
        row = {"model": name, "params_M": round(sum(x.numel() for x in model.model.parameters()) / 1e6, 1),
               "mAP50": float(ap.mean()), "precision": float(p), "recall": float(rc),
               "f1": float(2 * p * rc / max(p + rc, 1e-9)), "ms_per_frame": 1000 * float(np.mean(times)),
               "detections": int(op.sum())}
        present = np.unique(tcls)
        for k, c in enumerate(present):
            row[f"AP50_{NAMES[c]}"] = float(ap[k])
        for key, (miss, tot) in fn.items():
            row[f"FN%_{key}"] = 100.0 * miss / max(tot, 1)
        results.append(row)
        print(f"  {name:<11} mAP50 {row['mAP50']:.3f}  P {p:.3f}  R {rc:.3f}  F1 {row['f1']:.3f}  "
              f"{row['ms_per_frame']:.0f} ms  ({row['params_M']}M param)")
        print("      false negative: " + "  ".join(f"{k} {m}/{t}" for k, (m, t) in sorted(fn.items())))
        print("      frame terburuk: " + ", ".join(f"{n} ({m})" for m, n in sorted(worst, reverse=True)[:3]))
        del model
        if device == 0:
            torch.cuda.empty_cache()

    best = max(results, key=lambda r: r["f1"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "Output" / "reports" / f"benchmark_coco_{stamp}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in results for k in r}, key=lambda k: (k != "model", k))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in results)
        fh.write(f"\nbest_by_f1,{best['model']}\nreference,pre-label yolo11x (bias ke yolo11x)\n")
    print(f"\nTerbaik (F1): {best['model']}\nCSV: {out}")


if __name__ == "__main__":
    main()
