"""
CCTVMAKOTA - Fine-tune YOLOv11x pada frame CCTV Malang
======================================================
Dataset : dataset_malang_labeled  (hasil prepare_malang_labels.py, sudah dikoreksi)
Base    : yolo11x.pt (COCO)  ->  fine-tune, BUKAN dari nol
Output  : runs/yolo11x_malang_live/weights/best.pt, CSV, PDF

Pemakaian:
    python training/train_malang.py              # default 40 epoch
    python training/train_malang.py 60 960       # epoch, imgsz
    python training/train_malang.py 80 960 yolo11s.pt   # model ringan (produksi CPU/GPU kecil)

Catatan:
  - Mulai dari yolo11x.pt COCO (bukan dataset drone) supaya tidak regresi.
  - imgsz 960 (bukan 640) karena kendaraan CCTV Malang relatif kecil.
  - freeze=0: semua layer dilatih. Kalau dataset masih < 500 gambar,
    pakai freeze=10 (lihat komentar di bagian train).
"""

import sys
import csv
import json
import time
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # training/ -> root proyek
sys.path.insert(0, str(ROOT))
DATA_YAML = ROOT / "dataset_malang_labeled" / "data.yaml"
LIVE_DIR = ROOT / "dataset_malang_live"
RUNS_DIR = ROOT / "runs"
REPORTS_DIR = ROOT / "Output" / "reports"

BASE_MODEL = sys.argv[3] if len(sys.argv) > 3 else "yolo11s.pt"
if (ROOT / "models" / BASE_MODEL).exists():
    BASE_MODEL = str(ROOT / "models" / BASE_MODEL)
RUN_NAME = f"{Path(BASE_MODEL).stem}_malang_live"

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 960
BATCH = 8 if "11x" in BASE_MODEL else 16
WORKERS = 4
SEED = 42
PATIENCE = 15

VAL_IMGSZ = IMGSZ
LIVE_CONF = 0.20
LIVE_IOU = 0.50
LIVE_IMGSZ = 1280
MAX_LIVE_FRAMES = 200

NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}
COLORS = {0: (60, 200, 60), 1: (0, 160, 255), 2: (255, 120, 0), 3: (60, 60, 235)}


def banner(text):
    line = "=" * 80
    print(f"\n{line}\n {text}\n{line}")


def verify_dataset():
    """Cek gambar vs label + hitung distribusi kelas."""
    base = DATA_YAML.parent
    stats = {}
    for split in ("train", "val"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        imgs = sorted(img_dir.glob("*.jpg")) if img_dir.exists() else []
        missing = 0
        empty = 0
        per_class = {i: 0 for i in NAMES}
        for img in imgs:
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                missing += 1
                continue
            lines = [l for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                empty += 1
                continue
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    c = int(float(parts[0]))
                    per_class[c] = per_class.get(c, 0) + 1
        stats[split] = {
            "images": len(imgs),
            "missing_labels": missing,
            "empty_labels": empty,
            "per_class": per_class,
            "total_boxes": sum(per_class.values()),
        }
    return stats


def write_pdf(path, info):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    )

    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9, leading=13)

    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm)
    story = []

    story.append(Paragraph("CCTVMAKOTA — Fine-tune YOLOv11x pada CCTV Malang", h1))
    story.append(Paragraph(
        f"Model dasar <b>{info['base_model']}</b> (COCO) · "
        f"{info['epochs']} epoch · imgsz {info['imgsz']} · batch {info['batch']}<br/>"
        f"Dataset <b>dataset_malang_labeled</b> · dibuat {info['stamp']}", body))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Dataset", h2))
    rows = [["Split", "Gambar", "Tanpa label", "Label kosong", "Total kotak",
             "car", "motor", "bus", "truck"]]
    for split, s in info["dataset"].items():
        pc = s["per_class"]
        rows.append([split, s["images"], s["missing_labels"], s["empty_labels"],
                     s["total_boxes"], pc.get(0, 0), pc.get(1, 0), pc.get(2, 0), pc.get(3, 0)])
    t = Table(rows, colWidths=[2.2 * cm] + [1.9 * cm] * 8)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
    ]))
    story.append(t)
    story.append(Paragraph(
        "Label kosong = gambar background (tidak ada kendaraan) dan tetap valid untuk training.", body))

    story.append(Paragraph("2. Hasil validasi (split val)", h2))
    rows = [["Kelas", "Precision", "Recall", "mAP50", "mAP50-95"]]
    rows.append(["SEMUA", f"{info['val']['mp']:.4f}", f"{info['val']['mr']:.4f}",
                 f"{info['val']['map50']:.4f}", f"{info['val']['map']:.4f}"])
    for c in info["val"]["per_class"]:
        rows.append([c["class"], f"{c['p']:.4f}", f"{c['r']:.4f}",
                     f"{c['map50']:.4f}", f"{c['map50_95']:.4f}"])
    t = Table(rows, colWidths=[4 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#dbeafe")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t)

    story.append(Paragraph("3. Inference pada frame CCTV Malang", h2))
    li = info["live"]
    rows = [
        ["Frame dianalisis", li["frames"]],
        ["Frame ada deteksi", li["frames_with_detection"]],
        ["Total kendaraan", li["total"]],
        ["Rata-rata per frame", f"{li['avg_per_frame']:.2f}"],
        ["Kecepatan", f"{li['ms_per_frame']:.0f} ms/frame"],
        ["car", li["per_class"].get("car", 0)],
        ["motorcycle", li["per_class"].get("motorcycle", 0)],
        ["bus", li["per_class"].get("bus", 0)],
        ["truck", li["per_class"].get("truck", 0)],
    ]
    t = Table(rows, colWidths=[6 * cm, 5 * cm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
    ]))
    story.append(t)

    story.append(PageBreak())
    story.append(Paragraph("4. Perbandingan dengan model sebelumnya", h2))
    story.append(Paragraph(
        "Sebelum dipakai produksi, bandingkan model COCO vs model ini pada 200 frame yang sama sebelum dipakai produksi.", body))

    story.append(Paragraph("5. Rekomendasi", h2))
    for txt in info["notes"]:
        story.append(Paragraph("• " + txt, body))

    story.append(Paragraph("6. File terkait", h2))
    for k, v in info["files"].items():
        story.append(Paragraph(f"<b>{k}</b>: {v}", body))

    doc.build(story)


def main():
    t_start = time.time()

    banner("CCTVMAKOTA - FINE-TUNE YOLOv11x PADA CCTV MALANG")

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        print(f" Device : GPU {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
    else:
        print(" Device : CPU (lambat, sebaiknya pakai GPU)")

    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml tidak ada: {DATA_YAML}")
        print("        Jalankan dulu: python prepare_malang_labels.py")
        sys.exit(1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- STEP 1: verifikasi dataset ----------
    banner("[STEP 1] Verifikasi dataset berlabel")
    ds = verify_dataset()
    for split, s in ds.items():
        print(f"  {split:<6} gambar={s['images']:<5} tanpa_label={s['missing_labels']:<4} "
              f"label_kosong={s['empty_labels']:<4} kotak={s['total_boxes']}")
        print(f"         " + " · ".join(f"{NAMES[k]}={v}" for k, v in s["per_class"].items()))

    if ds["train"]["images"] < 20:
        print("[ERROR] data train terlalu sedikit (< 20 gambar).")
        sys.exit(1)
    if ds["train"]["total_boxes"] < 50:
        print("[ERROR] total kotak train < 50. Pre-label gagal?")
        sys.exit(1)
    if ds["train"]["missing_labels"]:
        print(f"[WARN] {ds['train']['missing_labels']} gambar tanpa file label "
              f"(dianggap background oleh YOLO)")

    n_train = ds["train"]["images"]
    freeze = 10 if n_train < 500 else 0
    print(f"\n  Strategi : fine-tune dari {BASE_MODEL}, freeze={freeze} "
          f"({'kepala saja' if freeze else 'semua layer'})")

    # ---------- STEP 2: training ----------
    banner("[STEP 2] Training")
    print(f"  Base    : {BASE_MODEL}")
    print(f"  Epochs  : {EPOCHS} | imgsz {IMGSZ} | batch {BATCH} | patience {PATIENCE}")

    model = YOLO(BASE_MODEL)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=device,
        seed=SEED,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        patience=PATIENCE,
        cos_lr=True,
        close_mosaic=10,
        # pagi/malam/hujan: variasi terang-gelap & warna kuat, blur ringan (titik hujan, lensa basah)
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.6,
        degrees=0.0, translate=0.1, scale=0.5, fliplr=0.5, mixup=0.1,
        freeze=freeze,
        cache=False,
        amp=device != "cpu",
        plots=True,
        save=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    print(f"\n  Run dir : {save_dir}")
    print(f"  best.pt : {best_pt}  ({'ADA' if best_pt.exists() else 'TIDAK ADA'})")

    # ---------- STEP 3: validasi ----------
    banner("[STEP 3] Validasi split val")
    metrics = model.val(
        data=str(DATA_YAML), split="val", imgsz=VAL_IMGSZ, device=device,
        project=str(RUNS_DIR), name=f"{RUN_NAME}_val", exist_ok=True, verbose=False,
    )

    val = {
        "mp": float(metrics.box.mp),
        "mr": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map": float(metrics.box.map),
        "per_class": [],
    }
    names = model.names
    name_list = list(names.values()) if isinstance(names, dict) else list(names)
    for i, cname in enumerate(name_list):
        try:
            val["per_class"].append({
                "class": cname,
                "p": float(metrics.box.p[i]),
                "r": float(metrics.box.r[i]),
                "map50": float(metrics.box.ap50[i]),
                "map50_95": float(metrics.box.ap[i]),
            })
        except Exception:
            pass

    print(f"  Precision : {val['mp']:.4f}")
    print(f"  Recall    : {val['mr']:.4f}")
    print(f"  mAP50     : {val['map50']:.4f}")
    print(f"  mAP50-95  : {val['map']:.4f}")
    for c in val["per_class"]:
        print(f"    {c['class']:<12} P {c['p']:.3f}  R {c['r']:.3f}  "
              f"mAP50 {c['map50']:.3f}  mAP50-95 {c['map50_95']:.3f}")

    # ---------- STEP 4: inference live Malang ----------
    banner("[STEP 4] Inference frame CCTV Malang")
    infer_model = YOLO(str(best_pt)) if best_pt.exists() else model

    frames = []
    for cam in sorted(p for p in LIVE_DIR.glob("*CCTV") if p.is_dir()):
        fs = sorted(cam.glob("*.jpg"))
        if not fs:
            continue
        quota = max(1, round(MAX_LIVE_FRAMES * len(fs) / 699))
        idx = np.linspace(0, len(fs) - 1, min(quota, len(fs))).astype(int)
        frames += [fs[i] for i in sorted(set(idx.tolist()))]
    frames = frames[:MAX_LIVE_FRAMES]
    print(f"  Frame sampel : {len(frames)}")

    per_frame = []
    t1 = time.time()
    for i, fpath in enumerate(frames, 1):
        import cv2
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        r = infer_model.predict(img, conf=LIVE_CONF, iou=LIVE_IOU, imgsz=LIVE_IMGSZ,
                                device=device, verbose=False)[0]
        cls_arr = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.array([], dtype=int)
        conf_arr = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.array([])
        counts = {n: int((cls_arr == k).sum()) for k, n in enumerate(name_list)}
        per_frame.append({
            "frame": fpath.name,
            "camera": fpath.parent.name,
            "total": int(len(cls_arr)),
            **counts,
            "conf_mean": float(conf_arr.mean()) if len(conf_arr) else 0.0,
            "conf_max": float(conf_arr.max()) if len(conf_arr) else 0.0,
        })
        if i % 50 == 0:
            print(f"  {i}/{len(frames)} frame")

    live_dur = time.time() - t1
    live_total = sum(d["total"] for d in per_frame)
    live = {
        "frames": len(per_frame),
        "frames_with_detection": sum(1 for d in per_frame if d["total"] > 0),
        "total": live_total,
        "avg_per_frame": live_total / max(len(per_frame), 1),
        "ms_per_frame": live_dur / max(len(per_frame), 1) * 1000,
        "per_class": {c: sum(d.get(c, 0) for d in per_frame) for c in name_list},
    }
    print(f"  Total kendaraan : {live_total}  ({live['avg_per_frame']:.2f}/frame)")
    print(f"  Kecepatan       : {live['ms_per_frame']:.0f} ms/frame")

    # ---------- STEP 5: CSV ----------
    banner("[STEP 5] CSV report")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORTS_DIR / f"train_malang_live_{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Frame", "Camera", "Total"] + name_list + ["Conf Mean", "Conf Max"])
        for d in per_frame:
            w.writerow([d["frame"], d["camera"], d["total"]]
                       + [d.get(c, 0) for c in name_list]
                       + [f"{d['conf_mean']:.3f}", f"{d['conf_max']:.3f}"])
        w.writerow([])
        w.writerow(["SUMMARY"])
        w.writerow(["Base model", BASE_MODEL])
        w.writerow(["Epochs", EPOCHS])
        w.writerow(["imgsz", IMGSZ])
        w.writerow(["Frames", live["frames"]])
        w.writerow(["Frames with detection", live["frames_with_detection"]])
        w.writerow(["Total vehicles", live_total])
        w.writerow(["Avg per frame", f"{live['avg_per_frame']:.2f}"])
        w.writerow([])
        w.writerow(["VAL METRICS"])
        w.writerow(["Precision", f"{val['mp']:.4f}"])
        w.writerow(["Recall", f"{val['mr']:.4f}"])
        w.writerow(["mAP50", f"{val['map50']:.4f}"])
        w.writerow(["mAP50-95", f"{val['map']:.4f}"])
        w.writerow([])
        w.writerow(["Class", "Precision", "Recall", "mAP50", "mAP50-95"])
        for c in val["per_class"]:
            w.writerow([c["class"], f"{c['p']:.4f}", f"{c['r']:.4f}",
                        f"{c['map50']:.4f}", f"{c['map50_95']:.4f}"])
    print(f"  CSV : {csv_path}")

    # metrik per epoch (dari results.csv ultralytics)
    csv_epoch = None
    res_csv = save_dir / "results.csv"
    if res_csv.exists():
        csv_epoch = REPORTS_DIR / f"train_malang_live_epochs_{stamp}.csv"
        with open(res_csv, "r", encoding="utf-8") as src, \
                open(csv_epoch, "w", newline="", encoding="utf-8") as dst:
            dst.write(src.read().strip() + "\n")
        print(f"  Epoch metrics : {csv_epoch}")

    # ---------- STEP 6: PDF ----------
    banner("[STEP 6] PDF report")
    pdf_path = REPORTS_DIR / f"train_malang_live_report_{stamp}.pdf"

    notes = [
        f"Model dasar COCO {BASE_MODEL} dipakai ulang agar tidak kehilangan kemampuan "
        f"mengenali kendaraan dari domain CCTV.",
        f"Dataset Malang {ds['train']['images']} gambar train / {ds['val']['images']} val. "
        f"Kualitas label sangat menentukan; koreksi manual pada review/index.html masih disarankan.",
        "Kelas truck rawan tertukar dengan MPV/van. Jika masih sering salah, tambahkan "
        "contoh MPV ke kelas car pada label.",
    ]
    info = {
        "base_model": BASE_MODEL,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "stamp": datetime.now().strftime("%d %b %Y %H:%M"),
        "dataset": ds,
        "val": val,
        "live": live,
        "notes": notes,
        "files": {
            "Model": str(best_pt),
            "Run folder": str(save_dir),
            "CSV deteksi": str(csv_path),
            "CSV epoch": str(csv_epoch) if csv_epoch else "-",
        },
    }

    try:
        write_pdf(pdf_path, info)
        print(f"  PDF : {pdf_path}")
    except Exception as e:
        print(f"  [WARN] PDF gagal: {e}")
        pdf_path = None

    # ---------- ringkasan ----------
    dur = time.time() - t_start
    banner("RINGKASAN")
    print(f" Base model      : {BASE_MODEL} (freeze={freeze})")
    print(f" Dataset train   : {ds['train']['images']} gambar / {ds['train']['total_boxes']} kotak")
    print(f" Val P/R/mAP50   : {val['mp']:.3f} / {val['mr']:.3f} / {val['map50']:.3f}")
    print(f" Live kendaraan  : {live_total} ({live['avg_per_frame']:.2f}/frame)")
    print(f" Durasi total    : {dur/60:.1f} menit")
    print("-" * 80)
    print(f" best.pt : {best_pt}")
    print(" Uji banding:")
    print("=" * 80)

    summary = REPORTS_DIR / f"train_malang_live_summary_{stamp}.json"
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump({
            "base_model": BASE_MODEL,
            "freeze": freeze,
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "dataset": ds,
            "val": val,
            "live": live,
            "best_pt": str(best_pt),
            "csv": str(csv_path),
            "pdf": str(pdf_path) if pdf_path else None,
            "duration_minutes": round(dur / 60, 2),
        }, fh, indent=2, ensure_ascii=False)
    print(f" JSON : {summary}")


if __name__ == "__main__":
    main()
