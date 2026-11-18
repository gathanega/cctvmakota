"""
Ukur recall/precision deteksi produksi (vision.detect_vehicles, termasuk tiles & CLAHE) terhadap label val.
Kelas diabaikan (kendaraan = kendaraan), cocok bila IoU >= 0.5.

    python training/eval_recall.py [model] [tiles]      # contoh: runs/.../best.pt 2x3
"""
import glob
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # training/ -> root proyek
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import vision

model = sys.argv[1] if len(sys.argv) > 1 else "runs/yolo11s_malang_live/weights/best.pt"
tiles = sys.argv[2] if len(sys.argv) > 2 else "off"
imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 960
tp = fp = fn = 0
ms = []
files = sorted(glob.glob(str(ROOT / "dataset_malang_labeled/images/val/*.jpg")))
for f in files:
    lab = Path(f.replace("images", "labels")).with_suffix(".txt")
    gt = []
    for line in (lab.read_text().split("\n") if lab.exists() else []):
        p = line.split()
        if len(p) == 5:
            x, y, w, h = map(float, p[1:])
            gt.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    t = time.time()
    r = vision.detect_vehicles(open(f, "rb").read(), options={"model": model, "tiles": tiles, "imgsz": imgsz})
    ms.append((time.time() - t) * 1000)
    used = set()
    for b in r.boxes:
        best, bi = 0.0, -1
        for i, g in enumerate(gt):
            if i not in used:
                v = vision._box_iou(b["box"], g)
                if v > best:
                    best, bi = v, i
        if best >= 0.5:
            used.add(bi)
            tp += 1
        else:
            fp += 1
    fn += len(gt) - len(used)
print(f"{model} tiles={tiles} imgsz={imgsz}: recall {tp / max(1, tp + fn):.3f}  precision {tp / max(1, tp + fp):.3f}  "
      f"kend/frame {(tp + fp) / len(files):.2f}  label/frame {(tp + fn) / len(files):.2f}  {sum(ms[3:]) / max(1, len(ms) - 3):.0f} ms")
