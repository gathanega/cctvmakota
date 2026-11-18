"""
CCTVMAKOTA - Tambah frame kondisi khusus (malam / hujan) ke dataset_malang_labeled
================================================================================
1. Ambil frame dari server yang sedang jalan (GET /api/frame) tiap N detik, semua kamera
2. Pre-label pakai guru yolo11x.pt COCO @1280 (frame gelap/buram dipertajam CLAHE dulu)
3. Simpan sebagai <tag>_<lokasi>_<waktu>.jpg ke images/{train,val} (1 dari 6 ke val) + label YOLO

Label tetap PRE-LABEL (perlu dikoreksi manual untuk hasil terbaik). Frame tanpa kendaraan dilewati.

Pemakaian:
    python training/add_condition_frames.py malam 20 30      # tag, menit, jeda detik
    python training/add_condition_frames.py hujan 15 30
"""

import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

import numpy as np
import yaml
from PIL import Image
import io

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent   # training/ -> root proyek
sys.path.insert(0, str(ROOT))
DS = ROOT / "dataset_malang_labeled"
SERVER = "http://127.0.0.1:8011"
COCO_TO_DS = {2: 0, 3: 1, 5: 2, 7: 3}   # car, motorcycle, bus, truck


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "malam"
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 20
    gap = float(sys.argv[3]) if len(sys.argv) > 3 else 30
    from ultralytics import YOLO
    from vision import enhance_frame
    teacher = YOLO(str(ROOT / "models" / "yolo11x.pt"))
    locs = list((yaml.safe_load(open(ROOT / "config.cctvmakota.yaml", encoding="utf-8-sig"))["locations"]))
    seen, saved, n = set(), 0, 0
    end = time.time() + minutes * 60
    while time.time() < end:
        for loc in locs:
            try:
                jpeg = urllib.request.urlopen(f"{SERVER}/api/frame?location={quote(loc)}", timeout=40).read()
            except Exception as e:
                print(f"  {loc}: gagal ({e})")
                continue
            if hash(jpeg) in seen:      # frame cache belum berganti
                continue
            seen.add(hash(jpeg))
            frame = enhance_frame(np.array(Image.open(io.BytesIO(jpeg)).convert("RGB")))
            r = teacher.predict(frame, imgsz=1280, conf=0.25, iou=0.6, classes=list(COCO_TO_DS), verbose=False)[0]
            lines = [f"{COCO_TO_DS[int(c)]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                     for c, (x, y, w, h) in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist())]
            if not lines:
                continue
            split = "val" if n % 6 == 5 else "train"
            n += 1
            stem = f"{tag}_{loc}_{time.strftime('%Y%m%d_%H%M%S')}"
            (DS / "images" / split / f"{stem}.jpg").write_bytes(jpeg)
            (DS / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            saved += 1
            print(f"  [{saved}] {split} {stem}: {len(lines)} kotak, terang {frame.mean():.0f}")
        time.sleep(gap)
    print(f"selesai: {saved} frame '{tag}' ditambahkan")


if __name__ == "__main__":
    main()
