"""
CCTVMAKOTA vision + density logic.

- detect_vehicles(): runs YOLO (yolo11x, see config detect.model) over a JPEG frame, returns per-class
  vehicle counts (car/motorcycle/bus/truck) plus every box (class, conf,
  normalized xyxy, inside-ROI flag) so the dashboard can show/export them.
  Optionally filtered to a region of interest (ROI) polygon representing a single lane, so a wide
  CCTV frame covering multiple directions of traffic can be calibrated to
  count only the lane that actually feeds the traffic light being served
  (e.g. "jalur kanan" / right lane). See app.py's `roi` config field and
  GET /api/debug/roi-preview for calibrating the polygon.
- render_detections(): draws those boxes + ROI on the frame (JPEG output),
  served by GET /api/detections/image.

  conf/iou: conf=0.25 (default ultralytics 0.35) menangkap motor yang sebagian tertutup.
  iou=0.65 (default 0.7) adalah ambang NMS per kelas. Nilai 0.85 yang dulu dipakai
  terbukti meloloskan kotak ganda pada satu motor yang sama (kotak bersarang dengan
  IoU 0.6-0.85), sehingga hitungan membengkak saat model/resolusi dinaikkan. Motor yang
  benar-benar berbeda jarang tumpang tindih > 0.5 pada kamera CCTV biasa. Kotak beda
  kelas pada satu kendaraan ditangani terpisah oleh refine_detections().

- classify_density(): turns a vehicle count into LANCAR / PADAT / MACET.
- density_info(): status + density_percent (count / calibration_max_count) untuk
  Informasi Kepadatan di dashboard. Tidak ada kontrol lampu lalu lintas.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from PIL import Image
from ultralytics import YOLO

# COCO class ids we treat as "vehicles"
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

log = logging.getLogger("greenflow.vision")

# Pengaturan deteksi bawaan. Bisa ditimpa lewat blok `detect:` di config.yaml
# (global) dan/atau di dalam entri lokasi (per kamera). Lihat config.example.yaml.
DEFAULT_DETECT = {
    "model": "models/yolo11s.pt",        # n < s < m < l: makin besar makin akurat, makin lambat di CPU
    "fallback_model": "models/yolov8n.pt",  # dipakai kalau model di atas gagal diunduh/dimuat
    "imgsz": 960,                  # resolusi masukan YOLO; makin besar makin baik untuk objek kecil/jauh
    "conf": 0.25,                  # ambang keyakinan minimum
    "iou": 0.65,                   # NMS per kelas; turunkan (0.5) bila satu kendaraan masih dapat dua kotak
    "cross_class_iou": 0.6,        # kotak beda kelas dengan IoU > ini dianggap kendaraan yang sama
    "enhance": "auto",            # auto = CLAHE saat frame gelap (malam) / kontras rendah (hujan, kabut); "off" = mati
    "tiles": "2x3",               # potong frame jadi baris x kolom (overlap 25%) + frame penuh: kendaraan kecil/jauh ikut terdeteksi; "off" = mati
    "tile_imgsz": 640,
    "heavy_min_conf": 0.45,        # bus/truk di bawah ini dianggap salah kelas -> dihitung sebagai mobil
}

_models: dict[str, "YOLO"] = {}


def get_model(name: str = DEFAULT_DETECT["model"]) -> YOLO:
    """Muat model sekali per nama. File .pt diunduh otomatis saat pertama kali
    (butuh internet sekali). Kalau gagal, jatuh ke `fallback_model` supaya
    deteksi tetap jalan, dan hasilnya diingat agar tidak mengunduh ulang tiap frame."""
    model = _models.get(name)
    if model is None:
        try:
            model = YOLO(name)
        except Exception as e:
            fb = DEFAULT_DETECT["fallback_model"]
            if name == fb:
                raise
            log.error("gagal memuat model %s (%s); memakai %s", name, e, fb)
            model = _models.get(fb) or YOLO(fb)
            _models[fb] = model
        _models[name] = model
    return model


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def vehicle_classes(model) -> dict[int, str]:
    """id kelas kendaraan menurut nama di model: COCO (2,3,5,7) atau model Malang (0-3)."""
    names = getattr(model, "names", None) or {}
    found = {int(i): n for i, n in names.items() if n in VEHICLE_CLASSES.values()}
    return found or dict(VEHICLE_CLASSES)


def tile_windows(h: int, w: int, rows: int, cols: int, overlap: float = 0.25):
    """Jendela (x, y, lebar, tinggi) grid rows x cols yang saling tumpang 25%."""
    th = int(h / (rows - (rows - 1) * overlap)) if rows > 1 else h
    tw = int(w / (cols - (cols - 1) * overlap)) if cols > 1 else w
    for y in np.linspace(0, h - th, rows).astype(int):
        for x in np.linspace(0, w - tw, cols).astype(int):
            yield int(x), int(y), tw, th


def merge_boxes(raw: list[dict], iou: float) -> list[dict]:
    """Gabung hasil frame penuh + potongan: kotak yang tumpang (IoU, atau satu kotak hampir
    menutupi yang lain karena terpotong batas tile) disatukan, ambil yang paling yakin."""
    keep: list[dict] = []
    for d in sorted(raw, key=lambda d: -d["conf"]):
        a = d["xyxy"]
        dup = False
        for k in keep:
            b = k["xyxy"]
            ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1.0
            if _box_iou(a, b) > iou or ix / small > 0.7:
                dup = True
                break
        if not dup:
            keep.append(d)
    return keep


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Malam/hujan: perjelas kontras lokal (CLAHE pada kanal terang) hanya bila frame gelap atau buram.
    Frame siang normal dikembalikan apa adanya supaya hasil siang tidak berubah."""
    import cv2
    gray = frame.mean(axis=2)
    if gray.mean() >= 90 and gray.std() >= 35:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def refine_detections(dets: list[dict], cross_class_iou: float, heavy_min_conf: float) -> list[dict]:
    """Rapikan hasil mentah YOLO. Setiap item: {"cls", "conf", "xyxy": (x1,y1,x2,y2)}.

    1. Satu kendaraan sering terdeteksi dua kali dengan kelas berbeda (mis. 'car' dan
       'truck'), karena NMS YOLO bekerja per kelas. Kotak berkeyakinan lebih rendah yang
       menimpa kotak kelas lain (IoU > cross_class_iou) dibuang, jadi hitungan tidak ganda.
       Kotak sekelas tidak disentuh, supaya motor yang berdempetan tetap terhitung satu-satu.
    2. Bus/truk dengan keyakinan rendah (< heavy_min_conf) hampir selalu mobil yang salah
       kelas. Dihitung sebagai 'car' (jumlah tetap, kelas dikoreksi); kelas asal disimpan
       di 'raw_cls'.
    """
    kept: list[dict] = []
    for d in sorted(dets, key=lambda x: -x["conf"]):
        dup = any(k["cls"] != d["cls"] and _box_iou(k["xyxy"], d["xyxy"]) > cross_class_iou for k in kept)
        if not dup:
            kept.append(dict(d))
    for d in kept:
        if d["cls"] in ("bus", "truck") and d["conf"] < heavy_min_conf:
            d["raw_cls"] = d["cls"]
            d["cls"] = "car"
    return kept


def _point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test. polygon is a list of
    (x, y) points in the SAME units as the point (here: pixels)."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


@dataclass
class DetectionResult:
    vehicle_count: int
    counts_by_class: dict = field(default_factory=dict)
    # Always the WHOLE-FRAME totals, regardless of ROI filtering above, so
    # callers can show "jalur kanan: X / semua jalur: Y" style comparisons.
    total_vehicle_count: int = 0
    total_counts_by_class: dict = field(default_factory=dict)
    # Hasil per kotak (untuk overlay/unduhan di dashboard). Koordinat dinormalisasi 0-1:
    #   {"cls": "car", "conf": 0.93, "box": [x1, y1, x2, y2], "in_roi": True}
    boxes: list = field(default_factory=list)
    width: int = 0
    height: int = 0


def detect_vehicles(
    jpeg_bytes: bytes,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    roi_polygon: Optional[Sequence[tuple[float, float]]] = None,
    options: Optional[dict] = None,
) -> DetectionResult:
    """
    roi_polygon: optional list of (x, y) points in NORMALIZED coordinates
    (each 0.0-1.0, fraction of frame width/height) describing the lane to
    count. Using normalized coordinates means the same polygon config works
    regardless of the camera's actual resolution. A vehicle is counted as
    "inside" the lane if its bounding-box CENTER point falls inside the
    polygon. When roi_polygon is None, every detected vehicle is counted
    (original whole-frame behaviour).

    options: pengaturan deteksi (model, imgsz, conf, iou, cross_class_iou,
    heavy_min_conf); yang tidak diisi memakai DEFAULT_DETECT. `conf`/`iou`
    di atas tetap diterima demi kompatibilitas dan menimpa `options`.

    conf/iou: lihat docstring modul untuk alasan nilai bawaannya.
    """
    opts = {**DEFAULT_DETECT, **{k: v for k, v in (options or {}).items() if v is not None}}
    if conf is not None:
        opts["conf"] = conf
    if iou is not None:
        opts["iou"] = iou

    image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    frame = np.array(image)
    if str(opts.get("enhance", "auto")) == "auto":
        frame = enhance_frame(frame)
    height, width = frame.shape[:2]

    pixel_polygon = None
    if roi_polygon:
        pixel_polygon = [(px * width, py * height) for (px, py) in roi_polygon]

    model = get_model(str(opts["model"]))
    classes = vehicle_classes(model)
    results = model.predict(
        frame,
        conf=float(opts["conf"]),
        iou=float(opts["iou"]),
        imgsz=int(opts["imgsz"]),
        classes=list(classes),   # hanya kelas kendaraan: lebih cepat, tanpa kotak 'person' dll.
        verbose=False,
    )

    raw: list[dict] = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id not in classes:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        raw.append({"cls": classes[cls_id], "conf": float(box.conf[0]), "xyxy": (x1, y1, x2, y2)})
    grid = str(opts.get("tiles") or "off")
    if "x" in grid:
        rows, cols = (int(v) for v in grid.split("x"))
        tiles = list(tile_windows(height, width, rows, cols))
        crops = [frame[y:y + th, x:x + tw] for x, y, tw, th in tiles]
        for (x, y, _, _), r in zip(tiles, model.predict(crops, conf=float(opts["conf"]), iou=float(opts["iou"]),
                                                       imgsz=int(opts.get("tile_imgsz", 640)), classes=list(classes), verbose=False)):
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in classes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    raw.append({"cls": classes[cls_id], "conf": float(box.conf[0]), "xyxy": (x1 + x, y1 + y, x2 + x, y2 + y)})
        raw = merge_boxes(raw, float(opts["iou"]))

    counts_roi: dict[str, int] = {name: 0 for name in VEHICLE_CLASSES.values()}
    counts_all: dict[str, int] = {name: 0 for name in VEHICLE_CLASSES.values()}
    boxes: list[dict] = []

    for d in refine_detections(raw, float(opts["cross_class_iou"]), float(opts["heavy_min_conf"])):
        cls_name = d["cls"]
        counts_all[cls_name] += 1

        x1, y1, x2, y2 = d["xyxy"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        in_roi = pixel_polygon is None or _point_in_polygon(cx, cy, pixel_polygon)
        if in_roi:
            counts_roi[cls_name] += 1
        item = {
            "cls": cls_name,
            "conf": round(d["conf"], 3),
            "box": [round(x1 / width, 4), round(y1 / height, 4), round(x2 / width, 4), round(y2 / height, 4)],
            "in_roi": in_roi,
        }
        if "raw_cls" in d:
            item["raw_cls"] = d["raw_cls"]
        boxes.append(item)

    total_all = sum(counts_all.values())
    total_roi = sum(counts_roi.values())
    used_roi = pixel_polygon is not None

    return DetectionResult(
        vehicle_count=total_roi if used_roi else total_all,
        counts_by_class=counts_roi if used_roi else counts_all,
        total_vehicle_count=total_all,
        total_counts_by_class=counts_all,
        boxes=boxes,
        width=width,
        height=height,
    )


# warna kotak per kelas (RGB), mirip tampilan umum YOLO
CLASS_COLORS = {
    "car": (0, 200, 255),
    "motorcycle": (255, 190, 0),
    "bus": (180, 120, 255),
    "truck": (255, 90, 160),
}


def _font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def render_detections(
    jpeg_bytes: bytes,
    boxes: Sequence[dict],
    roi_polygon: Optional[Sequence[Sequence[float]]] = None,
    roi_label: str = "",
    show_outside: bool = True,
    show_roi: bool = True,
    quality: int = 88,
) -> bytes:
    """Gambar hasil deteksi di atas frame: kotak + 'kelas: conf' per kendaraan,
    poligon ROI, dan ringkasan 'Kendaraan terdeteksi' di pojok kiri atas.
    Kendaraan di luar ROI digambar abu-abu tipis (atau disembunyikan)."""
    from PIL import ImageDraw

    image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    w, h = image.size
    draw = ImageDraw.Draw(image, "RGBA")
    lw = max(2, round(w / 400))
    font = _font(max(11, round(w / 70)))
    big = _font(max(13, round(w / 48)))

    if roi_polygon and show_roi:
        pts = [(x * w, y * h) for x, y in roi_polygon]
        draw.polygon(pts, fill=(34, 197, 94, 40))
        draw.line(pts + [pts[0]], fill=(34, 197, 94, 255), width=lw + 1)
        if roi_label:
            tx, ty = min(pts, key=lambda p: p[1])
            tb = draw.textbbox((0, 0), roi_label, font=font)
            tx = min(max(4, tx - tb[2] / 2), w - tb[2] - 10)
            ty = max(ty, tb[3] + 12)
            draw.rectangle([tx - 4, ty - tb[3] - 10, tx + tb[2] + 6, ty - 2], fill=(11, 59, 32, 220))
            draw.text((tx + 1, ty - tb[3] - 7), roi_label, fill=(255, 255, 255), font=font)

    inside = 0
    for b in boxes:
        x1, y1, x2, y2 = b["box"][0] * w, b["box"][1] * h, b["box"][2] * w, b["box"][3] * h
        if not b.get("in_roi", True):
            if show_outside:
                draw.rectangle([x1, y1, x2, y2], outline=(170, 170, 170, 200), width=max(1, lw - 1))
            continue
        inside += 1
        color = CLASS_COLORS.get(b["cls"], (0, 200, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=lw)
        text = f"{b['cls']}: {b['conf']:.2f}"
        tb = draw.textbbox((0, 0), text, font=font)
        ty = y1 - tb[3] - 4 if y1 - tb[3] - 4 > 0 else y1
        draw.rectangle([x1, ty, x1 + tb[2] + 6, ty + tb[3] + 4], fill=color + (230,))
        draw.text((x1 + 3, ty + 1), text, fill=(0, 0, 0), font=font)

    total = len(boxes)
    head = f"Kendaraan terdeteksi: {inside}" + (f"  (semua jalur: {total})" if roi_polygon else "")
    tb = draw.textbbox((0, 0), head, font=big)
    draw.rectangle([6, 6, tb[2] + 18, tb[3] + 16], fill=(0, 0, 0, 150))
    draw.text((12, 10), head, fill=(60, 255, 120), font=big)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def classify_density(vehicle_count: int, low: int, high: int) -> str:
    """low/high thresholds are per-location calibration (see config.yaml)."""
    if vehicle_count < low:
        return "LANCAR"
    if vehicle_count < high:
        return "PADAT"
    return "MACET"


def density_info(vehicle_count: int, density_low: int = 5, density_high: int = 15,
                 calibration_max_count: int = 20) -> dict:
    """Informasi kepadatan untuk dashboard: status + persentase terhadap kapasitas kalibrasi kamera."""
    return {
        "traffic_status": classify_density(vehicle_count, density_low, density_high),
        "density_percent": round(min(100.0, 100.0 * vehicle_count / max(calibration_max_count, 1)), 1),
    }
