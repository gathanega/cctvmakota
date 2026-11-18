"""
CCTVMAKOTA - Smart Frame Collection
- 20 detik delay per frame (lebih stabil)
- Skip frame yang sudah exist
- Resume dari terakhir yang dikumpulkan
"""

import os
import sys
import shutil
import subprocess
import time
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

import yaml

ROOT = Path(__file__).resolve().parent.parent   # training/ -> root proyek
sys.path.insert(0, str(ROOT))
DATASET_DIR = ROOT / "dataset_malang_live"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# CCTV URLs
with open(ROOT / "config.cctvmakota.yaml") as f:
    config = yaml.safe_load(f)
    CCTV_URLS = {
        name: cfg["cctv_snapshot_url"]
        for name, cfg in config.get("locations", {}).items()
    }

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

print("""
==============================================================
   CCTVMAKOTA - Smart Frame Collection
   20 detik delay | Skip existing | Resume dari terakhir
==============================================================
""")

# Target
TARGET_TOTAL = 2000
FRAMES_PER_CAMERA = TARGET_TOTAL // len(CCTV_URLS)
INTERVAL_SECONDS = 5  # 5 detik per frame

print(f"\nKonfigurasi:")
print(f"  Target: {TARGET_TOTAL} total frames")
print(f"  Per camera: {FRAMES_PER_CAMERA} frames")
print(f"  Delay: {INTERVAL_SECONDS} detik per frame")
print(f"  Jumlah camera: {len(CCTV_URLS)}")
print(f"  Est. durasi: {(FRAMES_PER_CAMERA * INTERVAL_SECONDS * len(CCTV_URLS) / 3600):.1f} jam (~{(FRAMES_PER_CAMERA * INTERVAL_SECONDS * len(CCTV_URLS) / 60):.0f} menit)\n")

def capture_frame_ffmpeg(url, output_path, timeout=30):
    """Capture frame dari HLS stream"""
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", url,
        "-frames:v", "1",
        "-q:v", "2",
        "-f", "image2",
        str(output_path)
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=timeout, check=True)
        if output_path.exists():
            return True
        return False
    except Exception as e:
        return False

# Smart collection - skip existing
frame_counts = defaultdict(int)
collection_start = datetime.now()

print("[CAPTURING FRAMES - SKIP EXISTING]")
print("=" * 70)

for location, url in CCTV_URLS.items():
    location_dir = DATASET_DIR / location
    location_dir.mkdir(parents=True, exist_ok=True)
    
    # Count existing frames
    existing_frames = list(location_dir.glob("frame_*.jpg"))
    existing_count = len(existing_frames)
    
    print(f"\n[{location}]")
    print(f"  Existing: {existing_count}")
    print(f"  Target: {FRAMES_PER_CAMERA}")
    print(f"  Need: {max(0, FRAMES_PER_CAMERA - existing_count)}")
    print("  Progress: ", end="", flush=True)
    
    captured = existing_count
    start_num = existing_count
    
    # Only capture new frames
    for frame_num in range(start_num, FRAMES_PER_CAMERA):
        output_file = location_dir / f"frame_{frame_num:04d}.jpg"
        
        # Skip if already exists
        if output_file.exists():
            captured += 1
            frame_num_local = frame_num - start_num + 1
            if frame_num_local % 20 == 0:
                print(f"{captured}", end=" ", flush=True)
            continue
        
        # Capture new frame
        success = capture_frame_ffmpeg(url, output_file)
        
        if success:
            captured += 1
            frame_num_local = frame_num - start_num + 1
            if frame_num_local % 20 == 0:
                print(f"{captured}", end=" ", flush=True)
        
        # Wait 20 detik sebelum frame berikutnya
        if frame_num < FRAMES_PER_CAMERA - 1:
            time.sleep(INTERVAL_SECONDS)
    
    frame_counts[location] = captured
    print(f"\n  Selesai: {captured}/{FRAMES_PER_CAMERA} ✓")

collection_end = datetime.now()
total_frames = sum(frame_counts.values())

print("\n" + "=" * 70)
print("COLLECTION SUMMARY")
print("=" * 70)

for location, count in sorted(frame_counts.items()):
    pct = (count / FRAMES_PER_CAMERA) * 100
    print(f"  {location}: {count}/{FRAMES_PER_CAMERA} ({pct:.0f}%)")

print(f"\nTotal frames: {total_frames}/{TARGET_TOTAL} ({(total_frames/TARGET_TOTAL)*100:.1f}%)")
print(f"Time spent: {(collection_end - collection_start).total_seconds() / 60:.1f} minutes")

# Save metadata
metadata = {
    "timestamp": datetime.now().isoformat(),
    "total_frames": total_frames,
    "target_frames": TARGET_TOTAL,
    "frames_per_camera": dict(frame_counts),
    "locations": list(CCTV_URLS.keys()),
    "dataset_path": str(DATASET_DIR),
    "collection_duration_seconds": (collection_end - collection_start).total_seconds(),
    "interval_seconds": INTERVAL_SECONDS,
}

metadata_file = DATASET_DIR / "metadata.json"
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nMetadata saved: {metadata_file}")
print(f"Frames saved to: {DATASET_DIR}")

if total_frames >= TARGET_TOTAL:
    print(f"\n✅ SUCCESS: Collected {total_frames} frames (100% target)")
elif total_frames >= 1500:
    print(f"\n✅ GOOD: Collected {total_frames} frames (>75% target)")
elif total_frames >= 1000:
    print(f"\n⚠️  OK: Collected {total_frames} frames (>50% target)")
    print(f"   Ready for training, tapi lebih banyak data lebih baik")
else:
    print(f"\n⚠️  WARNING: Only {total_frames} frames collected (<50% target)")
    print(f"   Lanjutkan collection atau mulai training dengan data yang ada")

# Ask to continue if not complete
if total_frames < TARGET_TOTAL:
    print(f"\n[?] Lanjutkan collection? (run script lagi untuk resume)")
    print(f"[?] Atau mulai training dengan {total_frames} frames? (run train_yolo11x_malang.py)")
