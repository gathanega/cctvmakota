"""
CCTVMAKOTA - ATCS Integrated Command Center
Version 2.0: Decision-support only (no light control, CCTV-only)

Endpoints:
  GET  /                              Dashboard (5 tabs)
  GET  /api/health                    Server health & status
  GET  /api/locations                 All CCTV locations + status
  GET  /api/frame?location=...        Latest frame from camera
  GET  /api/logs?location=&limit=     History from Supabase
  GET  /api/atcs/scenario             ATCS route allocation recommendation
  GET  /api/atcs/decision-log         Logged ATCS decisions
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import requests
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageDraw

try:
    from supabase import Client, create_client
except ImportError:
    Client = None
    create_client = None

import atcs_routing
import output_reader
from frame_queue import FrameQueueManager
import runtime
from vision import DEFAULT_DETECT, _point_in_polygon, classify_density, density_info, detect_vehicles, get_model, render_detections

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cctvmakota")

app = FastAPI(title="CCTVMAKOTA - ATCS Integrated Command Center", version="2.0.0")

# Load config
CONFIG_FILE = os.environ.get("CCTVMAKOTA_CONFIG", "config.cctvmakota.yaml")
with open(CONFIG_FILE, "r", encoding="utf-8-sig") as f:   # -sig: tahan BOM dari editor Windows
    CONFIG = yaml.safe_load(f)

# "${ADMIN_TOKEN}" di config adalah placeholder, bukan token: jangan pernah dipakai apa adanya
if str(CONFIG.get("admin_token") or "").startswith("${"):
    CONFIG["admin_token"] = os.environ.get("ADMIN_TOKEN", "")

# Override admin dari dashboard (ROI, ambang, koordinat, Admin Control rute GTS)
runtime.load()
runtime.apply_to_config(CONFIG)

# Supabase (optional; fail gracefully if key not set)
try:
    if create_client is None:
        raise RuntimeError("supabase package is not installed")
    service_key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    configured_key = CONFIG["supabase"].get("service_key", "")
    if not service_key and not configured_key.startswith("${"):
        service_key = configured_key
    supabase: Client = create_client(CONFIG["supabase"]["url"], service_key) if service_key else None
except Exception as e:
    log.warning(f"Supabase init failed: {e}; proceeding without Supabase")
    supabase = None

# Output reader (decisions traced to output files)
output_path = os.environ.get("CCTVMAKOTA_OUTPUT", "Output")
out_reader = output_reader.OutputReader(output_path)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG.get("cors_origins", ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory caches
LATEST_DENSITY: dict[str, dict] = {}   # lokasi -> status kepadatan terakhir
LATEST_FRAMES: dict[str, tuple[bytes, float]] = {}
LATEST_DETECTIONS: dict[str, dict] = {}
DECISION_LOG: list[dict] = []  # ATCS decision audit trail

# FFMPEG path
FFMPEG_BIN = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"

# Detection lock (YOLO is not thread-safe)
_DETECT_LOCK = threading.Lock()


def location_calibration(location: str) -> dict:
    """Get per-location thresholds."""
    locations = CONFIG.get("locations", {})
    return locations.get(location, {}) or {}


def detect_options(location: str) -> dict:
    """Get detection settings."""
    return {**(CONFIG.get("detect") or {}), **(location_calibration(location).get("detect") or {})}


def grab_frame_ffmpeg(stream_url: str, timeout: int = 20) -> bytes:
    """Extract JPEG from HLS/MJPEG stream using ffmpeg."""
    if not shutil.which(FFMPEG_BIN) and not (FFMPEG_BIN.startswith("/") and os.path.exists(FFMPEG_BIN)):
        raise RuntimeError(f"ffmpeg not found at {FFMPEG_BIN}")
    
    cmd = [FFMPEG_BIN, "-y", "-loglevel", "error", "-i", stream_url, "-frames:v", "1", "-q:v", "2", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0 or not result.stdout:
        stderr = result.stderr.decode(errors="ignore")[:300]
        raise RuntimeError(f"ffmpeg failed: {stderr}")
    return result.stdout


def fetch_cctv_frame(cfg: dict) -> bytes:
    """Fetch one frame from CCTV."""
    url = cfg["cctv_snapshot_url"]
    if cfg.get("cctv_stream_type") in ("hls", "mjpeg"):
        return grab_frame_ffmpeg(url, timeout=cfg.get("cctv_ffmpeg_timeout", CONFIG.get("cctv_ffmpeg_timeout", 20)))
    
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.content


def process_frame(jpeg_bytes: bytes, location: str, device_id: str):
    """Process one frame: detect, classify density, store."""
    calib = location_calibration(location)
    roi_polygon = calib.get("roi")
    
    with _DETECT_LOCK:
        detection = detect_vehicles(jpeg_bytes, roi_polygon=roi_polygon, options=detect_options(location))
    
    density = density_info(
        detection.vehicle_count,
        density_low=calib.get("density_low", 5),
        density_high=calib.get("density_high", 15),
        calibration_max_count=calib.get("calibration_max_count", 20),
    )
    status = density["traffic_status"]

    # ROI kedua = arah berlawanan; dihitung dari kotak yang sama (titik tengah, koordinat 0-1)
    roi2 = calib.get("roi2")
    dir2 = None
    if roi2:
        n2 = sum(1 for b in detection.boxes
                 if _point_in_polygon((b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2, roi2))
        dir2 = {**density_info(n2, calib.get("density_low", 5), calib.get("density_high", 15),
                               calib.get("calibration_max_count", 20)),
                "vehicle_count": n2, "label": calib.get("roi2_label", "") or "arah 2"}

    LATEST_FRAMES[location] = (jpeg_bytes, time.time())
    LATEST_DETECTIONS[location] = {
        "ts": time.time(),
        "device_id": device_id,
        "width": detection.width,
        "height": detection.height,
        "roi": roi_polygon,
        "roi_label": calib.get("roi_label", "") if roi_polygon else "",
        "roi2": roi2,
        "dir2": dir2,
        "vehicle_count": detection.vehicle_count,
        "all_lanes_total": detection.total_vehicle_count,
        "counts_by_class": detection.counts_by_class,
        "total_counts_by_class": detection.total_counts_by_class,
        "traffic_status": status,
        "density_percent": density["density_percent"],
        "boxes": detection.boxes,
        "jpeg": jpeg_bytes,
    }
    
    counts_payload = dict(detection.counts_by_class)
    if roi_polygon:
        counts_payload["_lane_label"] = calib.get("roi_label", "jalur kanan")
        counts_payload["_all_lanes_total"] = detection.total_vehicle_count
        counts_payload["_all_lanes_by_class"] = detection.total_counts_by_class
    
    row = {
        "device_id": device_id,
        "location": location,
        "vehicle_count": detection.vehicle_count,
        "counts_by_class": counts_payload,
        "traffic_status": status,
    }
    
    # Try to save to Supabase (optional); nama tabel lama dipertahankan supaya riwayat tidak hilang
    if supabase:
        try:
            supabase.table("greenflow_logs").insert(row).execute()
        except Exception as e:
            log.warning(f"Supabase insert failed: {e}")
    
    LATEST_DENSITY[location] = {
        "traffic_status": status,
        "density_percent": density["density_percent"],
        "vehicle_count": detection.vehicle_count,
        "all_lanes_total": detection.total_vehicle_count,
        "counts_by_class": detection.counts_by_class,
        "dir2": dir2,
        "updated_at": time.time(),
    }
    
    log.info(f"[{location}] {device_id}: {detection.vehicle_count} veh -> {status} ({density['density_percent']}%)")
    return row, density


# ==================== ENDPOINTS ====================

DASHBOARD_FILE = os.path.join("cctvmakota-dashboard", "dashboard.html")


@app.get("/")
async def dashboard():
    """Serve CCTVMAKOTA dashboard (single file, built by cctvmakota-dashboard/build_single_file.py)."""
    if not os.path.isfile(DASHBOARD_FILE):
        raise HTTPException(404, f"{DASHBOARD_FILE} not found")
    return FileResponse(DASHBOARD_FILE, media_type="text/html", headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
async def health():
    """Health check."""
    return {
        "ok": True,
        "version": "2.0.0",
        "name": "CCTVMAKOTA",
        "locations_active": len(LATEST_DENSITY),
        "supabase": supabase is not None,
        "output_available": os.path.isdir(output_path),
    }


# /api/locations dan /api/frame: lihat admin_api.py


@app.get("/api/atcs/scenario")
async def atcs_scenario(objective: str = "", primary: str = ""):
    """ATCS allocation recommendation (decision-support only, not operationally verified)."""
    return _atcs_scenario({}, objective, primary)


@app.post("/api/atcs/scenario")
async def atcs_scenario_sim(body: dict):
    """Manual scenario: {"counts": {location: vehicle_count}, "objective": "fastest|shortest", "primary": route_id}."""
    counts = body.get("counts") or {}
    if not isinstance(counts, dict):
        raise HTTPException(422, "counts harus objek {lokasi: jumlah}")
    known = CONFIG.get("locations", {})
    try:
        counts = {k: max(0, int(v)) for k, v in counts.items() if k.removesuffix("#2") in known}
    except (TypeError, ValueError):
        raise HTTPException(422, "jumlah kendaraan harus bilangan bulat")
    return _atcs_scenario(counts, str(body.get("objective") or ""), str(body.get("primary", "")))


def node_label(node: str) -> str:
    """Label titik rute: "lokasi" = arah 1 (ROI utama), "lokasi#2" = arah 2 (ROI kedua)."""
    loc = node.removesuffix("#2")
    cfg = CONFIG["locations"].get(loc) or {}
    label = cfg.get("label", loc)
    if node == loc:
        return label
    return f"{label} ({cfg.get('roi2_label') or 'arah 2'})"


def _atcs_scenario(sim_counts: dict, objective: str, primary: str) -> dict:
    admin = runtime.get_routes_admin()
    rules = {**CONFIG.get("atcs", {}).get("activation_rules", {}), **admin["corridor"]}   # opsi koridor admin menang
    objective = objective or rules.get("objective", "fastest")
    if objective not in ("fastest", "shortest"):
        raise HTTPException(422, "objective harus 'fastest' atau 'shortest'")

    # Get current location statuses (simulated counts override live detection)
    location_statuses = {}
    nodes = []   # (id titik, data deteksi terakhir): "lokasi" = arah 1, "lokasi#2" = arah 2 (ROI kedua)
    for base in CONFIG.get("locations", {}):
        prog = LATEST_DENSITY.get(base) or {}
        nodes += [(base, prog), (base + "#2", prog.get("dir2") or {})]
    for loc, prog in nodes:
        if loc in sim_counts:
            calib = location_calibration(loc.removesuffix("#2"))
            n = sim_counts[loc]
            location_statuses[loc] = {
                **density_info(n, calib.get("density_low", 5), calib.get("density_high", 15),
                               calib.get("calibration_max_count", 20)),
                "vehicle_count": n,
                "source": "simulasi",
            }
        else:
            location_statuses[loc] = {
                "traffic_status": prog.get("traffic_status", "LANCAR"),
                "density_percent": prog.get("density_percent", 0),
                "vehicle_count": prog.get("vehicle_count", 0),
                "source": "deteksi" if prog else "belum ada data",
            }
    
    # Compute recommendation
    atcs_cfg = CONFIG.get("atcs", {})
    routes_config = [{**r, **admin["routes"].get(r["id"], {})} for r in runtime.effective_routes(atcs_cfg.get("routes", []))]
    paths = {r["id"]: r.get("path") for r in routes_config}   # likuk jalan untuk peta
    if not routes_config:
        return {"error": "No routes configured in atcs.routes"}
    primary_id = primary or admin["primary_route"] or atcs_cfg.get("primary_route", routes_config[0]["id"])
    # koridor titik-ke-titik: rute utama + daftar alternatifnya (kalau diisi admin); kosong = semua ruas dibandingkan
    by_id = {r["id"]: r for r in routes_config}
    alts = [a for a in (by_id.get(primary_id) or {}).get("alternatives") or [] if a in by_id]
    if alts:
        routes_config = [by_id[primary_id]] + [by_id[a] for a in alts]

    rec = atcs_routing.compute_recommendation(
        routes_config,
        location_statuses,
        activation_threshold_vc=rules.get("vc_threshold", 0.85),
        activation_threshold_status=rules.get("status_threshold", "MACET"),
        primary_route_id=primary_id,
        objective=objective,
    )
    
    # Log decision
    decision_log_entry = {
        "timestamp": time.time(),
        "scenario_id": rec.scenario_id,
        "activated": rec.activated,
        "reason": rec.activation_reason,
        "allocation": rec.allocation_percent,
        "reroute": rec.reroute,
        "recommended_route_id": rec.recommended_route_id,
        "simulated": bool(sim_counts),
        "source_files": {
            "config": "config.cctvmakota.yaml",
            "output": output_path,
        },
    }
    DECISION_LOG.append(decision_log_entry)
    
    return {
        "timestamp": rec.timestamp,
        "scenario_id": rec.scenario_id,
        "activated": rec.activated,
        "activation_reason": rec.activation_reason,
        "routes": [
            {
                "route_id": r.route_id,
                "route_name": r.route_name,
                "capacity": r.capacity_veh_per_hour,
                "current_demand": r.current_demand,
                "vc_ratio": round(r.vc_ratio, 3),
                "traffic_status": r.traffic_status,
                "is_active": r.is_active,
                "distance_km": r.distance_km,
                "baseline_traveltime_minutes": r.baseline_traveltime_minutes,
                "est_traveltime_minutes": r.est_traveltime_minutes,
                "closed": r.closed,
                "status_source": r.status_source,
                "admin_note": admin["routes"].get(r.route_id, {}).get("note", ""),
                "direction": (by_id.get(r.route_id) or {}).get("direction", "dua_arah"),
                "alternatives": (by_id.get(r.route_id) or {}).get("alternatives") or [],
                "path": paths.get(r.route_id) or [],
                "nodes": [
                    {"location": n, "label": node_label(n),
                     "lat": (CONFIG["locations"].get(n.removesuffix("#2")) or {}).get("lat"),
                     "lng": (CONFIG["locations"].get(n.removesuffix("#2")) or {}).get("lng"),
                     **location_statuses.get(n, {})}
                    for n in next(c for c in routes_config if c["id"] == r.route_id).get("nodes", [])
                ],
            }
            for r in rec.routes
        ],
        "objective": objective,
        "primary_route_id": rec.primary_route_id,
        "recommended_route_id": rec.recommended_route_id,
        "fastest_route_id": rec.fastest_route_id,
        "shortest_route_id": rec.shortest_route_id,
        "reroute": rec.reroute,
        "reroute_reason": rec.reroute_reason,
        "thresholds": {"vc": rules.get("vc_threshold", 0.85), "status": rules.get("status_threshold", "MACET")},
        "simulated": bool(sim_counts),
        "allocation_percent": rec.allocation_percent,
        "total_demand": rec.total_demand_veh_per_hour,
        "estimated_system_saving_percent": round(rec.estimated_system_saving_percent, 1),
        "caveat": "MODEL ESTIMATE, NOT OPERATIONALLY VERIFIED",
    }


@app.get("/api/gis/roads")
async def gis_roads(major: int = 1):
    """Jaringan jalan OSM sekitar kamera (GeoJSON). Ruas dekat kamera membawa status kepadatannya."""
    import gis_malang
    layer = gis_malang.road_layer(CONFIG.get("locations", {}), LATEST_DENSITY)
    if major:   # peta ringan (HP): hanya jalan utama + ruas yang dipantau kamera
        layer = {**layer, "features": [f for f in layer["features"]
                                       if f["properties"]["highway"] in gis_malang.MAJOR or f["properties"].get("status")]}
    return JSONResponse(layer, headers={"Cache-Control": "max-age=30"})


@app.get("/api/gis/analysis")
async def gis_analysis():
    """Per kamera: ruas OSM yang cocok (nama, kelas, lajur, kapasitas), jarak pin ke jalan, usulan koordinat."""
    import gis_malang
    return {"source": gis_malang.roads().get("source", ""),
            "locations": gis_malang.analyze(CONFIG.get("locations", {}), LATEST_DENSITY)}


@app.get("/api/demo/macet")
async def demo_macet_api(level: int = 0, objective: str = "fastest"):
    """Demo kemacetan lampu merah A -> B: level 0..100 = beban jalur utama; balikan jalur yang disarankan."""
    import demo_macet
    level = max(0, min(100, int(level)))
    return demo_macet.scenario(level, "shortest" if objective == "shortest" else "fastest")


@app.get("/api/atcs/decision-log")
async def atcs_decision_log(limit: int = 20):
    """ATCS decision audit trail."""
    return {
        "decisions": DECISION_LOG[-limit:],
        "total": len(DECISION_LOG),
    }


@app.on_event("startup")
async def startup():
    """Startup: load model, start CCTV polling."""
    log.info("CCTVMAKOTA v2 starting up...")
    
    # Load detection models
    models = {detect_options(loc).get("model", DEFAULT_DETECT["model"]) for loc in CONFIG.get("locations", {})}
    for name in models or {DEFAULT_DETECT["model"]}:
        try:
            await asyncio.to_thread(get_model, name)
            log.info(f"Model {name} loaded")
        except Exception as e:
            log.error(f"Failed to load model {name}: {e}")

    if os.environ.get("CCTVMAKOTA_POLL", "1") != "0":
        asyncio.create_task(cctv_poll_loop())


async def cctv_poll_loop():
    """Ambil frame tiap kamera CCTV lalu deteksi, agar status rute memakai data live."""
    interval = CONFIG.get("cctv_poll_interval_seconds", 30)
    while True:
        for loc, cfg in CONFIG.get("locations", {}).items():
            cfg = cfg or {}
            if cfg.get("mode") != "cctv":
                continue
            try:
                jpeg = await asyncio.to_thread(fetch_cctv_frame, cfg)
                await asyncio.to_thread(process_frame, jpeg, loc, cfg.get("device_id", loc))
            except Exception as e:
                log.warning(f"[{loc}] poll gagal: {str(e)[:120]}")
        await asyncio.sleep(interval)


# Include admin router (optional)
from types import SimpleNamespace
try:
    from admin_api import create_router
    app.include_router(create_router(SimpleNamespace(
        CONFIG=CONFIG,
        CONFIG_FILE=CONFIG_FILE,
        supabase=supabase,
        fetch_cctv_frame=fetch_cctv_frame,
        LATEST_DENSITY=LATEST_DENSITY,
        LATEST_FRAMES=LATEST_FRAMES,
        LATEST_DETECTIONS=LATEST_DETECTIONS,
        render_detections=render_detections,
    )))
except ImportError:
    log.warning("admin_api not available; skipping admin router")


log.info(f"CCTVMAKOTA v2.0 initialized")
log.info(f"Config: {CONFIG_FILE}")
log.info(f"Locations: {len(CONFIG.get('locations', {}))}")
log.info(f"ATCS routes: {len(CONFIG.get('atcs', {}).get('routes', []))}")
