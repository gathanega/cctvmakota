"""
Endpoint tambahan untuk dashboard CCTVMAKOTA (satu file HTML). Data live lewat HTTP polling.

Publik (hanya baca):
  GET  /api/public-config                  opsi admin + zona waktu
  GET  /api/locations                      daftar lokasi + koordinat + ROI + informasi kepadatan terakhir
  GET  /api/logs?location=&limit=          riwayat deteksi (dari Supabase)
  GET  /api/hourly?location=&range=        rata-rata kendaraan per jam (today|yesterday|7days)
  GET  /api/frame?location=                frame JPEG terbaru (untuk pantauan & editor ROI)
  GET  /api/detections?location=           hasil deteksi terakhir: kotak {cls, conf, box[x1,y1,x2,y2] 0-1, in_roi} + jumlah
  GET  /api/detections/image?location=&raw=0&outside=1&roi=1&download=0
                                           JPEG beranotasi (kotak + label + "Vehicles detected: N"); raw=1 frame asli
  GET  /api/roi?location=                  polygon ROI tersimpan

Admin (header  Authorization: Bearer <admin_token>):
  POST /api/admin/login
  GET  /api/admin/settings?location=
  PUT  /api/admin/settings   {location, section, data}   section: location|roi|roi2|thresholds
                             roi = arah 1 (hitung kepadatan lokasi), roi2 = arah 2 (opsional, titik rute "lokasi#2")
  GET  /api/admin/routes                                  Admin Control rute GTS
  PUT  /api/admin/routes     {route_id?, closed, status_override, note, primary_route?}
  PUT  /api/admin/route-list {id, name, nodes[], capacity_veh_per_hour, baseline_traveltime_minutes, distance_km,
                              path[[lat,lng]], direction, alternatives[]}
  POST /api/admin/export-config                        tulis setelan admin ke file yaml
  DELETE /api/admin/route-list?id=                         hapus ruas jalan
"""

from __future__ import annotations

import asyncio
import hmac
import os
import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request, Response

import runtime


def _parse_ts(value: str) -> datetime:
    """Supabase kadang mengirim pecahan detik 5 digit; Python < 3.11 menolaknya."""
    s = value.replace("Z", "+00:00")
    s = re.sub(r"\.(\d+)", lambda m: "." + (m.group(1) + "000000")[:6], s, count=1)
    return datetime.fromisoformat(s)


OFFLINE = "Riwayat belum aktif (Supabase belum dikonfigurasi)"


def create_router(ctx) -> APIRouter:
    """ctx: objek dengan CONFIG, supabase, fetch_cctv_frame, LATEST_DENSITY, LATEST_FRAMES,
    LATEST_DETECTIONS, render_detections."""
    router = APIRouter()
    cfg_all = lambda: ctx.CONFIG.setdefault("locations", {})   # noqa: E731

    # ------------------------------------------------------------- helpers
    def slug_of(loc: str) -> str:
        return (cfg_all().get(loc) or {}).get("slug") or runtime.slugify(loc)

    def loc_or_404(loc: str) -> str:
        if not loc or loc not in cfg_all():
            raise HTTPException(404, f"lokasi '{loc}' tidak ada di config.yaml")
        return loc

    def require_admin(request: Request) -> None:
        token = str(ctx.CONFIG.get("admin_token") or "").strip()
        if not token:
            raise HTTPException(503, "admin_token belum diisi di config.yaml, fitur admin dimatikan")
        header = request.headers.get("authorization", "")
        given = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(given.encode(), token.encode()):
            raise HTTPException(401, "Token admin salah")

    def tz() -> ZoneInfo:
        return ZoneInfo(ctx.CONFIG.get("timezone", "Asia/Jakarta"))

    def full_settings(loc: str) -> dict:
        s = runtime.get_location(loc)
        cfg = cfg_all()[loc] or {}
        return {
            **s,
            "location": loc,
            "slug": slug_of(loc),
            "mode": cfg.get("mode", "cctv"),
            "has_roi": bool(s["roi"]),
        }

    # ------------------------------------------------------------- publik
    @router.get("/api/public-config")
    async def public_config():
        return {
            "admin_enabled": bool(str(ctx.CONFIG.get("admin_token") or "").strip()),
            "timezone": ctx.CONFIG.get("timezone", "Asia/Jakarta"),
            "poll_interval_seconds": ctx.CONFIG.get("cctv_poll_interval_seconds", 30),
        }

    @router.get("/api/locations")
    async def locations():
        out = []
        for loc, cfg in cfg_all().items():
            cfg = cfg or {}
            s = runtime.get_location(loc)
            latest = ctx.LATEST_DENSITY.get(loc) or {}
            out.append({
                "location": loc,
                "slug": slug_of(loc),
                "mode": cfg.get("mode", "cctv"),
                "device_id": cfg.get("device_id"),
                "label": s["label"] or loc,
                "lat": s["lat"],
                "lng": s["lng"],
                "has_roi": bool(s["roi"]),
                "roi": s["roi"],
                "roi_label": s["roi_label"],
                "roi2": s["roi2"],
                "roi2_label": s["roi2_label"],
                "stream_url": cfg.get("cctv_snapshot_url") if cfg.get("mode") == "cctv" else None,
                "stream_type": cfg.get("cctv_stream_type", "snapshot") if cfg.get("mode") == "cctv" else None,
                "thresholds": s["thresholds"],
                "traffic_status": latest.get("traffic_status"),
                "density_percent": latest.get("density_percent"),
                "vehicle_count": latest.get("vehicle_count"),
                "all_lanes_total": latest.get("all_lanes_total"),
                "counts_by_class": latest.get("counts_by_class"),
                "updated_at": latest.get("updated_at"),
                "dir2": latest.get("dir2"),
                "detected_at": (ctx.LATEST_DETECTIONS.get(loc) or {}).get("ts"),
            })
        return {"locations": out}

    @router.get("/api/logs")
    async def logs(request: Request):
        if ctx.supabase is None:
            raise HTTPException(503, OFFLINE)
        q = request.query_params
        try:
            limit = max(1, min(int(q.get("limit", 20)), 200))
        except ValueError:
            raise HTTPException(400, "limit harus angka")
        location = q.get("location") or ""

        def run():
            query = ctx.supabase.table("greenflow_logs").select("*").order("created_at", desc=True).limit(limit)
            if location:
                query = query.eq("location", location)
            return query.execute().data or []

        try:
            return {"rows": await asyncio.to_thread(run)}
        except Exception as e:
            raise HTTPException(502, f"gagal membaca Supabase: {e}")

    @router.get("/api/hourly")
    async def hourly(request: Request):
        if ctx.supabase is None:
            raise HTTPException(503, OFFLINE)
        q = request.query_params
        rng = q.get("range", "today")
        if rng not in ("today", "yesterday", "7days"):
            rng = "today"
        location = q.get("location") or ""

        zone = tz()
        today = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0)
        if rng == "yesterday":
            start, end = today - timedelta(days=1), today
        elif rng == "7days":
            start, end = today - timedelta(days=6), today + timedelta(days=1)
        else:
            start, end = today, today + timedelta(days=1)

        def run():
            rows, offset, page = [], 0, 1000      # PostgREST membatasi 1000 baris per permintaan
            while offset < 50000:
                query = (ctx.supabase.table("greenflow_logs")
                         .select("created_at,vehicle_count")
                         .gte("created_at", start.astimezone(ZoneInfo("UTC")).isoformat())
                         .lt("created_at", end.astimezone(ZoneInfo("UTC")).isoformat())
                         .order("created_at")
                         .range(offset, offset + page - 1))
                if location:
                    query = query.eq("location", location)
                data = query.execute().data or []
                rows.extend(data)
                if len(data) < page:
                    break
                offset += page
            return rows

        try:
            rows = await asyncio.to_thread(run)
        except Exception as e:
            raise HTTPException(502, f"gagal membaca Supabase: {e}")

        total, count = [0] * 24, [0] * 24
        for r in rows:
            h = _parse_ts(r["created_at"]).astimezone(zone).hour
            total[h] += int(r.get("vehicle_count") or 0)
            count[h] += 1
        values = [round(total[h] / count[h], 1) if count[h] else 0 for h in range(24)]
        peak = max(values)
        return {
            "range": rng,
            "labels": [f"{h:02d}" for h in range(24)],
            "values": values,
            "peak_hour": values.index(peak) if peak > 0 else None,
            "peak_value": peak,
            "samples": len(rows),
        }

    @router.get("/api/frame")
    async def frame(request: Request):
        loc = loc_or_404(request.query_params.get("location", ""))
        cached = ctx.LATEST_FRAMES.get(loc)
        if cached and time.time() - cached[1] < 120:
            return Response(content=cached[0], media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        cfg = cfg_all()[loc] or {}
        if cfg.get("mode") == "cctv":
            try:
                jpeg = await asyncio.to_thread(ctx.fetch_cctv_frame, cfg)
            except Exception as e:
                if cached:
                    return Response(content=cached[0], media_type="image/jpeg", headers={"Cache-Control": "no-store"})
                raise HTTPException(502, f"gagal ambil frame CCTV: {e}")
            ctx.LATEST_FRAMES[loc] = (jpeg, time.time())
            return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        if cached:
            return Response(content=cached[0], media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        raise HTTPException(404, "belum ada frame untuk lokasi ini")

    def detection_or_404(loc: str) -> dict:
        det = ctx.LATEST_DETECTIONS.get(loc)
        if not det:
            raise HTTPException(404, "belum ada hasil deteksi untuk lokasi ini")
        return det

    def _flag(q, name: str, default: str) -> bool:
        return (q.get(name) or default).lower() in ("1", "true", "yes", "on")

    @router.get("/api/detections")
    async def detections(request: Request):
        loc = loc_or_404(request.query_params.get("location", ""))
        det = detection_or_404(loc)
        out = {k: v for k, v in det.items() if k != "jpeg"}
        out["location"] = loc
        out["age_seconds"] = round(time.time() - det["ts"], 1)
        return out

    @router.get("/api/detections/image")
    async def detections_image(request: Request):
        q = request.query_params
        loc = loc_or_404(q.get("location", ""))
        det = detection_or_404(loc)
        if _flag(q, "raw", "0"):
            body = det["jpeg"]
        else:
            body = await asyncio.to_thread(
                ctx.render_detections, det["jpeg"], det["boxes"], det.get("roi"), det.get("roi_label", ""),
                _flag(q, "outside", "1"), _flag(q, "roi", "1"),
            )
        headers = {"Cache-Control": "no-store", "X-Detected-At": str(det["ts"])}
        if _flag(q, "download", "0"):
            stamp = datetime.fromtimestamp(det["ts"]).strftime("%Y%m%d-%H%M%S")
            kind = "frame" if _flag(q, "raw", "0") else "roi"
            headers["Content-Disposition"] = f'attachment; filename="{slug_of(loc)}-{kind}-{stamp}.jpg"'
        return Response(content=body, media_type="image/jpeg", headers=headers)

    @router.get("/api/roi")
    async def get_roi(request: Request):
        loc = loc_or_404(request.query_params.get("location", ""))
        s = runtime.get_location(loc)
        return {"roi": s["roi"], "label": s["roi_label"]}

    # -------------------------------------------------------------- admin
    @router.post("/api/admin/login")
    async def login(request: Request):
        try:
            require_admin(request)
        except HTTPException as e:
            if e.status_code == 401:
                await asyncio.sleep(0.4)     # perlambat tebak-tebakan token
            raise
        return {"ok": True}

    @router.get("/api/admin/settings")
    async def get_settings(request: Request):
        require_admin(request)
        loc = loc_or_404(request.query_params.get("location", ""))
        return full_settings(loc)

    @router.put("/api/admin/settings")
    async def put_settings(request: Request):
        require_admin(request)
        body = await request.json()
        loc = loc_or_404(body.get("location", ""))
        section = body.get("section", "")
        try:
            runtime.update(loc, section, body.get("data") or {})
        except runtime.ValidationError as e:
            raise HTTPException(422, str(e))

        runtime.apply_to_config(ctx.CONFIG, only=loc)    # ROI/ambang langsung dipakai frame berikutnya
        return {"ok": True, "settings": full_settings(loc)}

    @router.post("/api/admin/export-config")
    async def export_config(request: Request):
        """Tulis setelan admin (ROI, koordinat, ambang, ruas jalan, opsi koridor) ke file config yaml."""
        require_admin(request)
        try:
            path = runtime.export_config(ctx.CONFIG_FILE)
        except (OSError, runtime.ValidationError) as e:
            raise HTTPException(500, f"gagal menulis config: {e}")
        return {"ok": True, "file": os.path.basename(path), "backup": os.path.basename(path) + ".bak"}

    # ------------------------------------------- Admin Control rute GTS
    def base_routes() -> list:
        return (ctx.CONFIG.get("atcs") or {}).get("routes", [])

    def route_ids() -> list:
        return [r["id"] for r in runtime.effective_routes(base_routes())]

    @router.get("/api/admin/routes")
    async def get_routes_admin(request: Request):
        require_admin(request)
        state = runtime.get_routes_admin()
        state["primary_route"] = state["primary_route"] or (ctx.CONFIG.get("atcs") or {}).get("primary_route", "")
        state["route_ids"] = route_ids()
        state["route_list"] = runtime.effective_routes(base_routes())
        rules = (ctx.CONFIG.get("atcs") or {}).get("activation_rules", {})
        state["corridor"] = {"objective": "fastest", "vc_threshold": rules.get("vc_threshold", 0.85),
                             "status_threshold": rules.get("status_threshold", "MACET"), **state["corridor"]}
        return state

    @router.put("/api/admin/route-list")
    async def put_route_list(request: Request):
        """Tambah/ubah ruas jalan: {id, name, nodes[], capacity_veh_per_hour, baseline_traveltime_minutes, distance_km}"""
        require_admin(request)
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(400, "body harus JSON")
        try:
            return {"ok": True, "route": runtime.save_route(cfg_all().keys(), body)}
        except runtime.ValidationError as e:
            raise HTTPException(422, str(e))

    @router.delete("/api/admin/route-list")
    async def delete_route_list(request: Request):
        require_admin(request)
        try:
            runtime.delete_route(base_routes(), request.query_params.get("id", ""))
        except runtime.ValidationError as e:
            raise HTTPException(422, str(e))
        return {"ok": True}

    @router.put("/api/admin/routes")
    async def put_routes_admin(request: Request):
        require_admin(request)
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(400, "body harus JSON")
        try:
            return {"ok": True, **runtime.update_route_admin(route_ids(), body)}
        except runtime.ValidationError as e:
            raise HTTPException(422, str(e))

    return router
