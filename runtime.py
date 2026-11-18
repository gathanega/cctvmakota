"""
Pengaturan runtime yang bisa diubah dari dashboard admin (tanpa edit config.yaml).

Disimpan di runtime_state.json (override di atas config.yaml) dan dipakai oleh:
  - app.py        -> ROI, ambang kepadatan, koordinat, Admin Control rute GTS
  - HTTP          -> konfigurasi kamera (GET /api/device/config)

Modul ini sengaja murni Python (tanpa FastAPI) supaya mudah diuji.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import threading
import time
from typing import Any, Optional

STATE_FILE = os.environ.get("GF_RUNTIME_FILE", "runtime_state.json")
_LOCK = threading.RLock()
_STATE: dict = {"locations": {}}
# CONFIG["locations"] dari config.yaml: nilai awal untuk lokasi yang belum diubah dari dashboard
_BASE: dict = {}

TRAFFIC_STATUSES = ("LANCAR", "PADAT", "MACET")

DEFAULTS: dict[str, dict] = {
    "thresholds": {"density_low": 5, "density_high": 15, "calibration_max_count": 20},
}


class ValidationError(ValueError):
    pass


# ----------------------------------------------------------------- util ----

def slugify(text: str) -> str:
    """Slug lokasi untuk URL/nama file."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug or "default"


def _num(value: Any, lo: float, hi: float, name: str, integer: bool = True):
    try:
        n = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} harus berupa angka")
    if integer:
        n = int(round(n))
    if not lo <= n <= hi:
        raise ValidationError(f"{name} harus antara {lo:g} dan {hi:g}")
    return n


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "ya", "on")
    return bool(value)


# ------------------------------------------------------------ validators ----

def clean_thresholds(data: dict, current: dict) -> dict:
    d = {**current, **{k: v for k, v in data.items() if k in DEFAULTS["thresholds"]}}
    low = _num(d["density_low"], 1, 500, "density_low")
    high = _num(d["density_high"], 1, 1000, "density_high")
    if high <= low:
        raise ValidationError("density_high harus lebih besar dari density_low")
    return {
        "density_low": low,
        "density_high": high,
        "calibration_max_count": _num(d["calibration_max_count"], 1, 1000, "calibration_max_count"),
    }


def clean_location(data: dict) -> dict:
    out: dict = {}
    if "label" in data:
        out["label"] = str(data["label"] or "").strip()[:80]
    if "lat" in data or "lng" in data:
        lat, lng = data.get("lat"), data.get("lng")
        if lat in (None, "") and lng in (None, ""):
            out["lat"] = out["lng"] = None            # hapus titik koordinat
        elif lat in (None, "") or lng in (None, ""):
            raise ValidationError("lat dan lng harus diisi bersamaan")
        else:
            out["lat"] = round(_num(lat, -90, 90, "lat", integer=False), 6)
            out["lng"] = round(_num(lng, -180, 180, "lng", integer=False), 6)
    return out


def _area(points: list) -> float:
    s = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def clean_roi(data: dict, key: str = "roi") -> dict:
    """key "roi" = arah 1 (dipakai hitung kepadatan lokasi), "roi2" = arah 2 (opsional, untuk rute custom)."""
    pts = data.get(key)
    out = {f"{key}_label": str(data.get(f"{key}_label", "") or "").strip()[:80]}
    if pts in (None, [], ""):
        out[key] = None          # ROI dimatikan -> arah 1: hitung seluruh frame; arah 2: tidak dipakai
        return out
    if not isinstance(pts, list) or not 3 <= len(pts) <= 40:
        raise ValidationError("ROI butuh 3 sampai 40 titik")
    clean = []
    for p in pts:
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            raise ValidationError("setiap titik ROI harus [x, y]")
        clean.append([round(_num(p[0], 0, 1, "x ROI", integer=False), 4),
                      round(_num(p[1], 0, 1, "y ROI", integer=False), 4)])
    if _area(clean) < 0.002:
        raise ValidationError("area ROI terlalu kecil atau titiknya segaris")
    out[key] = clean
    return out


# ----------------------------------------------------------------- store ----

def load(path: Optional[str] = None) -> None:
    global _STATE
    path = path or STATE_FILE
    with _LOCK:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _STATE = data if isinstance(data.get("locations"), dict) else {"locations": {}}
        except FileNotFoundError:
            _STATE = {"locations": {}}


def _save(path: Optional[str] = None) -> None:
    path = path or STATE_FILE
    folder = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".runtime_", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(_STATE, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)   # atomik: file tidak pernah setengah tertulis


def get_location(loc: str) -> dict:
    """Pengaturan satu lokasi, dengan nilai default untuk yang belum diubah."""
    with _LOCK:
        saved = _STATE["locations"].get(loc, {})
        base = _BASE.get(loc) or {}

        def pick(key, default=None):
            # kunci yang ada di runtime_state (termasuk None = sengaja dihapus) menang atas config.yaml
            return copy.deepcopy(saved[key] if key in saved else base.get(key, default))

        out = {
            "lat": pick("lat"),
            "lng": pick("lng"),
            "label": pick("label", "") or "",
            "roi": pick("roi"),
            "roi_label": pick("roi_label", "") or "",
            "roi2": pick("roi2"),
            "roi2_label": pick("roi2_label", "") or "",
        }
        thresholds = copy.deepcopy(DEFAULTS["thresholds"])
        thresholds.update({k: base[k] for k in DEFAULTS["thresholds"] if k in base})
        thresholds.update(saved.get("thresholds", {}))
        out["thresholds"] = thresholds
        return out


def update(loc: str, section: str, data: dict, path: Optional[str] = None) -> dict:
    """Validasi lalu simpan satu section. Melempar ValidationError kalau salah."""
    if not isinstance(data, dict):
        raise ValidationError("data harus berupa objek")
    with _LOCK:
        current = get_location(loc)
        new_entry = copy.deepcopy(_STATE["locations"].get(loc, {}))
        if section == "thresholds":
            new_entry["thresholds"] = clean_thresholds(data, current["thresholds"])
        elif section == "location":
            new_entry.update(clean_location(data))
        elif section in ("roi", "roi2"):
            new_entry.update(clean_roi(data, section))
        else:
            raise ValidationError(f"section '{section}' tidak dikenal")
        _STATE["locations"][loc] = new_entry      # baru disimpan setelah semua validasi lolos
        _save(path)
        return get_location(loc)


def apply_to_config(config: dict, only: Optional[str] = None) -> None:
    """Salin override ke CONFIG['locations'] supaya kode lama (app.py) membacanya."""
    global _BASE
    locations = config.setdefault("locations", {})
    _BASE = locations
    _merge_locations(locations, only)


def _merge_locations(locations: dict, only: Optional[str] = None) -> None:
    for loc in list(locations):
        if only and loc != only:
            continue
        saved = _STATE["locations"].get(loc)
        if not saved:
            continue
        cfg = locations[loc] = locations[loc] or {}
        s = get_location(loc)
        for key in ("roi", "roi2"):
            if key in saved:
                if s[key] is None:
                    cfg.pop(key, None)
                else:
                    cfg[key] = s[key]
                if s[f"{key}_label"]:
                    cfg[f"{key}_label"] = s[f"{key}_label"]
        if "thresholds" in saved:
            cfg.update(s["thresholds"])
        for key in ("lat", "lng", "label"):
            if key in saved:
                cfg[key] = saved[key]


def export_config(config_path: str) -> str:
    """Tulis semua setelan admin (ROI 1/2, koordinat, ambang, ruas jalan, opsi koridor) ke file yaml.
    Dibaca dari file mentah (bukan CONFIG di memori) supaya placeholder "${...}" tidak tertimpa rahasia.
    File lama disalin ke <nama>.bak. Komentar yaml tidak ikut (keterbatasan PyYAML)."""
    import shutil
    import yaml
    with _LOCK:
        with open(config_path, "r", encoding="utf-8-sig") as f:
            raw = yaml.safe_load(f) or {}
        _merge_locations(raw.setdefault("locations", {}))
        atcs = raw.setdefault("atcs", {})
        atcs["routes"] = effective_routes(atcs.get("routes") or [])
        admin = get_routes_admin()
        if admin["primary_route"]:
            atcs["primary_route"] = admin["primary_route"]
        atcs["activation_rules"] = {**(atcs.get("activation_rules") or {}), **admin["corridor"]}
        text = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=200)
        shutil.copyfile(config_path, config_path + ".bak")
        tmp = config_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("# Ditulis dari Dashboard Admin (Simpan ke config). Versi sebelumnya: " + os.path.basename(config_path) + ".bak\n" + text)
        os.replace(tmp, config_path)
        return config_path


# ------------------------------------------------------- GTS Admin Control ----

def get_routes_admin() -> dict:
    """{"primary_route": str, "routes": {route_id: {closed, status_override, note}}, "corridor": {objective, vc_threshold, status_threshold}}"""
    with _LOCK:
        state = copy.deepcopy(_STATE.get("routes_admin") or {})
        state.setdefault("primary_route", "")
        state.setdefault("routes", {})
        state.setdefault("corridor", {})
        return state


def update_route_admin(route_ids: list, data: dict, path: Optional[str] = None) -> dict:
    """Validasi lalu simpan kontrol admin satu rute (dan/atau rute utama)."""
    if not isinstance(data, dict):
        raise ValidationError("data harus berupa objek")
    with _LOCK:
        state = get_routes_admin()
        if "primary_route" in data:
            if data["primary_route"] not in route_ids:
                raise ValidationError("primary_route tidak dikenal")
            state["primary_route"] = data["primary_route"]
        if "corridor" in data:   # opsi koridor: kapan rute alternatif diaktifkan + tujuan pemilihan rute
            c = data["corridor"] if isinstance(data["corridor"], dict) else {}
            obj = str(c.get("objective") or "fastest")
            if obj not in ("fastest", "shortest"):
                raise ValidationError("objective harus fastest atau shortest")
            st = str(c.get("status_threshold") or "MACET").upper()
            if st not in ("PADAT", "MACET"):
                raise ValidationError("status_threshold harus PADAT atau MACET")
            state["corridor"] = {"objective": obj, "status_threshold": st,
                                 "vc_threshold": round(_num(c.get("vc_threshold", 0.85), 0.1, 3, "ambang V/C", integer=False), 2)}
        rid = data.get("route_id")
        if rid is not None:
            if rid not in route_ids:
                raise ValidationError(f"rute '{rid}' tidak dikenal")
            status = str(data.get("status_override") or "").upper()
            if status and status not in TRAFFIC_STATUSES:
                raise ValidationError("status_override harus kosong (otomatis), LANCAR, PADAT, atau MACET")
            state["routes"][rid] = {
                "closed": _bool(data.get("closed", False)),
                "status_override": status,
                "note": str(data.get("note", "") or "").strip()[:120],
                "updated_at": time.time(),
            }
        _STATE["routes_admin"] = state
        _save(path)
        return copy.deepcopy(state)


# ------------------------------------------------- Daftar ruas jalan (rute) ----

def effective_routes(base: list) -> list:
    """Rute dari config.yaml ditimpa/ditambah/dihapus lewat dashboard admin (urutan: yaml dulu, lalu yang baru)."""
    with _LOCK:
        edits = _STATE.get("routes_edit") or {"routes": {}, "deleted": []}
        out = [copy.deepcopy(edits["routes"].get(r["id"], r)) for r in base if r["id"] not in edits["deleted"]]
        known = {r["id"] for r in base}
        out += [copy.deepcopy(r) for rid, r in edits["routes"].items() if rid not in known and rid not in edits["deleted"]]
        return out


def clean_path(pts) -> list:
    """Jalur gambar ruas: titik [lat, lng] mengikuti likuk jalan (kosong = garis lurus antar CCTV)."""
    if pts in (None, "", []):
        return []
    if not isinstance(pts, list) or len(pts) > 300:
        raise ValidationError("jalur jalan maksimal 300 titik")
    out = []
    for p in pts:
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            raise ValidationError("setiap titik jalur harus [lat, lng]")
        out.append([round(_num(p[0], -90, 90, "lat jalur", integer=False), 6),
                    round(_num(p[1], -180, 180, "lng jalur", integer=False), 6)])
    if len(out) < 2:
        raise ValidationError("jalur jalan butuh minimal 2 titik")
    return out


def clean_route(data: dict, locations) -> dict:
    rid = str(data.get("id") or "").strip()
    if not re.fullmatch(r"[a-z0-9_]{1,40}", rid):
        raise ValidationError("id rute: 1-40 karakter huruf kecil, angka, atau _")
    name = str(data.get("name") or "").strip()[:80]
    if not name:
        raise ValidationError("nama jalan wajib diisi")
    nodes = data.get("nodes")
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 12 or len(set(nodes)) != len(nodes):
        raise ValidationError("titik CCTV: pilih 1-12 lokasi berbeda")
    # "lokasi#2" = arah 2 kamera itu (ROI kedua)
    unknown = [n for n in nodes if str(n).removesuffix("#2") not in locations]
    if unknown:
        raise ValidationError(f"titik CCTV tidak dikenal: {', '.join(map(str, unknown))}")
    direction = str(data.get("direction") or "dua_arah")
    if direction not in ("dua_arah", "satu_arah"):
        raise ValidationError("arah jalan harus dua_arah atau satu_arah")
    # rute alternatif untuk ruas ini bila dijadikan rute utama (boleh lebih dari satu, urut prioritas)
    alts = data.get("alternatives") or []
    if not isinstance(alts, list) or len(alts) > 10 or not all(isinstance(a, str) and re.fullmatch(r"[a-z0-9_]{1,40}", a) for a in alts):
        raise ValidationError("alternatif: maksimal 10 id rute")
    alts = [a for a in dict.fromkeys(alts) if a != rid]
    return {
        "id": rid,
        "name": name,
        "nodes": list(nodes),
        "capacity_veh_per_hour": _num(data.get("capacity_veh_per_hour"), 50, 20000, "kapasitas (kend/jam)"),
        "baseline_traveltime_minutes": round(_num(data.get("baseline_traveltime_minutes"), 0.1, 240, "waktu tempuh normal", integer=False), 2),
        "distance_km": round(_num(data.get("distance_km"), 0.01, 200, "jarak (km)", integer=False), 2),
        "path": clean_path(data.get("path")),
        "direction": direction,
        "alternatives": alts,
    }


def save_route(locations, data: dict, path: Optional[str] = None) -> dict:
    """Tambah atau ubah satu ruas jalan."""
    if not isinstance(data, dict):
        raise ValidationError("data harus berupa objek")
    route = clean_route(data, locations)
    with _LOCK:
        edits = copy.deepcopy(_STATE.get("routes_edit") or {"routes": {}, "deleted": []})
        edits["routes"][route["id"]] = route
        edits["deleted"] = [d for d in edits["deleted"] if d != route["id"]]
        _STATE["routes_edit"] = edits
        _save(path)
        return route


def delete_route(base: list, rid: str, path: Optional[str] = None) -> None:
    with _LOCK:
        ids = [r["id"] for r in effective_routes(base)]
        if rid not in ids:
            raise ValidationError(f"rute '{rid}' tidak dikenal")
        if len(ids) == 1:
            raise ValidationError("minimal harus ada satu ruas jalan")
        edits = copy.deepcopy(_STATE.get("routes_edit") or {"routes": {}, "deleted": []})
        edits["routes"].pop(rid, None)
        if rid in {r["id"] for r in base}:
            edits["deleted"].append(rid)
        _STATE["routes_edit"] = edits
        admin = get_routes_admin()          # bersihkan kontrol admin rute yang dihapus
        admin["routes"].pop(rid, None)
        if admin["primary_route"] == rid:
            admin["primary_route"] = ""
        _STATE["routes_admin"] = admin
        _save(path)


if __name__ == "__main__":
    # self-check: python runtime.py
    import tempfile as _tf
    p = os.path.join(_tf.mkdtemp(), "state.json")
    ids = ["a", "b"]
    s = update_route_admin(ids, {"route_id": "a", "closed": "true", "status_override": "padat"}, path=p)
    assert s["routes"]["a"]["closed"] is True and s["routes"]["a"]["status_override"] == "PADAT"
    s = update_route_admin(ids, {"primary_route": "b"}, path=p)
    assert s["primary_route"] == "b" and s["routes"]["a"]["closed"]
    for bad in ({"route_id": "zzz"}, {"primary_route": "zzz"}, {"route_id": "a", "status_override": "X"}):
        try:
            update_route_admin(ids, bad, path=p)
            raise AssertionError(f"harus ditolak: {bad}")
        except ValidationError:
            pass
    load(p)
    assert get_routes_admin()["primary_route"] == "b"   # tersimpan ke file
    for gone in ("program", "camera"):
        try:
            update("x", gone, {}, path=p)
            raise AssertionError(f"section {gone} harus sudah dihapus")
        except ValidationError:
            pass
    base = [{"id": "a", "name": "A", "nodes": ["L1"], "capacity_veh_per_hour": 900, "baseline_traveltime_minutes": 5, "distance_km": 2}]
    new = {"id": "c", "name": "Jl. Baru", "nodes": ["L1", "L2"], "capacity_veh_per_hour": "800", "baseline_traveltime_minutes": "6.5", "distance_km": 3}
    assert save_route({"L1", "L2"}, new, path=p)["capacity_veh_per_hour"] == 800
    assert save_route({"L1"}, {**new, "id": "d", "nodes": ["L1", "L1#2"]}, path=p)["nodes"] == ["L1", "L1#2"]
    assert save_route({"L1", "L2"}, {**new, "id": "d", "path": [[-7.1, 112.2], [-7.2, 112.3]]}, path=p)["path"][0] == [-7.1, 112.2]
    assert save_route({"L1", "L2"}, {**new, "id": "d", "path": None}, path=p)["path"] == []
    r = save_route({"L1", "L2"}, {**new, "id": "d", "direction": "satu_arah", "alternatives": ["a", "a", "d", "c"]}, path=p)
    assert r["direction"] == "satu_arah" and r["alternatives"] == ["a", "c"]
    for bad in ({"direction": "x"}, {"alternatives": ["Bad Id"]}):
        try:
            save_route({"L1", "L2"}, {**new, "id": "d", **bad}, path=p)
            raise AssertionError(f"harus ditolak: {bad}")
        except ValidationError:
            pass
    for bad in ([[-7.1, 112.2]], [[-7.1, 112.2, 1], [0, 0]], [[999, 0], [0, 0]]):
        try:
            save_route({"L1", "L2"}, {**new, "id": "d", "path": bad}, path=p)
            raise AssertionError(f"jalur harus ditolak: {bad}")
        except ValidationError:
            pass
    delete_route(base, "d", path=p)
    assert update("L1", "roi2", {"roi2": [[0, 0], [1, 0], [1, 1]], "roi2_label": "arah balik"}, path=p)["roi2_label"] == "arah balik"
    assert update("L1", "roi2", {"roi2": None}, path=p)["roi2"] is None
    save_route({"L1"}, {**base[0], "name": "A2"}, path=p)
    assert [(r["id"], r["name"]) for r in effective_routes(base)] == [("a", "A2"), ("c", "Jl. Baru")]
    for bad in ({**new, "nodes": ["zzz"]}, {**new, "id": "Bad Id"}, {**new, "distance_km": -1}, {**new, "nodes": ["L1", "L1"]}):
        try:
            save_route({"L1", "L2"}, bad, path=p)
            raise AssertionError(f"harus ditolak: {bad}")
        except ValidationError:
            pass
    update_route_admin(["a", "c"], {"corridor": {"objective": "shortest", "vc_threshold": "0.7", "status_threshold": "padat"}}, path=p)
    assert get_routes_admin()["corridor"] == {"objective": "shortest", "status_threshold": "PADAT", "vc_threshold": 0.7}
    for bad in ({"objective": "x"}, {"status_threshold": "LANCAR"}, {"vc_threshold": 9}):
        try:
            update_route_admin(["a"], {"corridor": bad}, path=p)
            raise AssertionError(f"corridor harus ditolak: {bad}")
        except ValidationError:
            pass
    update_route_admin(["a", "c"], {"route_id": "a", "closed": True, "primary_route": "a"}, path=p)
    delete_route(base, "a", path=p)
    assert [r["id"] for r in effective_routes(base)] == ["c"]
    assert "a" not in get_routes_admin()["routes"] and get_routes_admin()["primary_route"] == ""
    try:
        delete_route(base, "c", path=p)
        raise AssertionError("rute terakhir tidak boleh dihapus")
    except ValidationError:
        pass
    save_route({"L1"}, base[0], path=p)          # dibuat lagi -> tidak lagi terhapus
    assert [r["id"] for r in effective_routes(base)] == ["a", "c"]
    import yaml as _y
    cp = os.path.join(os.path.dirname(p), "cfg.yaml")
    with open(cp, "w", encoding="utf-8") as fh:
        fh.write("supabase: {service_key: \"${KEY}\"}\natcs:\n  routes:\n  - {id: a, name: A, nodes: [L1], capacity_veh_per_hour: 900, baseline_traveltime_minutes: 5, distance_km: 2}\nlocations:\n  L1: {lat: 1, lng: 2}\n")
    update("L1", "roi", {"roi": [[0, 0], [1, 0], [1, 1]], "roi_label": "kanan"}, path=p)
    export_config(cp)
    out = _y.safe_load(open(cp, encoding="utf-8"))
    assert out["locations"]["L1"]["roi"] == [[0, 0], [1, 0], [1, 1]] and out["supabase"]["service_key"] == "${KEY}"
    assert [r["id"] for r in out["atcs"]["routes"]] == ["a", "c"] and os.path.exists(cp + ".bak")
    print("runtime self-check OK")
