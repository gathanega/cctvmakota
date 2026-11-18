"""
GIS Malang: jaringan jalan OpenStreetMap (Overpass) di sekitar kamera + analisis lokasi CCTV.

- fetch_roads(): unduh jalan (primary..residential) dalam bbox kamera +1,5 km, simpan ke gis/malang_roads.geojson
- analyze(locations): per kamera -> ruas jalan OSM terdekat (nama, kelas, lajur, jarak m, kapasitas perkiraan)
- road_layer(statuses): GeoJSON jalan; ruas dalam 150 m dari kamera diberi status kepadatan kamera itu

Tanpa geopandas: jarak titik-garis pakai proyeksi lokal (equirectangular), cukup akurat untuk skala kota.

    python gis_malang.py [--fetch]   # cetak analisis (--fetch: unduh ulang data jalan OSM)
"""
from __future__ import annotations

import json
import math
import re
import os
import urllib.parse
import urllib.request

GIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gis")
ROADS_FILE = os.path.join(GIS_DIR, "malang_roads.geojson")
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter"]
HIGHWAYS = "motorway|trunk|primary|secondary|tertiary|unclassified|residential"
# kapasitas perkiraan per lajur (smp/jam), MKJI 1997 disederhanakan
LANE_CAP = {"motorway": 2300, "trunk": 2000, "primary": 1800, "secondary": 1500,
            "tertiary": 1200, "unclassified": 900, "residential": 700}
DEFAULT_LANES = {"motorway": 4, "trunk": 4, "primary": 4, "secondary": 2, "tertiary": 2}
NEAR_M = 150

_cache: dict | None = None


def _overpass(q: str) -> dict:
    err = None
    for url in OVERPASS:
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "cctvmakota/2.0"})
            return json.load(urllib.request.urlopen(req, timeout=90))
        except Exception as ex:   # server Overpass sering sibuk: coba mirror berikutnya
            err = ex
    raise RuntimeError(f"semua server Overpass gagal: {err}")


SIGNALS_FILE = os.path.join(GIS_DIR, "traffic_signals.json")


def fetch_signals() -> list[dict]:
    """Lampu merah (highway=traffic_signals) OSM dalam bbox data jalan, + nama jalan yang bertemu di situ.
    Bila Overpass gagal: pakai simpang jalan utama (titik temu 2 jalan utama bernama beda) dari data jalan lokal."""
    w, s, e, n = roads()["bbox"]
    try:
        data = _overpass(f'[out:json][timeout:60];node["highway"="traffic_signals"]({s},{w},{n},{e});out;')
        nodes = [(el["id"], el["lat"], el["lon"], "osm_traffic_signals") for el in data.get("elements", [])]
    except RuntimeError:
        nodes = [(i, la, ln, "simpang_jalan_utama") for i, (la, ln) in enumerate(major_junctions())]
    out = []
    for nid, lat, lng, src in nodes:
        names = sorted({f["properties"]["name"] for f in roads()["features"]
                        if f["properties"]["name"] and _dist_m(lat, lng, f["geometry"]["coordinates"]) < 25})
        out.append({"id": nid, "lat": lat, "lng": lng, "roads": names, "source": src})
    with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    return out


def major_junctions() -> list[tuple[float, float]]:
    """Titik yang dipakai bersama >= 2 jalan utama dengan nama berbeda (simpang)."""
    seen: dict[tuple, set] = {}
    for f in roads()["features"]:
        p = f["properties"]
        if p["highway"] not in MAJOR or not p["name"]:
            continue
        for x, y in f["geometry"]["coordinates"]:
            seen.setdefault((round(y, 5), round(x, 5)), set()).add(p["name"])
    pts = [k for k, v in seen.items() if len(v) >= 2]
    merged: list[tuple[float, float]] = []
    for p in pts:   # simpang lebar = beberapa titik berdekatan; gabung yang < 40 m
        if all(math.hypot((p[0] - q[0]) * 111320, (p[1] - q[1]) * 110000) > 40 for q in merged):
            merged.append(p)
    return merged


def signals() -> list[dict]:
    if not os.path.isfile(SIGNALS_FILE):
        return fetch_signals()
    with open(SIGNALS_FILE, encoding="utf-8") as f:
        return json.load(f)


def fetch_roads(locations: dict, margin_deg: float = 0.0135) -> dict:
    pts = [(float(v["lat"]), float(v["lng"])) for v in locations.values() if v and v.get("lat") is not None]
    s, n = min(p[0] for p in pts) - margin_deg, max(p[0] for p in pts) + margin_deg
    w, e = min(p[1] for p in pts) - margin_deg, max(p[1] for p in pts) + margin_deg
    data = _overpass(f'[out:json][timeout:60];way["highway"~"^({HIGHWAYS})$"]({s},{w},{n},{e});out tags geom;')
    feats = []
    for el in data.get("elements", []):
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        tg = el.get("tags", {})
        feats.append({"type": "Feature", "id": el["id"],
                      "properties": {"name": tg.get("name", ""), "highway": tg.get("highway", ""),
                                     "lanes": tg.get("lanes"), "oneway": tg.get("oneway") == "yes"},
                      "geometry": {"type": "LineString", "coordinates": [[round(g["lon"], 6), round(g["lat"], 6)] for g in geom]}})
    gj = {"type": "FeatureCollection", "bbox": [w, s, e, n], "source": "OpenStreetMap (ODbL)", "features": feats}
    os.makedirs(GIS_DIR, exist_ok=True)
    tmp = ROADS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(gj, f, separators=(",", ":"))
    os.replace(tmp, ROADS_FILE)
    global _cache
    _cache = gj
    return gj


def roads() -> dict:
    global _cache
    if _cache is None:
        if not os.path.isfile(ROADS_FILE):
            return {"type": "FeatureCollection", "features": []}
        with open(ROADS_FILE, encoding="utf-8") as f:
            _cache = json.load(f)
    return _cache


def _dist_m(lat, lng, line) -> float:
    return _nearest_point(lat, lng, line)[0]


def _nearest_point(lat, lng, line):
    """(jarak m, lat, lng) titik terdekat pada garis."""
    k = 111320.0
    kx = k * math.cos(math.radians(lat))
    best = (float("inf"), lat, lng)
    for (x1, y1), (x2, y2) in zip(line, line[1:]):
        ax, ay = (x1 - lng) * kx, (y1 - lat) * k
        bx, by = (x2 - lng) * kx, (y2 - lat) * k
        dx, dy = bx - ax, by - ay
        L = dx * dx + dy * dy
        t = 0.0 if L == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / L))
        d = math.hypot(ax + t * dx, ay + t * dy)
        if d < best[0]:
            best = (d, y1 + t * (y2 - y1), x1 + t * (x2 - x1))
    return best


def _lanes(p) -> int:
    try:
        return max(1, int(str(p.get("lanes")).split(";")[0]))
    except (TypeError, ValueError):
        return DEFAULT_LANES.get(p.get("highway"), 1 if p.get("oneway") else 2)


def road_hint(label: str) -> str:
    """"Jl. Gajayana Arah Utara (Sardo)" -> "gajayana"; dipakai mencocokkan nama jalan OSM."""
    s = re.sub(r"\(.*?\)", " ", label or "").lower()
    s = re.sub(r"^\s*(jl\.?|jalan|pertigaan|perempatan|simpang)\s+", "", s)
    s = re.split(r"\barah\b", s)[0]
    return re.sub(r"[^a-z0-9]", "", s.split("mt ")[-1])   # "mt haryono" -> "haryono", "sumber sari" -> "sumbersari"


MAJOR = {"motorway", "trunk", "primary", "secondary", "tertiary"}


def nearest_road(lat: float, lng: float, named_only: bool = True, hint: str = "", max_m: float = 1500,
                 major_only: bool = False, exact: bool = False) -> dict | None:
    best = None
    for f in roads()["features"]:
        p = f["properties"]
        if named_only and not p["name"]:
            continue
        if hint and hint not in re.sub(r"[^a-z0-9]", "", p["name"].lower()):
            continue
        if exact and re.sub(r"[^a-z0-9]", "", p["name"].lower()) not in ("jalan" + hint, "jl" + hint):
            continue
        if major_only and (p["highway"] not in MAJOR or re.search(r"\bgang\b|\d", p["name"].lower())):
            continue
        pt = _nearest_point(lat, lng, f["geometry"]["coordinates"])
        if best is None or pt[0] < best[0][0]:
            best = (pt, f)
    if not best or best[0][0] > max_m:
        return None
    (d, plat, plng), f = best
    p = f["properties"]
    lanes = _lanes(p)
    return {"osm_id": f["id"], "name": p["name"], "highway": p["highway"], "lanes": lanes, "oneway": p["oneway"],
            "distance_m": round(d, 1), "snap": [round(plat, 6), round(plng, 6)], "capacity_veh_per_hour": lanes * LANE_CAP.get(p["highway"], 800)}


def analyze(locations: dict, statuses: dict | None = None) -> list[dict]:
    """Analisis tiap kamera: jalan OSM terdekat + perkiraan V/C dari hitungan kendaraan."""
    out = []
    for loc, cfg in locations.items():
        cfg = cfg or {}
        if cfg.get("lat") is None:
            continue
        lat, lng = float(cfg["lat"]), float(cfg["lng"])
        hint = road_hint(cfg.get("label", ""))
        road = ((nearest_road(lat, lng, hint=hint, exact=True) or nearest_road(lat, lng, hint=hint, major_only=True) or nearest_road(lat, lng, hint=hint)) if hint else None) \
            or nearest_road(lat, lng)
        st = (statuses or {}).get(loc) or {}
        item = {"location": loc, "label": cfg.get("label", loc), "lat": cfg["lat"], "lng": cfg["lng"], "road": road,
                "traffic_status": st.get("traffic_status"), "vehicle_count": st.get("vehicle_count")}
        item["matched_by_name"] = bool(hint and road and hint in re.sub(r"[^a-z0-9]", "", road["name"].lower()))
        if road and road["distance_m"] > 40:
            item["warning"] = f"pin {road['distance_m']:.0f} m dari {road['name'] or 'jalan'}; usul koordinat {road['snap'][0]}, {road['snap'][1]}"
        out.append(item)
    return out


def road_layer(locations: dict, statuses: dict) -> dict:
    """GeoJSON jalan; ruas bernama sama dengan jalan kamera dan <= NEAR_M*2 dari titik pada jalan itu
    mendapat status kamera (pin yang salah taruh tidak mewarnai gang di sebelahnya)."""
    cams = []
    for a in analyze(locations, statuses):
        r = a.get("road")
        if r:
            cams.append((r["snap"][0], r["snap"][1], r["name"], a["traffic_status"], a["location"]))
    feats = []
    for f in roads()["features"]:
        line = f["geometry"]["coordinates"]
        props = dict(f["properties"])
        near = min(((_dist_m(la, ln, line), st, l) for la, ln, nm, st, l in cams if nm == props["name"]), default=None)
        if near and near[0] <= NEAR_M * 2:
            props.update(status=near[1] or "BELUM ADA DATA", location=near[2])
        feats.append({**f, "properties": props})
    return {"type": "FeatureCollection", "source": roads().get("source", ""), "features": feats}


if __name__ == "__main__":
    import sys
    import yaml
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = yaml.safe_load(open("config.cctvmakota.yaml", encoding="utf-8-sig"))
    if "--fetch" in sys.argv or not os.path.isfile(ROADS_FILE):
        print(f"{len(fetch_roads(cfg['locations'])['features'])} ruas jalan OSM -> {ROADS_FILE}")
    for a in analyze(cfg["locations"]):
        r = a["road"] or {}
        print(f"  {a['label']:<40} -> {r.get('name')} ({r.get('highway')}, {r.get('lanes')} lajur, "
              f"{r.get('distance_m')} m, ~{r.get('capacity_veh_per_hour')} kend/jam) {a.get('warning', '')}")
