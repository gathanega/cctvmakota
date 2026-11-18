"""
ATCS (Adaptive Traffic Control System) - Route recommendation engine.
Decision-support only: recommends diversion allocations but does NOT control lights.
All decisions traced to config and output files.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, List

log = logging.getLogger("cctvmakota.atcs")


@dataclass
class RouteStatus:
    """Status of one route."""
    route_id: str
    route_name: str
    capacity_veh_per_hour: int
    baseline_traveltime_minutes: float
    current_demand: int  # vehicles per hour (from CCTV occupancy proxy)
    vc_ratio: float  # V/C = current_demand / capacity
    traffic_status: str  # LANCAR, PADAT, MACET
    is_active: bool  # true if route can handle traffic
    distance_km: float = 0.0
    est_traveltime_minutes: float = 0.0  # BPR: baseline * (1 + 0.15 * (V/C)^4)
    closed: bool = False  # ditutup admin (Admin Control GTS)
    status_source: str = "deteksi"  # deteksi | admin


_SEVERITY = {"LANCAR": 0, "PADAT": 1, "MACET": 2}


def _at_or_worse(status: str, threshold: str) -> bool:
    """Ambang status PADAT juga terpicu saat MACET."""
    return _SEVERITY.get(status, 0) >= _SEVERITY.get(threshold, 2)


def bpr_traveltime(baseline_minutes: float, vc: float) -> float:
    """Bureau of Public Roads delay curve (alpha 0.15, beta 4)."""
    return baseline_minutes * (1.0 + 0.15 * vc ** 4)
    

@dataclass
class AllocationRecommendation:
    """Traffic allocation recommendation across routes."""
    timestamp: float
    scenario_id: str
    activated: bool  # true if any route is congested
    activation_reason: str  # reason for diversion
    routes: List[RouteStatus]  # all routes with status
    allocation_percent: Dict[str, float]  # route_id -> percent allocation (0-100, sum=100)
    total_demand_veh_per_hour: int  # total vehicles on network
    estimated_system_saving_percent: float  # model estimate, not field verified
    primary_route_id: str = ""
    recommended_route_id: str = ""   # route to use (primary when not congested)
    fastest_route_id: str = ""
    shortest_route_id: str = ""
    reroute: bool = False
    reroute_reason: str = ""


def choose_route(
    routes: List[RouteStatus],
    primary_id: str,
    objective: str = "fastest",
    activation_threshold_vc: float = 0.85,
    activation_threshold_status: str = "MACET",
) -> Dict[str, object]:
    """Pick fastest/shortest usable route; reroute only when the primary is congested."""
    by_id = {r.route_id: r for r in routes}
    primary = by_id.get(primary_id) or routes[0]
    open_routes = [r for r in routes if not r.closed] or routes
    usable = [r for r in open_routes if r.is_active and r.vc_ratio < activation_threshold_vc] or open_routes
    fastest = min(usable, key=lambda r: r.est_traveltime_minutes)
    shortest = min(usable, key=lambda r: r.distance_km)
    congested = (primary.closed or primary.vc_ratio >= activation_threshold_vc
                 or _at_or_worse(primary.traffic_status, activation_threshold_status))
    pick = shortest if objective == "shortest" else fastest
    reroute = congested and pick.route_id != primary.route_id
    why = "ditutup admin" if primary.closed else (
        f"{primary.traffic_status}{' (admin)' if primary.status_source == 'admin' else ''}, "
        f"V/C {primary.vc_ratio:.2f} (ambang {activation_threshold_vc})")
    if reroute:
        reason = f"{primary.route_name}: {why} -> {pick.route_name}"
    elif congested:
        reason = f"{primary.route_name}: {why}, tetapi tidak ada alternatif yang lebih baik"
    else:
        reason = f"{primary.route_name} normal (V/C {primary.vc_ratio:.2f}), tetap di rute utama"
    return {
        "primary_route_id": primary.route_id,
        "recommended_route_id": pick.route_id if reroute else primary.route_id,
        "fastest_route_id": fastest.route_id,
        "shortest_route_id": shortest.route_id,
        "reroute": reroute,
        "reroute_reason": reason,
    }
    

def compute_route_metrics(
    route_config: Dict,
    location_statuses: Dict[str, dict],  # location -> {traffic_status, vehicle_count}
) -> RouteStatus:
    """Compute one route's current status."""
    route_id = route_config["id"]
    route_name = route_config["name"]
    capacity = route_config["capacity_veh_per_hour"]
    baseline_tt = route_config["baseline_traveltime_minutes"]
    
    # Aggregate demand from all CCTV on this route
    nodes = route_config.get("nodes", [])
    total_demand = 0
    is_active = True
    traffic_statuses = []
    
    for node in nodes:
        status = location_statuses.get(node, {})
        # From CCTV detection: convert vehicle_count to veh/hour proxy
        # (simple heuristic: vehicle_count * 60 / poll_interval_seconds)
        vehicle_count = status.get("vehicle_count", 0)
        # Assume poll_interval ~30s, so veh/hour ≈ count * 120
        demand_proxy = int(vehicle_count * 120)
        total_demand += demand_proxy
        traffic_status = status.get("traffic_status", "LANCAR")
        traffic_statuses.append(traffic_status)
        
        # Route becomes inactive (blocked) if any node reports MACET
        if traffic_status == "MACET":
            is_active = False
    
    vc = total_demand / capacity if capacity > 0 else 0.0
    # Worst traffic status on route
    worst_status = "LANCAR"
    if "MACET" in traffic_statuses:
        worst_status = "MACET"
    elif "PADAT" in traffic_statuses:
        worst_status = "PADAT"

    # Admin Control: status manual menimpa deteksi; rute ditutup tidak pernah direkomendasikan
    override = route_config.get("status_override") or ""
    if override in ("LANCAR", "PADAT", "MACET"):
        worst_status = override
        is_active = override != "MACET"
    closed = bool(route_config.get("closed"))
    if closed:
        is_active = False
    
    return RouteStatus(
        route_id=route_id,
        route_name=route_name,
        capacity_veh_per_hour=capacity,
        baseline_traveltime_minutes=baseline_tt,
        current_demand=total_demand,
        vc_ratio=vc,
        traffic_status=worst_status,
        is_active=is_active,
        distance_km=float(route_config.get("distance_km", 0.0)),
        est_traveltime_minutes=round(bpr_traveltime(baseline_tt, vc), 2),
        closed=closed,
        status_source="admin" if override else "deteksi",
    )


def compute_allocation(
    routes: List[RouteStatus],
    activation_threshold_vc: float = 0.85,
    activation_threshold_status: str = "MACET",
) -> Dict[str, float]:
    """
    Recommend traffic allocation across routes.
    - If all routes have low V/C and no MACET: return proportional (baseline).
    - If any route is congested: redistribute to non-congested routes.
    Total allocation always = 100%.
    """
    allocation = {}
    
    # Check if any route is congested
    has_congestion = any(
        r.closed or r.vc_ratio >= activation_threshold_vc or _at_or_worse(r.traffic_status, activation_threshold_status)
        for r in routes
    )
    
    if not has_congestion:
        # No congestion: allocate proportional to capacity
        total_capacity = sum(r.capacity_veh_per_hour for r in routes)
        for route in routes:
            pct = (route.capacity_veh_per_hour / total_capacity * 100.0) if total_capacity > 0 else 0.0
            allocation[route.route_id] = round(pct, 1)
    else:
        # Congestion detected: allocate to non-congested routes based on remaining capacity
        active_routes = [r for r in routes if r.is_active and r.vc_ratio < activation_threshold_vc]
        if not active_routes:
            # All routes congested: fall back to proportional
            total_capacity = sum(r.capacity_veh_per_hour for r in routes)
            for route in routes:
                pct = (route.capacity_veh_per_hour / total_capacity * 100.0) if total_capacity > 0 else 0.0
                allocation[route.route_id] = round(pct, 1)
        else:
            # Allocate to active routes proportional to remaining capacity
            total_remaining = sum((1.0 - r.vc_ratio) * r.capacity_veh_per_hour for r in active_routes)
            for route in routes:
                if route.is_active and route.vc_ratio < activation_threshold_vc:
                    pct = ((1.0 - route.vc_ratio) * route.capacity_veh_per_hour / total_remaining * 100.0) if total_remaining > 0 else 0.0
                    allocation[route.route_id] = round(pct, 1)
                else:
                    allocation[route.route_id] = 0.0
    
    # Ensure sum = 100% (due to rounding)
    total_pct = sum(allocation.values())
    if total_pct > 0:
        for key in allocation:
            allocation[key] = round(allocation[key] * 100.0 / total_pct, 1)
    
    return allocation


def compute_recommendation(
    routes_config: List[Dict],
    location_statuses: Dict[str, dict],
    activation_threshold_vc: float = 0.85,
    activation_threshold_status: str = "MACET",
    primary_route_id: str = "",
    objective: str = "fastest",
) -> AllocationRecommendation:
    """Compute full ATCS recommendation."""
    
    # Compute status for each route
    route_statuses = [
        compute_route_metrics(cfg, location_statuses)
        for cfg in routes_config
    ]
    
    # Check activation
    has_congestion = any(
        r.closed or r.vc_ratio >= activation_threshold_vc or _at_or_worse(r.traffic_status, activation_threshold_status)
        for r in route_statuses
    )
    
    # Compute allocation
    allocation = compute_allocation(route_statuses, activation_threshold_vc, activation_threshold_status)
    
    # Total demand across network
    total_demand = sum(r.current_demand for r in route_statuses)
    
    # Estimated system saving (model estimate, not verified)
    # Assumption: diversion reduces peak V/C by ~15% if activated
    if has_congestion:
        peak_vc = max(r.vc_ratio for r in route_statuses)
        estimated_saving = min(15.0, (peak_vc - 0.70) * 100.0) if peak_vc > 0.70 else 0.0
    else:
        estimated_saving = 0.0
    
    return AllocationRecommendation(
        timestamp=time.time(),
        scenario_id=f"atcs_{int(time.time())}",
        activated=has_congestion,
        activation_reason="High V/C or MACET status detected" if has_congestion else "No congestion",
        routes=route_statuses,
        allocation_percent=allocation,
        total_demand_veh_per_hour=total_demand,
        estimated_system_saving_percent=estimated_saving,
        **choose_route(route_statuses, primary_route_id, objective,
                       activation_threshold_vc, activation_threshold_status),
    )


if __name__ == "__main__":
    # self-check: python atcs_routing.py
    cfg = [
        {"id": "a", "name": "A", "nodes": ["x"], "capacity_veh_per_hour": 1200, "baseline_traveltime_minutes": 5, "distance_km": 3.0},
        {"id": "b", "name": "B", "nodes": ["y"], "capacity_veh_per_hour": 1100, "baseline_traveltime_minutes": 6, "distance_km": 2.5},
        {"id": "c", "name": "C", "nodes": ["z"], "capacity_veh_per_hour": 800, "baseline_traveltime_minutes": 8, "distance_km": 4.0},
    ]
    assert abs(bpr_traveltime(10, 1.0) - 11.5) < 1e-9
    calm = {"x": {"vehicle_count": 2, "traffic_status": "LANCAR"}}
    r = compute_recommendation(cfg, calm, primary_route_id="a")
    assert not r.reroute and r.recommended_route_id == "a"
    jam = {"x": {"vehicle_count": 20, "traffic_status": "MACET"}, "y": {"vehicle_count": 1}, "z": {"vehicle_count": 1}}
    r = compute_recommendation(cfg, jam, primary_route_id="a", objective="fastest")
    assert r.reroute and r.recommended_route_id == "b", r.recommended_route_id
    r = compute_recommendation(cfg, jam, primary_route_id="a", objective="shortest")
    assert r.recommended_route_id == "b" and r.shortest_route_id == "b"
    allj = {k: {"vehicle_count": 20, "traffic_status": "MACET"} for k in "xyz"}
    r = compute_recommendation(cfg, allj, primary_route_id="a")
    assert r.recommended_route_id in {"a", "b", "c"}  # semua macet: tetap ada jawaban, tidak crash
    # Admin Control: tutup rute utama -> alihkan; tutup alternatif tercepat -> pilih berikutnya
    closed_a = [{**cfg[0], "closed": True}, *cfg[1:]]
    r = compute_recommendation(closed_a, calm, primary_route_id="a")
    assert r.reroute and r.recommended_route_id == "b" and "ditutup" in r.reroute_reason
    closed_b = [cfg[0], {**cfg[1], "closed": True}, cfg[2]]
    r = compute_recommendation(closed_b, jam, primary_route_id="a")
    assert r.recommended_route_id == "c" and r.allocation_percent["b"] == 0.0
    forced = [{**cfg[0], "status_override": "MACET"}, *cfg[1:]]
    r = compute_recommendation(forced, calm, primary_route_id="a")
    assert r.reroute and r.routes[0].status_source == "admin"
    all_closed = [{**c, "closed": True} for c in cfg]
    assert compute_recommendation(all_closed, calm, primary_route_id="a").recommended_route_id  # tidak crash
    print("atcs_routing self-check OK")
