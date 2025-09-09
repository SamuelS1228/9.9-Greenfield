import numpy as np
from sklearn.cluster import KMeans
from utils import haversine, warehousing_cost

# Multiplier to approximate road miles from great‑circle distance
ROAD_FACTOR = 1.3

# ───────────────────────── helpers ─────────────────────────

def _distance_matrix(lon, lat, centers):
    d = np.empty((len(lon), len(centers)))
    for j, (clon, clat) in enumerate(centers):
        d[:, j] = haversine(lon, lat, clon, clat) * ROAD_FACTOR
    return d

def _assign(df, centers):
    lon = df["Longitude"].values
    lat = df["Latitude"].values
    dmat = _distance_matrix(lon, lat, centers)
    idx = dmat.argmin(axis=1)
    dmin = dmat[np.arange(len(df)), idx]
    return idx, dmin

def _greedy_select(df, k, fixed, sites, rate_out):
    """Greedy add candidate sites to minimize outbound cost, seeding with `fixed`."""
    # avoid duplicates in "fixed"
    fixed_uniq = []
    seen = set()
    for lon, lat in fixed:
        key = (round(lon, 6), round(lat, 6))
        if key not in seen:
            seen.add(key)
            fixed_uniq.append([lon, lat])

    chosen = fixed_uniq.copy()
    pool = [s for s in sites if (round(s[0],6), round(s[1],6)) not in {(round(x[0],6),round(x[1],6)) for x in chosen}]
    while len(chosen) < k and pool:
        best_site, best_cost = None, None
        for cand in pool:
            cost, _, _ = _outbound(df, chosen + [cand], rate_out)
            if best_cost is None or cost < best_cost:
                best_site, best_cost = cand, cost
        chosen.append(best_site)
        pool.remove(best_site)
    return chosen

def _outbound(df, centers, rate_out):
    idx, dmin = _assign(df, centers)
    return (df["DemandLbs"] * dmin * rate_out).sum(), idx, dmin

def _service_levels(dmin, weights):
    """
    Compute demand-weighted service levels by distance band:
      by7  = <= 350 miles        (Next day by 7AM)
      by10 = (350, 500] miles    (Next day by 10AM)
      eod  = (500, 700] miles    (Next day by end of day)
      d2p  = > 700 miles         (2 day +)
    Returns a dict of fractions in [0,1].
    """
    import numpy as _np
    wtot = float(_np.sum(weights)) if _np.sum(weights) > 0 else 1.0
    d = _np.asarray(dmin, dtype=float)
    w = _np.asarray(weights, dtype=float)

    by7  = _np.sum(w[d <= 350.0]) / wtot
    by10 = _np.sum(w[(d > 350.0) & (d <= 500.0)]) / wtot
    eod  = _np.sum(w[(d > 500.0) & (d <= 700.0)]) / wtot
    d2p  = _np.sum(w[d > 700.0]) / wtot
    return {"by7": float(by7), "by10": float(by10), "eod": float(eod), "2day": float(d2p)}

# ───────────────────────── core ─────────────────────────

def optimize(
    df, k_vals, rate_out,
    sqft_per_lb, cost_sqft, fixed_cost,
    consider_inbound=False, inbound_rate_mile=0.0, inbound_pts=None,
    fixed_centers=None, rdc_list=None, transfer_rate_mile=0.0,
    rdc_sqft_per_lb=None, rdc_cost_per_sqft=None,
    candidate_sites=None, restrict_cand=False, candidate_costs=None,
    service_level_targets=None, enforce_service_levels=False
):
    """
    Solve a 1- or 2‑tier network.

    If rdc_list is provided, those points are tier‑1 nodes:
      • RDCs: ship only to other warehouses (no direct customers)
      • SDCs: ship to other warehouses AND can be customer‑facing

    Flow with tier‑1:
      inbound supply → (RDC/SDC) → facing warehouses (including SDCs) → customers

    Costs computed:
      • Outbound (centers→customers)
      • Transfers (tier1→centers)
      • Inbound (suppliers→tier1 or →centers if no tier1)
      • Warehousing at centers (std params) + at tier1 (rdc params)

    Service-level behavior:
      • Always computes achieved service levels by demand weight.
      • If `enforce_service_levels` is True and `service_level_targets` provided,
        a large penalty is applied proportional to the sum of shortfalls across bands.
        This steers selection toward solutions that meet or exceed targets where feasible,
        and otherwise minimizes total shortfall.
    """
    inbound_pts = inbound_pts or []
    fixed_centers = fixed_centers or []
    rdc_list = rdc_list or []
    candidate_costs = candidate_costs or {}
    service_level_targets = service_level_targets or {}

    # Split tier1 nodes
    tier1_nodes = [dict(coords=r["coords"], is_sdc=bool(r.get("is_sdc"))) for r in rdc_list]
    sdc_coords = [r["coords"] for r in tier1_nodes if r["is_sdc"]]

    def _cost_sqft(lon, lat):
        if restrict_cand:
            return candidate_costs.get((round(lon, 6), round(lat, 6)), cost_sqft)
        return cost_sqft

    best = None

    for k in k_vals:
        # Ensure capacity to include any required fixed centers (user fixed + SDCs)
        fixed_all = fixed_centers + sdc_coords
        # de‑dup exact duplicates to avoid exceeding k unnecessarily
        seen = set()
        fixed_all_uniq = []
        for lon, lat in fixed_all:
            key = (round(lon,6), round(lat,6))
            if key not in seen:
                seen.add(key)
                fixed_all_uniq.append([lon, lat])
        k_eff = max(k, len(fixed_all_uniq))

        # choose customer‑facing centers
        if candidate_sites and len(candidate_sites) >= k_eff:
            centers = _greedy_select(df, k_eff, fixed_all_uniq, candidate_sites, rate_out)
        else:
            km = KMeans(n_clusters=k_eff, n_init=10, random_state=42).fit(df[["Longitude", "Latitude"]])
            centers = km.cluster_centers_.tolist()
            # overwrite first slots with required fixed centers (user fixed first, then SDCs)
            for i, fc in enumerate(fixed_all_uniq[:k_eff]):
                centers[i] = fc

        # Identify which centers are SDCs (exact coordinate match)
        sdc_keyset = {(round(x,6), round(y,6)) for x, y in sdc_coords}
        center_is_sdc = [((round(cx,6), round(cy,6)) in sdc_keyset) for cx, cy in centers]

        # Assign customers to nearest center and compute outbound
        idx, dmin = _assign(df, centers)
        assigned = df.copy()
        assigned["Warehouse"] = idx
        assigned["DistMi"] = dmin
        out_cost = (assigned["DemandLbs"] * dmin * rate_out).sum()

        # Demand handled at each center (customer demand)
        demand_per_wh = []
        for i in range(len(centers)):
            dem = assigned.loc[assigned["Warehouse"] == i, "DemandLbs"].sum()
            demand_per_wh.append(dem)
        demand_per_wh = np.array(demand_per_wh, dtype=float)

        # ── Transfers & inbound with tier‑1 (if any) ─────────────────────
        trans_cost = 0.0
        in_cost = 0.0

        if tier1_nodes:
            t1_coords = [t["coords"] for t in tier1_nodes]
            # Map each center → nearest tier1
            t1_dists = np.zeros((len(centers), len(t1_coords)), dtype=float)
            for j, (wx, wy) in enumerate(centers):
                t1_dists[j, :] = [haversine(wx, wy, tx, ty) * ROAD_FACTOR for tx, ty in t1_coords]
            center_to_t1_idx = t1_dists.argmin(axis=1)
            center_to_t1_dist = t1_dists[np.arange(len(centers)), center_to_t1_idx]

            # Transfer cost: ship the center's total demand from its tier1
            trans_cost = float(np.sum(demand_per_wh * center_to_t1_dist) * transfer_rate_mile)

            # Downstream demand aggregated by tier1
            t1_downstream_dem = np.zeros(len(t1_coords), dtype=float)
            for j in range(len(centers)):
                t1_downstream_dem[center_to_t1_idx[j]] += demand_per_wh[j]

            # Inbound cost: suppliers → tier1 proportional to each t1's downstream demand
            if consider_inbound and inbound_pts:
                for slon, slat, pct in inbound_pts:
                    d_to_t1 = np.array([haversine(slon, slat, tx, ty) * ROAD_FACTOR for tx, ty in t1_coords])
                    in_cost += float(np.sum(d_to_t1 * t1_downstream_dem) * pct * inbound_rate_mile)
        else:
            # No tier1: inbound ships directly to centers according to each center's demand
            center_to_t1_idx = None
            center_to_t1_dist = None
            t1_coords = []
            t1_downstream_dem = np.array([], dtype=float)
            if consider_inbound and inbound_pts:
                for lon, lat, pct in inbound_pts:
                    dists = np.array([haversine(lon, lat, cx, cy) * ROAD_FACTOR for cx, cy in centers])
                    in_cost += float((dists * demand_per_wh * pct * inbound_rate_mile).sum())

        # ── Warehousing costs ─────────────────────────────────────────────
        wh_cost_centers = 0.0
        for (clon, clat), dem in zip(centers, demand_per_wh):
            wh_cost_centers += warehousing_cost(dem, sqft_per_lb, _cost_sqft(clon, clat), fixed_cost)

        wh_cost_tier1 = 0.0
        if tier1_nodes:
            # Volume handled at tier1 equals the sum of demands of the centers mapped to it
            # (includes SDC "self" demand if the SDC is also a center)
            _sqft = (rdc_sqft_per_lb if rdc_sqft_per_lb is not None else sqft_per_lb)
            _csqft = (rdc_cost_per_sqft if rdc_cost_per_sqft is not None else cost_sqft)
            for handled in (t1_downstream_dem if len(t1_downstream_dem) else []):
                wh_cost_tier1 += warehousing_cost(handled, _sqft, _csqft, fixed_cost)

        wh_cost = wh_cost_centers + wh_cost_tier1
        total = out_cost + trans_cost + in_cost + wh_cost

        # ── Service levels & optional penalty ────────────────────────────
        sl = _service_levels(dmin, assigned["DemandLbs"].values)
        shortfall_sum = 0.0
        if enforce_service_levels and service_level_targets:
            for key in ("by7", "by10", "eod", "2day"):
                tgt = float(service_level_targets.get(key, 0.0))
                ach = float(sl.get(key, 0.0))
                if tgt > ach:
                    shortfall_sum += (tgt - ach)
        # Large penalty to push feasibility if possible; also enables "least shortfall" otherwise
        penalty = shortfall_sum * (total + 1.0) * 1000.0
        score = total + penalty

        if (best is None) or (score < best["score"]):
            best = dict(
                centers=centers,
                assigned=assigned,
                demand_per_wh=demand_per_wh.tolist(),
                total_cost=total,
                out_cost=out_cost,
                in_cost=in_cost,
                trans_cost=trans_cost,
                wh_cost=wh_cost,
                rdc_list=rdc_list,
                # extras for lane export / diagnostics
                tier1_coords=t1_coords,
                center_to_t1_idx=(center_to_t1_idx.tolist() if center_to_t1_idx is not None else None),
                center_to_t1_dist=(center_to_t1_dist.tolist() if center_to_t1_dist is not None else None),
                tier1_downstream_dem=t1_downstream_dem.tolist() if len(t1_downstream_dem) else [],
                center_is_sdc=center_is_sdc,
                # service-levels
                service_levels=sl,
                sl_targets=service_level_targets,
                sl_penalty=float(penalty),
                score=float(score),
            )

    return best
