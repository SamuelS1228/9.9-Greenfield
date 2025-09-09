
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from utils import demand_weighted_centroid_fast, compute_service_mix, haversine_miles

rng = np.random.default_rng(42)

def _init_kmeans_pp(points: np.ndarray, weights: np.ndarray, k: int, fixed_centers: Optional[np.ndarray]) -> np.ndarray:
    """
    Fast K-means++ initialization with weights and optional fixed centers.
    points: (n,2) [lon,lat]
    weights: (n,)
    returns centers (k,2)
    """
    centers = []
    if fixed_centers is not None and len(fixed_centers) > 0:
        centers.extend([c.tolist() for c in fixed_centers])
    m = k - len(centers)
    if m <= 0:
        return np.array(centers[:k], dtype=float)

    # choose first by weight
    wsum = weights.sum() if weights.sum() > 0 else 1.0
    idx = rng.choice(len(points), p=(weights/wsum))
    centers.append(points[idx].tolist())

    # Pre-allocate distance-to-nearest array
    C = np.array(centers, dtype=float)
    d2_nearest = np.min((points[:,None,:] - C[None,:,:])**2, axis=1).min(axis=1)

    for _ in range(m-1):
        # Weighted probability proportional to distance^2 * weight
        probs = d2_nearest * (weights / wsum)
        psum = probs.sum()
        if psum <= 0:
            idx = rng.integers(0, len(points))
        else:
            idx = rng.choice(len(points), p=(probs/psum))
        centers.append(points[idx].tolist())
        C = np.array(centers, dtype=float)
        d2_nearest = np.minimum(d2_nearest, np.min((points[:,None,:] - C[None,:,:])**2, axis=1).min(axis=1))

    return np.array(centers[:k], dtype=float)

def _assign_to_nearest(c_lon, c_lat, centers):
    """
    Vectorized assignment using haversine vs each center; returns argmin over centers.
    """
    # centers: (k,2)
    k = centers.shape[0]
    d_stack = []
    for j in range(k):
        d_stack.append(haversine_miles(c_lon, c_lat, centers[j,0], centers[j,1]))
    D = np.stack(d_stack, axis=1)  # (n,k)
    return np.argmin(D, axis=1)

def _update_centers_fast(c_lon, c_lat, demand, assign_idx, k, fixed_mask=None, fixed_centers=None):
    """
    Update centers via weighted means using bincount (fully vectorized).
    """
    centers = np.zeros((k,2), dtype=float)
    # Sum of weights per cluster
    wsum = np.bincount(assign_idx, weights=demand, minlength=k)
    # Weighted sums
    lon_sum = np.bincount(assign_idx, weights=c_lon * demand, minlength=k)
    lat_sum = np.bincount(assign_idx, weights=c_lat * demand, minlength=k)
    # Avoid divide-by-zero; if wsum==0, keep previous or global mean (handled below)
    with np.errstate(invalid='ignore', divide='ignore'):
        centers[:,0] = np.where(wsum>0, lon_sum/wsum, np.nan)
        centers[:,1] = np.where(wsum>0, lat_sum/wsum, np.nan)

    # Fill empty clusters with global demand-weighted centroid
    mask_empty = ~np.isfinite(centers[:,0])
    if mask_empty.any():
        global_lon, global_lat = demand_weighted_centroid_fast(c_lon, c_lat, demand)
        centers[mask_empty,0] = global_lon
        centers[mask_empty,1] = global_lat

    # Re-impose fixed centers
    if fixed_mask is not None and fixed_centers is not None:
        centers[fixed_mask] = fixed_centers[fixed_mask]
    return centers

def solve_with_service_levels(
    cust_df: pd.DataFrame,
    k: int,
    service_bands: List[Dict],
    max_iters: int = 50,
    restarts: int = 50,
    fixed_wh: Optional[pd.DataFrame] = None,
    penalty_weight: float = 5e5,
    tol_improve: float = 1e-6,
    early_stop_if_met: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Optimized heuristic solver.
    - Fully vectorized center updates and cost pieces
    - Faster K-means++ init
    - Early stopping when service constraints met and improvement < tol
    """
    points = cust_df[['lon','lat']].to_numpy()
    demand = cust_df['demand'].to_numpy()
    c_lon = cust_df['lon'].to_numpy()
    c_lat = cust_df['lat'].to_numpy()

    fixed_centers = None
    fixed_mask = None
    if fixed_wh is not None and len(fixed_wh) > 0:
        fixed_centers = fixed_wh[['lon','lat']].to_numpy()
        fixed_mask = np.zeros(k, dtype=bool)
        fixed_mask[:len(fixed_centers)] = True

    best = None
    best_obj = np.inf
    best_diag = None

    for r in range(restarts):
        centers = _init_kmeans_pp(points, demand, k, fixed_centers)
        prev_assign = None
        prev_obj = np.inf

        for it in range(max_iters):
            assign_idx = _assign_to_nearest(c_lon, c_lat, centers)
            centers = _update_centers_fast(c_lon, c_lat, demand, assign_idx, k, fixed_mask, centers if fixed_centers is None else np.vstack((fixed_centers, np.zeros((max(0,k-len(fixed_centers)),2)))) )

            # Objective = demand-weighted outbound distance
            w_lon = centers[:,0][assign_idx]
            w_lat = centers[:,1][assign_idx]
            d = haversine_miles(c_lon, c_lat, w_lon, w_lat)
            obj = float((demand * d).sum())

            # Early-stopping if assignment stabilized
            if prev_assign is not None and np.array_equal(prev_assign, assign_idx):
                break

            # Lightweight early-stop on tiny improvement after constraints met
            if early_stop_if_met and np.isfinite(prev_obj) and (prev_obj - obj) / (prev_obj + 1e-9) < tol_improve:
                # Check constraints quickly; compute only when near-stalled
                tmp_wh = pd.DataFrame(centers, columns=['lon','lat'])
                svc_results, _ = compute_service_mix(cust_df, assign_idx, tmp_wh, service_bands)
                if all(row['met'] for row in svc_results):
                    break

            prev_assign = assign_idx
            prev_obj = obj

        # Final eval with penalty
        wh_df = pd.DataFrame(centers, columns=['lon','lat'])
        svc_results, _ = compute_service_mix(cust_df, assign_idx, wh_df, service_bands)
        shortfall = sum(max(0.0, row['target'] - row['pct']) for row in svc_results)
        obj_penalized = prev_obj + penalty_weight * shortfall

        if obj_penalized < best_obj:
            best = (wh_df.copy(), assign_idx.copy())
            best_obj = obj_penalized
            best_diag = {
                'restart': r,
                'iters': it+1,
                'objective': float(best_obj),
                'service_results': svc_results
            }

    return best[0], best[1], best_diag
