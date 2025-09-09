
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from utils import demand_weighted_centroid, compute_service_mix, check_service_constraints, haversine_miles

rng = np.random.default_rng(42)

def _init_kmeans_pp(cust_df: pd.DataFrame, k: int, fixed_wh: Optional[pd.DataFrame]=None) -> np.ndarray:
    """
    K-means++ style initialization, honoring pre-specified fixed warehouses.
    Returns initial centers as array shape (k, 2) [lon, lat].
    """
    points = cust_df[['lon','lat']].values
    weights = cust_df['demand'].values
    centers = []

    if fixed_wh is not None and len(fixed_wh) > 0:
        for _, row in fixed_wh.iterrows():
            centers.append([row['lon'], row['lat']])

    # how many left to place
    m = k - len(centers)
    if m <= 0:
        return np.array(centers[:k])

    # Pick first center weighted by demand
    probs = weights / (weights.sum() if weights.sum() > 0 else 1.0)
    idx = rng.choice(len(points), p=probs)
    centers.append(points[idx].tolist())

    for _ in range(m-1):
        # compute distance to nearest center
        C = np.array(centers)
        d2 = np.min((points[:,None,:] - C[None,:,:])**2, axis=2).min(axis=1)
        d2 = d2 * (weights / (weights.sum() if weights.sum() > 0 else 1.0))
        probs = d2 / (d2.sum() if d2.sum() > 0 else 1.0)
        idx = rng.choice(len(points), p=probs)
        centers.append(points[idx].tolist())

    return np.array(centers[:k])

def _assign_to_nearest(cust_df: pd.DataFrame, centers: np.ndarray) -> np.ndarray:
    """
    Assign each customer to nearest center (great-circle).
    centers: (k,2) [lon,lat]
    """
    c_lon = cust_df['lon'].values
    c_lat = cust_df['lat'].values
    k = centers.shape[0]
    # compute distances to each center
    dists = []
    for j in range(k):
        d = haversine_miles(c_lon, c_lat, centers[j,0], centers[j,1])
        dists.append(d)
    D = np.vstack(dists).T  # shape (n,k)
    return np.argmin(D, axis=1)

def _update_centers(cust_df: pd.DataFrame, assign_idx: np.ndarray, k: int, fixed_mask: Optional[np.ndarray]=None, fixed_centers: Optional[np.ndarray]=None) -> np.ndarray:
    centers = np.zeros((k,2))
    for j in range(k):
        cluster = cust_df[assign_idx == j]
        if len(cluster) == 0:
            centers[j,:] = centers[j-1,:] if j > 0 else cust_df[['lon','lat']].mean().values
        else:
            lon, lat = demand_weighted_centroid(cluster, 'lon','lat','demand')
            centers[j,:] = [lon, lat]

    # re-impose fixed centers
    if fixed_mask is not None and fixed_centers is not None:
        centers[fixed_mask] = fixed_centers[fixed_mask]
    return centers

def _service_penalty(cust_df, assign_idx, wh_df, bands: List[Dict], penalty_weight: float = 1e6):
    service_results, _ = compute_service_mix(cust_df, assign_idx, wh_df, bands)
    penalty = 0.0
    for row in service_results:
        shortfall = max(0.0, row['target'] - row['pct'])
        penalty += shortfall * penalty_weight
    return penalty

def solve_with_service_levels(
    cust_df: pd.DataFrame,
    k: int,
    service_bands: List[Dict],
    max_iters: int = 50,
    restarts: int = 50,
    fixed_wh: Optional[pd.DataFrame] = None,
    penalty_weight: float = 5e5
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Heuristic solver:
      - KMeans-like with multiple restarts
      - Objective: demand-weighted outbound distance + service shortfall penalty
      - Honors fixed warehouse locations (kept static during updates)
    Returns: (warehouses_df, assign_idx, diagnostics)
    """
    service_bands = service_bands[:]  # shallow copy
    best = None
    best_obj = np.inf
    best_diag = None

    # fixed mask/centers
    fixed_mask = None
    fixed_centers = None
    if fixed_wh is not None and len(fixed_wh) > 0:
        fixed_centers = fixed_wh[['lon','lat']].values
        # If fixed count < k, we will pad later; mask marks those fixed indices
        fixed_mask = np.zeros(k, dtype=bool)
        fixed_mask[:len(fixed_wh)] = True

    for r in range(restarts):
        centers = _init_kmeans_pp(cust_df, k, fixed_wh=fixed_wh)
        if fixed_mask is not None and fixed_centers is not None and len(fixed_centers) <= k:
            # Put fixed centers in the first positions
            centers[:len(fixed_centers),:] = fixed_centers

        prev_assign = None
        for it in range(max_iters):
            assign_idx = _assign_to_nearest(cust_df, centers)
            # Update centers except fixed ones
            centers = _update_centers(cust_df, assign_idx, k, fixed_mask=fixed_mask, fixed_centers=centers if fixed_centers is None else np.vstack((fixed_centers, np.zeros((max(0,k-len(fixed_centers)),2)))) )
            if prev_assign is not None and np.array_equal(prev_assign, assign_idx):
                break
            prev_assign = assign_idx

        wh_df = pd.DataFrame(centers, columns=['lon','lat'])
        # Objective: demand-weighted distance + penalty
        c_lon = cust_df['lon'].values
        c_lat = cust_df['lat'].values
        w_lon = wh_df['lon'].values[assign_idx]
        w_lat = wh_df['lat'].values[assign_idx]
        d = haversine_miles(c_lon, c_lat, w_lon, w_lat)
        obj = float((cust_df['demand'].values * d).sum())
        obj += _service_penalty(cust_df, assign_idx, wh_df, service_bands, penalty_weight=penalty_weight)

        if obj < best_obj:
            service_results, dists = compute_service_mix(cust_df, assign_idx, wh_df, service_bands)
            best = (wh_df.copy(), assign_idx.copy(), dists.copy())
            best_obj = obj
            best_diag = {
                'restart': r,
                'iters': it+1,
                'objective': best_obj,
                'service_results': service_results
            }

    return best[0], best[1], best_diag
