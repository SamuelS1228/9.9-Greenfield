
import math
import numpy as np
import pandas as pd

EARTH_RADIUS_MI = 3958.7613

def haversine_miles(lon1, lat1, lon2, lat2):
    """
    Vectorized haversine distance in miles.
    Inputs can be numpy arrays / pandas series.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_MI * c

def demand_weighted_centroid(df, lon_col="lon", lat_col="lat", w_col="demand"):
    w = df[w_col].values
    if w.sum() == 0:
        return df[lon_col].mean(), df[lat_col].mean()
    return (df[lon_col].values * w).sum() / w.sum(), (df[lat_col].values * w).sum() / w.sum()

def normalize_service_targets(sl_bands):
    """
    Ensure bands are sorted and normalized into a consistent list of dicts:
    [{'name': '0-350 (7AM NDD)', 'min_pct': 0.8, 'max_miles': 350}, ...]
    The last band can have max_miles = None to represent "701+" (no cap).
    """
    # Sort by max_miles, None last
    bands = sorted(sl_bands, key=lambda x: (float('inf') if x['max_miles'] is None else x['max_miles']))
    return bands

def compute_service_mix(cust_df, assign_idx, wh_df, bands):
    """
    Compute % of total demand delivered within each distance band.
    cust_df: customers with columns lon, lat, demand
    assign_idx: numpy array of length len(cust_df) with warehouse index per customer
    wh_df: warehouses with lon, lat
    bands: list of dicts with 'max_miles' possibly None
    Returns: list of dicts [{'name', 'pct', 'target', 'met', 'total_demand_in_band'}...], also distances per customer
    """
    c_lon = cust_df['lon'].values
    c_lat = cust_df['lat'].values
    w_lon = wh_df['lon'].values[assign_idx]
    w_lat = wh_df['lat'].values[assign_idx]
    dists = haversine_miles(c_lon, c_lat, w_lon, w_lat)
    demand = cust_df['demand'].values
    total = demand.sum() if demand.sum() > 0 else 1.0

    # Prepare band edges
    edges = [b['max_miles'] for b in bands]
    # Convert to concrete numeric edges for binning; np.digitize expects increasing bins without None
    finite_edges = [e for e in edges if e is not None]
    # Bin: values <= edge go to that band; values > last edge go to final band
    bin_idx = np.digitize(dists, finite_edges, right=True)
    # If there is a trailing None band, any value > last finite edge maps there automatically

    results = []
    for i, b in enumerate(bands):
        mask = bin_idx == i
        band_demand = demand[mask].sum()
        pct = band_demand / total
        target = b.get('min_pct', 0.0)
        results.append({
            'name': b['name'],
            'pct': pct,
            'target': target,
            'met': pct >= target,
            'total_demand_in_band': band_demand
        })
    return results, dists

def check_service_constraints(service_results):
    return all(row['met'] for row in service_results)

def total_outbound_cost(cust_df, assign_idx, wh_df, outbound_cost_per_lb_mile):
    w_lon = wh_df['lon'].values[assign_idx]
    w_lat = wh_df['lat'].values[assign_idx]
    dists = haversine_miles(cust_df['lon'].values, cust_df['lat'].values, w_lon, w_lat)
    return float((cust_df['demand'].values * dists * outbound_cost_per_lb_mile).sum())

def estimate_wh_space_and_cost(cust_df, assign_idx, wh_df, space_per_lb, wh_cost_per_sqft):
    """
    Very simple warehousing cost proxy:
    - For each warehouse, total assigned demand determines required space proxy.
    - Multiply by wh_cost_per_sqft to estimate annual cost (proxy).
    """
    demand = cust_df['demand'].values
    k = len(wh_df)
    wh_loads = np.zeros(k)
    for i, a in enumerate(assign_idx):
        wh_loads[a] += demand[i]
    sqft_needed = wh_loads * space_per_lb
    return float((sqft_needed * wh_cost_per_sqft).sum()), wh_loads, sqft_needed

def total_inbound_cost(cust_df, assign_idx, wh_df, sources_df, inbound_cost_per_lb_mile):
    """
    Simple inbound model:
      - Distribute each warehouse's assigned demand back to inbound sources proportionally
        to source 'share' (or equal if not provided) to compute inbound lane miles.
      - Cost = sum over sources and warehouses (volume * source->warehouse distance * cost per lb-mile)
    """
    demand = cust_df['demand'].values
    k = len(wh_df)
    wh_loads = np.zeros(k)
    for i, a in enumerate(assign_idx):
        wh_loads[a] += demand[i]

    if sources_df is None or len(sources_df) == 0:
        return 0.0

    if 'share' in sources_df.columns:
        shares = sources_df['share'].values
        if shares.sum() <= 0:
            shares = np.ones(len(sources_df)) / len(sources_df)
        else:
            shares = shares / shares.sum()
    else:
        shares = np.ones(len(sources_df)) / len(sources_df)

    cost = 0.0
    for wi in range(k):
        wlon, wlat = wh_df.iloc[wi][['lon','lat']].values
        for si, srow in sources_df.iterrows():
            slon, slat = srow['lon'], srow['lat']
            dist = haversine_miles(wlon, wlat, slon, slat)
            vol = wh_loads[wi] * shares[si]
            cost += vol * dist * inbound_cost_per_lb_mile
    return float(cost)
