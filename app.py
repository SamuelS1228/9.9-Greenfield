
import streamlit as st
import pandas as pd
import numpy as np
from utils import normalize_service_targets, compute_service_mix, total_outbound_cost, total_inbound_cost, estimate_wh_space_and_cost
from optimization import solve_with_service_levels
from visualization import make_map

st.set_page_config(page_title="Network Solver (Service Levels) — Optimized", layout="wide")

st.title("🧭 Network Solver with Service-Level Targets — Optimized")

@st.cache_data(show_spinner=False)
def load_csv(file):
    return pd.read_csv(file)

with st.expander("📥 Data Uploads", True):
    st.markdown("**Customers CSV** must contain columns: `lon`, `lat`, `demand`. Optional: `name`.")
    cust_file = st.file_uploader("Upload customers CSV", type=["csv"], key="cust_csv")
    st.markdown("**Inbound Sources CSV (optional)** with `lon`, `lat`, and optional `share` (weights sum to 1).")
    src_file = st.file_uploader("Upload inbound sources CSV (optional)", type=["csv"], key="src_csv")
    st.markdown("**Fixed Warehouses CSV (optional)** with `lon`, `lat`, and optional `name`.")
    fixed_file = st.file_uploader("Upload fixed warehouses CSV (optional)", type=["csv"], key="fixed_csv")

    if st.button("Load Sample Data"):
        sample = pd.DataFrame({
            "lon":[-122.33,-118.24,-104.99,-87.62,-95.36,-90.20,-84.39,-80.19,-77.03,-71.06],
            "lat":[47.60,34.05,39.74,41.88,29.76,38.63,33.75,25.76,38.90,42.36],
            "demand":[500,1200,900,1300,800,600,1100,700,950,1000],
            "name":["Seattle","LA","Denver","Chicago","Houston","St. Louis","Atlanta","Miami","DC","Boston"]
        })
        st.session_state['customers'] = sample
        st.success("Loaded sample customers.")
    else:
        st.session_state['customers'] = st.session_state.get('customers', None)

with st.sidebar:
    st.header("⚙️ Solver Settings")
    k = st.number_input("Number of Warehouses (k)", min_value=1, max_value=50, value=5, step=1)

    st.subheader("📊 Cost Parameters (optional)")
    outbound_c = st.number_input("Outbound cost ($/lb-mile)", min_value=0.0, value=0.02, step=0.005, format="%.3f")
    inbound_c  = st.number_input("Inbound cost ($/lb-mile)", min_value=0.0, value=0.015, step=0.005, format="%.3f")
    space_per_lb = st.number_input("Warehouse space (sqft per lb shipped)", min_value=0.0, value=0.002, step=0.001, format="%.4f")
    wh_cost_sqft = st.number_input("Warehouse cost ($/sqft)", min_value=0.0, value=8.0, step=0.5, format="%.2f")

    st.subheader("⏱️ Service Level Targets by Distance")
    st.caption("Minimum share of total demand that must be served within each band.")
    b1 = st.slider("0–350 mi (Next-day by 7AM) — min %", 0, 100, 60, 1)
    b2 = st.slider("351–500 mi (Next-day by 10AM) — min %", 0, 100, 20, 1)
    b3 = st.slider("501–700 mi (Next-day by EOD) — min %", 0, 100, 15, 1)
    b4 = st.slider("701+ mi (2+ days) — max % (advisory)", 0, 100, 15, 1)  # advisory; enforced via earlier mins

    bands = [
        {"name": "0–350 mi (NDD 7AM)", "max_miles": 350, "min_pct": b1/100},
        {"name": "351–500 mi (NDD 10AM)", "max_miles": 500, "min_pct": b2/100},
        {"name": "501–700 mi (NDD EOD)", "max_miles": 700, "min_pct": b3/100},
        {"name": "701+ mi (2+ days)", "max_miles": None, "min_pct": 0.0},
    ]
    bands = normalize_service_targets(bands)

    st.subheader("🔁 Solver Effort")
    restarts = st.slider("Random restarts", 5, 200, 40, 5)  # lower default
    iters = st.slider("Max iterations per restart", 10, 200, 50, 10)
    penalty_weight = st.number_input("Service shortfall penalty weight", min_value=1e3, value=5e5, step=1e4, format="%.0f")
    tol = st.number_input("Early-stop tolerance (relative)", min_value=0.0, value=1e-6, step=1e-6, format="%.6f")
    early_stop = st.checkbox("Early stop when constraints met & improvement is tiny", value=True)

# Load data
if st.session_state.get('customers') is not None:
    customers = st.session_state['customers']
elif cust_file is not None:
    customers = load_csv(cust_file)
else:
    st.info("Upload a customers CSV or click **Load Sample Data** above to get started.")
    st.stop()

required_cols = {'lon','lat','demand'}
missing = required_cols - set(customers.columns)
if missing:
    st.error(f"Customers CSV missing required columns: {missing}")
    st.stop()

customers = customers.copy()
if 'name' not in customers.columns:
    customers['name'] = [f"C{i}" for i in range(len(customers))]

sources = None
if src_file is not None:
    try:
        sources = load_csv(src_file)
        if not {'lon','lat'}.issubset(sources.columns):
            st.warning("Inbound Sources missing `lon`/`lat`; ignoring sources.")
            sources = None
    except Exception as e:
        st.warning(f"Could not read sources CSV: {e}")

fixed_wh = None
if fixed_file is not None:
    try:
        fixed_wh = load_csv(fixed_file)
        if not {'lon','lat'}.issubset(fixed_wh.columns):
            st.warning("Fixed Warehouses missing `lon`/`lat`; ignoring.")
            fixed_wh = None
    except Exception as e:
        st.warning(f"Could not read fixed warehouses CSV: {e}")

# Solve (cache by inputs to avoid recomputing on minor UI changes)
@st.cache_data(show_spinner=True)
def run_solver(cust_df, k, bands, max_iters, restarts, fixed_wh, penalty_weight, tol, early_stop):
    return solve_with_service_levels(
        cust_df, int(k), bands,
        max_iters=int(max_iters), restarts=int(restarts),
        fixed_wh=fixed_wh, penalty_weight=float(penalty_weight),
        tol_improve=float(tol), early_stop_if_met=bool(early_stop)
    )

wh_df, assign_idx, diag = run_solver(customers, k, bands, iters, restarts, fixed_wh, penalty_weight, tol, early_stop)

left, right = st.columns([0.58, 0.42], gap="large")

with left:
    st.subheader("🗺️ Map")
    deck = make_map(customers.assign(type="Customer"), wh_df.assign(type="Warehouse"))
    st.pydeck_chart(deck, use_container_width=True, height=560)

    st.subheader("📊 Assignments & Distances (first 2,000 rows preview)")
    svc_results, dists = compute_service_mix(customers, assign_idx, wh_df, bands)
    assigned = customers.copy()
    assigned['warehouse_id'] = assign_idx
    assigned['distance_miles'] = dists
    preview = assigned.head(2000)  # large tables can bog down Streamlit; provide preview + download
    st.dataframe(preview, use_container_width=True, hide_index=True)

with right:
    st.subheader("✅ Service Level Results")
    svc_table = pd.DataFrame([
        {"Band": r['name'], "Actual %": f"{100*r['pct']:.1f}%", "Target %": f"{100*r['target']:.1f}%", "Met?": "✅" if r['met'] else "❌"}
        for r in svc_results
    ])
    st.dataframe(svc_table, use_container_width=True, hide_index=True)

    st.subheader("💵 Cost Summary (proxy)")
    ob_cost = total_outbound_cost(customers, assign_idx, wh_df, outbound_c)
    ib_cost = total_inbound_cost(customers, assign_idx, wh_df, sources, inbound_c)
    wh_cost, wh_loads, sqft = estimate_wh_space_and_cost(customers, assign_idx, len(wh_df), space_per_lb, wh_cost_sqft)
    total_cost = ob_cost + ib_cost + wh_cost

    st.metric("Total Estimated Cost", f"${total_cost:,.0f}")
    st.caption("Breakdown")
    st.write({
        "Outbound ($)": round(ob_cost,2),
        "Inbound ($)": round(ib_cost,2),
        "Warehousing ($)": round(wh_cost,2)
    })

    st.subheader("🧪 Solver Diagnostics")
    st.write({
        "best_restart": diag['restart'],
        "iterations_used": diag['iters'],
        "objective": round(diag['objective'],3)
    })
    st.caption("Tip: If constraints aren't met, increase **Random restarts** and **Penalty weight**, or increase **k**.")

st.divider()
st.markdown("### 📥 Download Outputs")
out_assign = assigned  # full set
out_WH = wh_df.reset_index().rename(columns={'index':'warehouse_id'})
st.download_button("Download Assignments CSV", out_assign.to_csv(index=False).encode(), "assignments.csv", "text/csv")
st.download_button("Download Warehouses CSV", out_WH.to_csv(index=False).encode(), "warehouses.csv", "text/csv")
