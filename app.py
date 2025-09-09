
import streamlit as st
import pandas as pd
import numpy as np
from utils import normalize_service_targets, compute_service_mix
from optimization import solve_with_service_levels
from visualization import make_map
from utils import total_outbound_cost, total_inbound_cost, estimate_wh_space_and_cost

st.set_page_config(page_title="Network Solver (Service Levels)", layout="wide")

st.title("🧭 Network Solver with Service-Level Targets")

with st.expander("📥 Data Uploads", True):
    st.markdown("**Customers CSV** must contain columns: `lon`, `lat`, `demand`. Optional: `name`.")
    cust_file = st.file_uploader("Upload customers CSV", type=["csv"], key="cust_csv")
    st.markdown("**Inbound Sources CSV (optional)** with `lon`, `lat`, and optional `share` (weights sum to 1).")
    src_file = st.file_uploader("Upload inbound sources CSV (optional)", type=["csv"], key="src_csv")
    st.markdown("**Fixed Warehouses CSV (optional)** with `lon`, `lat`, and optional `name`.")
    fixed_file = st.file_uploader("Upload fixed warehouses CSV (optional)", type=["csv"], key="fixed_csv")

    if st.button("Load Sample Data"):
        import io
        sample = pd.DataFrame({
            "lon":[-122.33,-118.24,-104.99,-87.62,-95.36,-90.20,-84.39,-80.19,-77.03,-71.06],
            "lat":[47.60,34.05,39.74,41.88,29.76,38.63,33.75,25.76,38.90,42.36],
            "demand":[500,1200,900,1300,800,600,1100,700,950,1000],
            "name":["Seattle","LA","Denver","Chicago","Houston","St. Louis","Atlanta","Miami","DC","Boston"]
        })
        st.session_state['customers'] = sample
        st.success("Loaded sample customers.")
    else:
        st.session_state['customers'] = None

with st.sidebar:
    st.header("⚙️ Solver Settings")
    k = st.number_input("Number of Warehouses (k)", min_value=1, max_value=50, value=5, step=1)

    st.subheader("📊 Cost Parameters (optional)")
    outbound_c = st.number_input("Outbound cost ($/lb-mile)", min_value=0.0, value=0.02, step=0.005, format="%.3f")
    inbound_c  = st.number_input("Inbound cost ($/lb-mile)", min_value=0.0, value=0.015, step=0.005, format="%.3f")
    space_per_lb = st.number_input("Warehouse space (sqft per lb shipped)", min_value=0.0, value=0.002, step=0.001, format="%.4f")
    wh_cost_sqft = st.number_input("Warehouse cost ($/sqft)", min_value=0.0, value=8.0, step=0.5, format="%.2f")

    st.subheader("⏱️ Service Level Targets by Distance")
    st.caption("These represent the **minimum share of total demand** that must be served within each distance band.")
    b1 = st.slider("0–350 mi (Next-day by 7AM) — min %", 0, 100, 60, 1)
    b2 = st.slider("351–500 mi (Next-day by 10AM) — min %", 0, 100, 20, 1)
    b3 = st.slider("501–700 mi (Next-day by EOD) — min %", 0, 100, 15, 1)
    b4 = st.slider("701+ mi (2+ days) — max % (converted to min of complement)", 0, 100, 15, 1)
    # Convert last to a "min within <=700" by limiting tail share; but we'll keep explicit band with min% set to 1 - max_tail?
    # We'll set min% for 701+ as 0 to avoid forcing volume into tail; user uses earlier bands to constrain.
    bands = [
        {"name": "0–350 mi (NDD 7AM)", "max_miles": 350, "min_pct": b1/100},
        {"name": "351–500 mi (NDD 10AM)", "max_miles": 500, "min_pct": b2/100},
        {"name": "501–700 mi (NDD EOD)", "max_miles": 700, "min_pct": b3/100},
        {"name": "701+ mi (2+ days)", "max_miles": None, "min_pct": 0.0}, # advisory, tail discouraged via earlier mins
    ]
    bands = normalize_service_targets(bands)

    st.subheader("🔁 Solver Effort")
    restarts = st.slider("Random restarts", 5, 200, 60, 5)
    iters = st.slider("Max iterations per restart", 10, 200, 60, 10)
    penalty_weight = st.number_input("Service shortfall penalty weight", min_value=1e3, value=5e5, step=1e4, format="%.0f")

# Load dataframes from uploads / session
customers = None
if st.session_state.get('customers') is not None:
    customers = st.session_state['customers']
else:
    if cust_file is not None:
        customers = pd.read_csv(cust_file)
if customers is None:
    st.info("Upload a customers CSV or click **Load Sample Data** above to get started.")
    st.stop()

required_cols = {'lon','lat','demand'}
missing = required_cols - set(customers.columns)
if missing:
    st.error(f"Customers CSV missing required columns: {missing}")
    st.stop()

customers = customers.copy()
customers = customers[['lon','lat','demand'] + [c for c in customers.columns if c not in ['lon','lat','demand']]]
customers['name'] = customers.get('name', pd.Series([f"C{i}" for i in range(len(customers))]))

sources = None
if src_file is not None:
    try:
        sources = pd.read_csv(src_file)
        if not {'lon','lat'}.issubset(set(sources.columns)):
            st.warning("Inbound Sources missing `lon`/`lat`; ignoring sources.")
            sources = None
    except Exception as e:
        st.warning(f"Could not read sources CSV: {e}")

fixed_wh = None
if fixed_file is not None:
    try:
        fixed_wh = pd.read_csv(fixed_file)
        if not {'lon','lat'}.issubset(set(fixed_wh.columns)):
            st.warning("Fixed Warehouses missing `lon`/`lat`; ignoring.")
            fixed_wh = None
    except Exception as e:
        st.warning(f"Could not read fixed warehouses CSV: {e}")

# Solve
with st.spinner("Solving..."):
    wh_df, assign_idx, diag = solve_with_service_levels(
        customers, int(k), bands,
        max_iters=int(iters), restarts=int(restarts),
        fixed_wh=fixed_wh, penalty_weight=float(penalty_weight)
    )

# KPI + service table
left, right = st.columns([0.6, 0.4], gap="large")

with left:
    st.subheader("🗺️ Map")
    deck = make_map(customers.assign(type="Customer"), wh_df.assign(type="Warehouse"))
    st.pydeck_chart(deck, use_container_width=True, height=600)

    st.subheader("📊 Assignments & Distances")
    # compute service mix and distances
    service_results, dists = compute_service_mix(customers, assign_idx, wh_df, bands)
    assigned = customers.copy()
    assigned['warehouse_id'] = assign_idx
    assigned['distance_miles'] = dists
    st.dataframe(assigned, use_container_width=True, hide_index=True)

with right:
    st.subheader("✅ Service Level Results")
    svc_table = pd.DataFrame([
        {"Band": r['name'], "Actual %": f"{100*r['pct']:.1f}%", "Target %": f"{100*r['target']:.1f}%", "Met?": "✅" if r['met'] else "❌"}
        for r in service_results
    ])
    st.dataframe(svc_table, use_container_width=True, hide_index=True)

    st.subheader("💵 Cost Summary (proxy)")
    ob_cost = total_outbound_cost(customers, assign_idx, wh_df, outbound_c)
    ib_cost = total_inbound_cost(customers, assign_idx, wh_df, sources, inbound_c)
    wh_cost, wh_loads, sqft = estimate_wh_space_and_cost(customers, assign_idx, wh_df, space_per_lb, wh_cost_sqft)
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
    st.caption("If constraints aren't met, increase **Random restarts** and **Penalty weight**, or increase **k**.")

st.divider()
st.markdown("### 📥 Download Outputs")
out_assign = customers.copy()
out_assign['warehouse_id'] = assign_idx
out_assign['distance_miles'] = dists
out_WH = wh_df.reset_index().rename(columns={'index':'warehouse_id'})
st.download_button("Download Assignments CSV", out_assign.to_csv(index=False).encode(), "assignments.csv", "text/csv")
st.download_button("Download Warehouses CSV", out_WH.to_csv(index=False).encode(), "warehouses.csv", "text/csv")
