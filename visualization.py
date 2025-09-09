
import pandas as pd
import pydeck as pdk

def make_map(customers: pd.DataFrame, warehouses: pd.DataFrame, height=600):
    if len(customers) > 0:
        view_state = pdk.ViewState(
            latitude=float(customers['lat'].mean()),
            longitude=float(customers['lon'].mean()),
            zoom=4
        )
    elif len(warehouses) > 0:
        view_state = pdk.ViewState(
            latitude=float(warehouses['lat'].mean()),
            longitude=float(warehouses['lon'].mean()),
            zoom=4
        )
    else:
        view_state = pdk.ViewState(latitude=39.5, longitude=-98.35, zoom=3)

    cust_layer = pdk.Layer(
        "ScatterplotLayer",
        data=customers.rename(columns={'lon':'longitude','lat':'latitude'}),
        get_position='[longitude, latitude]',
        get_radius=4000,
        pickable=True,
        auto_highlight=True,
        get_fill_color='[0, 128, 255, 140]'
    )
    wh_layer = pdk.Layer(
        "ScatterplotLayer",
        data=warehouses.rename(columns={'lon':'longitude','lat':'latitude'}),
        get_position='[longitude, latitude]',
        get_radius=8000,
        pickable=True,
        auto_highlight=True,
        get_fill_color='[255, 64, 64, 200]'
    )
    return pdk.Deck(
        layers=[cust_layer, wh_layer],
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"text": "{name}"}
    )
