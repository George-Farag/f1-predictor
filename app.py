import pandas as pd
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent
flat = pd.read_csv(ROOT / "predictions_2024_flat.csv")

# NEW: read race & circuit metadata
races = pd.read_csv(ROOT / "races.csv")              # columns include: raceId, year, round, circuitId, name (Grand Prix name), date
circuits = pd.read_csv(ROOT / "circuits.csv")        # columns include: circuitId, name, location, country

# build nice labels for 2024
meta = (races[races["year"] == 2024]
        .merge(circuits[["circuitId","name","location","country"]],
               on="circuitId", how="left", suffixes=("_race","_circuit")))
meta["label"] = meta.apply(
    lambda r: f"{int(r['round']):02d} | {r['name_race']} — {r['name_circuit']} ({r['location']}, {r['country']})",
    axis=1
)
meta = meta.sort_values("round")[["raceId","label"]]

st.set_page_config(page_title="F1 Predictions 2024", layout="wide")
st.title("F1 2024 – Predicted Results vs Actual")

choice = st.selectbox("Select race", meta["label"])
race_id = meta.loc[meta["label"].eq(choice), "raceId"].iloc[0]

g = flat.loc[flat["raceId"].eq(race_id)].copy()

left, right = st.columns(2)
pred = g.sort_values("pred_pos")[["driverRef","team","grid","pred_pos"]].reset_index(drop=True); pred.index += 1
act  = g.sort_values("finish_pos")[["driverRef","team","grid","finish_pos"]].dropna().reset_index(drop=True); act.index += 1
left.subheader("Predicted order"); left.dataframe(pred, use_container_width=True)
right.subheader("Actual order");   right.dataframe(act,  use_container_width=True)

st.subheader("Predicted vs Actual scatter (this race)")
st.scatter_chart(g[["finish_pos","pred_pos"]].rename(columns={"finish_pos":"x","pred_pos":"y"}))
