# Predict 2024 finishing positions from your local CSVs
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# === paths ===
ROOT = Path(r"C:\Users\jamja\Downloads\f1 data")  # <-- change if different
base = pd.read_csv(ROOT / "base_all_years.csv")

# --- keep only rows with a numeric finishing position (finished) ---
base = base[~base["finish_pos"].isna()].copy()

# --- feature engineering (rolling history) ---
base = base.sort_values(["driverId", "year", "round"])

# driver recent form
base["drv_last5"]  = (
    base.groupby("driverId")["finish_pos"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
)
base["drv_last10"] = (
    base.groupby("driverId")["finish_pos"].transform(lambda s: s.shift().rolling(10, min_periods=1).mean())
)

# team recent form
base["team_last5"] = (
    base.groupby("constructorId")["finish_pos"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
)

# circuit history
base["drv_circuit_hist"] = (
    base.groupby(["driverId","circuitId"])["finish_pos"].transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
)
base["team_circuit_hist"] = (
    base.groupby(["constructorId","circuitId"])["finish_pos"].transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
)

# qualifying position if present
if "q_pos" not in base.columns:
    # add from qualifying.csv if you downloaded it
    qfile_csv = ROOT / "qualifying.csv"
    if qfile_csv.exists():
        q = pd.read_csv(qfile_csv)
        q = q.groupby(["raceId","driverId"])["position"].min().rename("q_pos")
        base = base.merge(q, on=["raceId","driverId"], how="left")
else:
    pass

# grid==0 means pit lane; keep as 0 (model can learn it)
# build feature list (drop missing columns safely)
feat_names = [c for c in [
    "grid","q_pos","drv_last5","drv_last10","team_last5",
    "drv_circuit_hist","team_circuit_hist"
] if c in base.columns]

# fill NA created by shifts with group means or global median
for c in feat_names:
    base[c] = base[c].fillna(base[c].median())

# --- split train vs 2024 holdout ---
train = base[base["year"] < 2024].copy()
hold  = base[base["year"] == 2024].copy()

X = train[feat_names].values
y = train["finish_pos"].values.astype(float)

# quick internal check
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
model.fit(X_tr, y_tr)
pred_te = model.predict(X_te)
print("MAE (val):", round(mean_absolute_error(y_te, pred_te), 3))

# --- predict 2024 ---
H = hold[feat_names].values
hold["pred_pos"] = model.predict(H)

# clip and round to plausible race positions
hold["pred_pos"] = np.clip(hold["pred_pos"], 1, 20)
hold["pred_pos"] = hold["pred_pos"].round(2)

# save flat predictions
pred_path = ROOT / "predictions_2024_flat.csv"
cols_keep = ["year","round","raceId","circuitId","driverRef","team","grid"]
cols_keep = [c for c in cols_keep if c in hold.columns]
hold[cols_keep + ["pred_pos","finish_pos"]].to_csv(pred_path, index=False)

# save per-race order (sorted by predicted finishing position)
ordered = (
    hold.sort_values(["raceId","pred_pos"])
        .groupby("raceId", as_index=False)
        .apply(lambda g: g[cols_keep + ["pred_pos","finish_pos"]].reset_index(drop=True))
)
ordered_path = ROOT / "predicted_order_by_race_2024.csv"
ordered.to_csv(ordered_path, index=False)

print("Saved:")
print(pred_path.resolve())
print(ordered_path.resolve())
