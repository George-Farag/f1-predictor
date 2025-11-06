from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
flat = pd.read_csv(ROOT / "predictions_2024_flat.csv")

# keep rows that have an actual finishing position
flat = flat.dropna(subset=["finish_pos"]).copy()

# ---------- overall metrics ----------
mae = (flat["pred_pos"] - flat["finish_pos"]).abs().mean()
print(f"MAE (2024): {mae:.3f}")

# Top-3 / Top-10 hit rate per race
def topk_hit(df, k):
    pred_top = set(df.sort_values("pred_pos").head(k)["driverRef"])
    act_top  = set(df.sort_values("finish_pos").head(k)["driverRef"])
    return len(pred_top & act_top) / k

by_race = flat.groupby("raceId", group_keys=False)
top3  = by_race.apply(lambda g: topk_hit(g, 3)).mean()
top10 = by_race.apply(lambda g: topk_hit(g,10)).mean()
print(f"Top-3 hit rate:  {top3:.3f}")
print(f"Top-10 hit rate: {top10:.3f}")

# Spearman rank corr per race
def spearman(g):
    g = g.copy()
    g = g.sort_values("pred_pos")
    g["rank_pred"] = np.arange(1, len(g)+1)
    g = g.sort_values("finish_pos")
    g["rank_act"] = np.arange(1, len(g)+1)
    return g[["rank_pred","rank_act"]].corr(method="spearman").iloc[0,1]

spearman_r = by_race.apply(spearman)
print("Mean Spearman (per race):", round(spearman_r.mean(),3))

# ---------- plots ----------
# 1) scatter Pred vs Actual
plt.figure(figsize=(8,6))
plt.scatter(flat["finish_pos"], flat["pred_pos"], s=10)
lims = [1, 20]
plt.plot(lims, lims, linestyle="--")
plt.xlim(lims); plt.ylim(lims)
plt.xlabel("Actual finishing position")
plt.ylabel("Predicted finishing position")
plt.title("Predicted vs Actual (2024)")
plt.tight_layout()
plt.savefig(ROOT / "plot_pred_vs_actual.png", dpi=150)

# 2) bar – Spearman by race (sorted)
order = spearman_r.sort_values(ascending=False)
plt.figure(figsize=(10,5))
order.reset_index(drop=True).plot(kind="bar")
plt.ylabel("Spearman rank corr")
plt.title("Per-race rank correlation (2024)")
plt.tight_layout()
plt.savefig(ROOT / "plot_spearman_by_race.png", dpi=150)

print("Saved plots:")
print((ROOT / "plot_pred_vs_actual.png").resolve())
print((ROOT / "plot_spearman_by_race.png").resolve())
