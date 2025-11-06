from pathlib import Path
import pandas as pd

# <<< change only if your folder is different >>>
ROOT = Path(r"C:\Users\jamja\Downloads\f1 data")

def read_any(stem: str) -> pd.DataFrame:
    """Read CSV or Excel with the given stem (e.g., 'drivers')."""
    for ext in (".csv", ".CSV", ".xlsx", ".xls"):
        p = ROOT / f"{stem}{ext}"
        if p.exists():
            if p.suffix.lower() == ".csv":
                return pd.read_csv(p)
            else:
                return pd.read_excel(p)
    raise FileNotFoundError(f"Missing file for {stem} (csv/xlsx) in {ROOT}")

# ---- load core tables ----
drivers      = read_any("drivers")
constructors = read_any("constructors")
races        = read_any("races")
results      = read_any("results")
status       = read_any("status")         # DNF/DQ text
# optional
have_q       = (ROOT/"qualifying.csv").exists() or (ROOT/"qualifying.xlsx").exists()
qualifying   = read_any("qualifying") if have_q else None

# ---- clean join: one row per (race, driver) ----
base = (
    results
      .merge(races[["raceId","year","round","circuitId"]], on="raceId", how="left")
      .merge(drivers[["driverId","driverRef","code","forename","surname","nationality"]], on="driverId", how="left")
      .merge(constructors[["constructorId","name"]].rename(columns={"name":"team"}), on="constructorId", how="left")
      .merge(status[["statusId","status"]], on="statusId", how="left")
)

# numeric finishing position (drops R/DQ/NC if you filter later)
base["finish_pos"] = pd.to_numeric(base["position"], errors="coerce")
base["grid"] = base["grid"].astype("Int64")  # 0 = pit-lane start

# add qualifying position if you downloaded it
if qualifying is not None:
    qpos = (qualifying.groupby(["raceId","driverId"])["position"]
            .min().rename("q_pos"))
    base = base.merge(qpos, on=["raceId","driverId"], how="left")

# ---- save ----
base_all = ROOT / "base_all_years.csv"
base_24  = ROOT / "base_2024.csv"
base.to_csv(base_all, index=False)
base[base["year"] == 2024].to_csv(base_24, index=False)

print("✅ Saved:")
print(base_all.resolve())
print(base_24.resolve())
print("\nPeek 2024:")
print(base.loc[base["year"] == 2024,
               ["year","round","driverRef","team","grid","finish_pos","status"]]
      .head(12))
