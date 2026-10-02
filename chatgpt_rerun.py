from pathlib import Path
import hashlib
import json
import platform
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

ROOT = Path(".")
DATA = ROOT / "notebooks/data/processed/feat.parquet"
MANIFEST = ROOT / "notebooks/data/processed/manifest.json"
OUT = ROOT / "chatgpt_rerun_results"
OUT.mkdir(exist_ok=True)

def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

print("Python:", sys.version)
print("Platform:", platform.platform())
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("Data:", DATA)

man = json.loads(MANIFEST.read_text(encoding="utf-8"))
actual_md5 = md5_file(DATA)
print("Manifest MD5:", man["md5"])
print("Actual MD5:  ", actual_md5)
assert actual_md5 == man["md5"], "feat.parquet does not match manifest"

feat = pd.read_parquet(DATA)
TIME_COL = man["time_col"]
UNIT_COL = man["unit_col"]
TARGET_COL = man["target_col"]
EXOG_COLS = man["exog_cols"]
HORIZONS = man["horizons"]
SEASON = man["season"]
TEST_START = pd.Timestamp(man["test_start"])

feat[TIME_COL] = pd.to_datetime(feat[TIME_COL])
print(f"Loaded {len(feat):,} rows x {len(feat.columns)} columns, {feat[UNIT_COL].nunique()} units")
print("Range:", feat[TIME_COL].min(), "to", feat[TIME_COL].max())

# This mirrors the feature selection and chronological split used in
# Notebook_D_Common_Models_RQ2_RQ3_COMMENT_PRESERVING_NO_HOURS_FIX.
lag_cols = [c for c in feat.columns if c.startswith("y_lag") or c.startswith("y_ma") or c.startswith("y_sd") or c == "y_diff1"]
exog_now = [c for c in EXOG_COLS if c in feat.columns]
exog_lag = [c for c in feat.columns if c.endswith("_lag_season")]
cal_cols = [c for c in ["hr", "dow", "mon", "doy"] if c in feat.columns]
feat["unit_code"] = feat[UNIT_COL].astype("category").cat.codes
X_cols = [TARGET_COL] + lag_cols + exog_now + exog_lag + cal_cols + ["unit_code"]
data = feat.dropna(subset=X_cols + [f"y_h{h}" for h in HORIZONS]).copy()

train = data[data[TIME_COL] < TEST_START].copy()
test = data[data[TIME_COL] >= TEST_START].copy()
assert len(train) > 0 and len(test) > 0
print(f"Usable rows: {len(data):,}; train: {len(train):,}; test: {len(test):,}")
print("Features:", X_cols)

def seasonal_naive(frame, h):
    return frame[f"{TARGET_COL}"].to_numpy() if not (h <= SEASON and f"y_lag{SEASON}" in frame.columns) else frame[f"y_lag{SEASON}"].to_numpy()

def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))

rows = []
SEEDS = [0, 1, 2, 3, 4]

for h in HORIZONS:
    ytr = train[f"y_h{h}"].to_numpy()
    yte = test[f"y_h{h}"].to_numpy()

    pers = test[TARGET_COL].to_numpy()
    seas = seasonal_naive(test, h)
    rows.append({"model":"Persistence","h":h,"seed":0,
                 "MAE":mean_absolute_error(yte,pers),"RMSE":rmse(yte,pers),
                 "n_train":len(train),"n_test":len(test)})
    rows.append({"model":"SeasonalNaive","h":h,"seed":0,
                 "MAE":mean_absolute_error(yte,seas),"RMSE":rmse(yte,seas),
                 "n_train":len(train),"n_test":len(test)})

    for seed in SEEDS:
        for name in ["Ridge","RandomForest","LightGBM"]:
            if name == "Ridge":
                model = Ridge(alpha=1.0)
            elif name == "RandomForest":
                model = RandomForestRegressor(
                    n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed
                )
            else:
                model = LGBMRegressor(
                    n_estimators=800, learning_rate=0.03, num_leaves=31,
                    subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1
                )
            model.fit(train[X_cols], ytr)
            pred = model.predict(test[X_cols])
            rows.append({
                "model":name,"h":h,"seed":seed,
                "MAE":mean_absolute_error(yte,pred),"RMSE":rmse(yte,pred),
                "n_train":len(train),"n_test":len(test)
            })
            print(f"h={h} {name} seed={seed}: MAE={rows[-1]['MAE']:.6f} RMSE={rows[-1]['RMSE']:.6f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "raw_results.csv", index=False)

summary = (
    res[res.model.isin(["Ridge","RandomForest","LightGBM"])]
    .groupby(["model","h"])[["MAE","RMSE"]]
    .agg(["mean","std"]).round(6)
)
summary.to_csv(OUT / "model_summary.csv")

summary_plain = summary.copy()
print("\n=== MODEL SUMMARY ===")
print(summary_plain.to_string())

# Exact H1/H6 result lines for the three trained models.
for h in HORIZONS:
    print(f"\nH={h}")
    for model in ["Ridge","RandomForest","LightGBM"]:
        s = res[(res.model==model)&(res.h==h)]
        print(model, "MAE mean/std =", s.MAE.mean(), s.MAE.std(ddof=1),
              "| RMSE mean/std =", s.RMSE.mean(), s.RMSE.std(ddof=1))

# A compact provenance file to make the rerun auditable.
meta = {
    "repository": "noobball111/ADYm_2",
    "branch": "chatgpt-rerun-20261002",
    "manifest_md5": man["md5"],
    "actual_md5": actual_md5,
    "test_start": str(TEST_START),
    "horizons": HORIZONS,
    "n_total": int(len(feat)),
    "n_usable": int(len(data)),
    "n_train": int(len(train)),
    "n_test": int(len(test)),
    "features": X_cols,
    "seeds": SEEDS,
    "protocol_note": "Faithful rerun of the model-selection/split logic in Notebook_D..._NO_HOURS_FIX."
}
(OUT / "rerun_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("\nRERUN COMPLETE")
