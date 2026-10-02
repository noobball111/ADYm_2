from pathlib import Path
import hashlib, json, platform, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

DATA=Path("notebooks/data/processed/feat.parquet")
MANIFEST=Path("notebooks/data/processed/manifest.json")
OUT=Path("chatgpt_rerun_results"); OUT.mkdir(exist_ok=True)

def md5_file(p):
    h=hashlib.md5()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

man=json.loads(MANIFEST.read_text())
actual_md5=md5_file(DATA)
assert actual_md5==man["md5"], "Parquet does not match manifest"
feat=pd.read_parquet(DATA)
print("Python:",sys.version)
print("Platform:",platform.platform())
print("pandas:",pd.__version__,"numpy:",np.__version__)
print("Manifest MD5:",man["md5"])
print("Actual MD5:  ",actual_md5)
print("Shape:",feat.shape)

UNIT_COL,TIME_COL,TARGET_COL="plant","ts","cf"
EXOG_COLS=["GHI","DNI","cloud","temp"]
HORIZONS=[1,6]; SEASON=24; TEST_START=pd.Timestamp("2006-10-01")
SEEDS=[0,1,2,3,4]
feat[TIME_COL]=pd.to_datetime(feat[TIME_COL])

lag_cols=[c for c in feat.columns if c.startswith("cf_lag") or c.startswith("y_lag") or c.startswith("y_ma") or c.startswith("y_sd") or c=="y_diff1"]
exog_now=[c for c in EXOG_COLS if c in feat.columns]
exog_lag=[c for c in feat.columns if c.endswith("_lag_season")]
cal_cols=[c for c in ["hr","dow","mon","doy"] if c in feat.columns]
feat["unit_code"]=feat[UNIT_COL].astype("category").cat.codes
X_cols=[TARGET_COL]+lag_cols+exog_now+exog_lag+cal_cols+["unit_code"]
target_cols={1:"y_1h",6:"y_6h"}
assert all(c in feat.columns for c in target_cols.values())
data=feat.dropna(subset=X_cols+list(target_cols.values())).copy()
train=data[data[TIME_COL]<TEST_START].copy()
test=data[data[TIME_COL]>=TEST_START].copy()
print("Features:",X_cols)
print(f"Usable={len(data):,}; train={len(train):,}; test={len(test):,}")
print("Train end:",train[TIME_COL].max(),"Test start:",test[TIME_COL].min())

def rmse(y,p): return float(np.sqrt(mean_squared_error(y,p)))
def seasonal_pred(frame): return frame["cf_lag24"].to_numpy()

def make_model(name,seed):
    if name=="Ridge": return Ridge(alpha=1.0)
    if name=="RandomForest": return RandomForestRegressor(n_estimators=300,min_samples_leaf=5,n_jobs=-1,random_state=seed)
    if name=="LightGBM": return LGBMRegressor(n_estimators=800,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=seed,verbose=-1)
    raise ValueError(name)

rows=[]
for h in HORIZONS:
    Y=target_cols[h]; yte=test[Y].to_numpy()
    pers=test[TARGET_COL].to_numpy(); seas=seasonal_pred(test)
    rows.append({"model":"Persistence","h":h,"seed":0,"MAE":mean_absolute_error(yte,pers),"RMSE":rmse(yte,pers)})
    rows.append({"model":"SeasonalNaive","h":h,"seed":0,"MAE":mean_absolute_error(yte,seas),"RMSE":rmse(yte,seas)})
    for seed in SEEDS:
        for name in ["Ridge","RandomForest","LightGBM"]:
            m=make_model(name,seed).fit(train[X_cols],train[Y])
            p=m.predict(test[X_cols])
            rows.append({"model":name,"h":h,"seed":seed,"MAE":mean_absolute_error(yte,p),"RMSE":rmse(yte,p)})
            print(f"h={h} {name} seed={seed} MAE={rows[-1]['MAE']:.6f} RMSE={rows[-1]['RMSE']:.6f}",flush=True)

res=pd.DataFrame(rows)
res.to_csv(OUT/"raw_results.csv",index=False)
summary=res[res.model.isin(["Ridge","RandomForest","LightGBM"])].groupby(["model","h"])[["MAE","RMSE"]].agg(["mean","std"]).round(6)
summary.to_csv(OUT/"model_summary.csv")
print("\n=== SUMMARY ===\n",summary.to_string())

pred_rows=[]
for h in HORIZONS:
    Y=target_cols[h]
    for name in ["Ridge","RandomForest","LightGBM"]:
        m=make_model(name,0).fit(train[X_cols],train[Y]); p=m.predict(test[X_cols])
        pred_rows.append(pd.DataFrame({"ts":test[TIME_COL].to_numpy(),"plant":test[UNIT_COL].to_numpy(),"actual":test[Y].to_numpy(),"pred":p,"model":name,"h":h}))
pd.concat(pred_rows,ignore_index=True).to_csv(OUT/"predictions_seed0.csv",index=False)
meta={"repository":"noobball111/ADYm_2","branch":"chatgpt-rerun-20261002","manifest_md5":man["md5"],"actual_md5":actual_md5,"shape":list(feat.shape),"usable":len(data),"train":len(train),"test":len(test),"features":X_cols,"target_cols":target_cols,"test_start":str(TEST_START.date()),"seeds":SEEDS}
(OUT/"rerun_metadata.json").write_text(json.dumps(meta,indent=2))
print("\nRERUN COMPLETE")
