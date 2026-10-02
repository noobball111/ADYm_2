from pathlib import Path
import json, hashlib, platform, sys
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

DATA=Path("notebooks/data/processed/feat.parquet")
MAN=Path("notebooks/data/processed/manifest.json")
OUT=Path("chatgpt_fresh_rerun"); OUT.mkdir(exist_ok=True)

def md5(p):
 h=hashlib.md5()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

m=json.loads(MAN.read_text())
assert md5(DATA)==m["md5"]
df=pd.read_parquet(DATA)
df[m["time_col"]]=pd.to_datetime(df[m["time_col"]])
T=m["time_col"]; U=m["unit_col"]; Y=m["target_col"]
test_start=pd.Timestamp(m["test_start"])
H=m["horizons"]
df["unit_code"]=df[U].astype("category").cat.codes

# Fresh descriptive analysis from the repository's current feat.parquet.
out={}
out["shape"]=[int(df.shape[0]),int(df.shape[1])]
out["units"]=int(df[U].nunique())
out["time_min"]=str(df[T].min()); out["time_max"]=str(df[T].max())
out["mean_by_month"]=df.groupby(df[T].dt.month)[Y].mean().to_dict()
for c in ["GHI","DNI","temp"]:
 if c in df:
  out["corr_"+c]=float(df[[Y,c]].corr(method="spearman").iloc[0,1])
out["ghi_decile"]=df.assign(decile=pd.qcut(df["GHI"],10,duplicates="drop")).groupby("decile",observed=True)[Y].mean().tolist()

# Feature columns present in the repository feature table.
lag=[c for c in df.columns if c.startswith("y_lag") or c.startswith("y_ma") or c.startswith("y_sd") or c=="y_diff1"]
exog=[c for c in m["exog_cols"] if c in df.columns]
exlag=[c for c in df.columns if c.endswith("_lag_season")]
cal=[c for c in ["hr","dow","mon","doy"] if c in df.columns]
X=[Y]+lag+exog+exlag+cal+["unit_code"]
data=df.dropna(subset=X+[f"y_h{h}" for h in H]).copy()
tr=data[data[T]<test_start].copy(); te=data[data[T]>=test_start].copy()

rows=[]
for h in H:
 y=tr[f"y_h{h}"]; yt=te[f"y_h{h}"]
 models={
  "Ridge":Ridge(alpha=1.0),
  "RandomForest":RandomForestRegressor(n_estimators=100,min_samples_leaf=5,n_jobs=-1,random_state=2026),
  "LightGBM":LGBMRegressor(n_estimators=400,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=2026,verbose=-1)
 }
 for name,model in models.items():
  model.fit(tr[X],y)
  p=model.predict(te[X])
  rows.append({"model":name,"horizon":h,"MAE":mean_absolute_error(yt,p),"RMSE":np.sqrt(mean_squared_error(yt,p)),"n_train":len(tr),"n_test":len(te)})
  print(name,h,rows[-1],flush=True)

# Independent plant-transfer check: fit on all but 4 plant codes, test on 4 held-out plants.
units=sorted(data[U].unique())
held=units[-4:]
fit=data[~data[U].isin(held)]
hold=data[data[U].isin(held)]
h=H[0]
model=LGBMRegressor(n_estimators=400,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=2026,verbose=-1)
X_transfer=[c for c in X if c!="unit_code"]
model.fit(fit[X_transfer],fit[f"y_h{h}"])
p=model.predict(hold[X_transfer])
transfer_mae=mean_absolute_error(hold[f"y_h{h}"],p)
seen_sample=data[data[U].isin(units[:4])]
p2=model.predict(seen_sample[X_transfer])
seen_mae=mean_absolute_error(seen_sample[f"y_h{h}"],p2)
out["transfer"]={"held_out_units":held,"seen_mae":float(seen_mae),"unseen_mae":float(transfer_mae)}

pd.DataFrame(rows).to_csv(OUT/"fresh_model_results.csv",index=False)
(Path(OUT/"fresh_analysis.json")).write_text(json.dumps(out,indent=2,default=str))
(Path(OUT/"fresh_metadata.json")).write_text(json.dumps({
 "repository":"noobball111/ADYm_2","branch":"chatgpt-fresh-rerun-20261002",
 "feat_md5":m["md5"],"rows":len(df),"usable_rows":len(data),
 "train_rows":len(tr),"test_rows":len(te),"features":X,
 "held_out_units":held,"model_seed":2026
},indent=2,default=str))
print("DONE",json.dumps(out,indent=2,default=str))
