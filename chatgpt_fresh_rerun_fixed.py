from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from lightgbm import LGBMRegressor
DATA=Path("notebooks/data/processed/feat.parquet"); MAN=Path("notebooks/data/processed/manifest.json"); OUT=Path("chatgpt_fresh_rerun"); OUT.mkdir(exist_ok=True)
def md5(p):
 h=hashlib.md5()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
m=json.loads(MAN.read_text()); assert md5(DATA)==m["md5"]
df=pd.read_parquet(DATA); T="ts";U="plant";Y="cf";H=[1,6]; TEST=pd.Timestamp("2006-10-01")
df[T]=pd.to_datetime(df[T]); df=df.sort_values([U,T]).reset_index(drop=True); df["unit_code"]=df[U].astype("category").cat.codes
lag=[c for c in df if c.startswith("cf_lag") or c.startswith("cf_ma") or c.startswith("y_lag") or c.startswith("y_ma") or c.startswith("y_sd") or c=="y_diff1"]
ex=[c for c in ["GHI","DNI","cloud","temp"] if c in df]; cal=[c for c in ["hr","dow","mon","doy"] if c in df]
X=[Y]+lag+ex+[c for c in df if c.endswith("_lag_season")]+cal+["unit_code"]; data=df.dropna(subset=X+[f"y_{h}h" for h in H]).copy(); tr=data[data[T]<TEST]; te=data[data[T]>=TEST]
out={"shape":list(df.shape),"units":int(df[U].nunique()),"time_min":str(df[T].min()),"time_max":str(df[T].max()),"usable_rows":len(data),"train_rows":len(tr),"test_rows":len(te),"features":X,"mean_by_month":{str(int(k)):float(v) for k,v in df.groupby(df[T].dt.month)[Y].mean().items()}}
for c in ex: out["corr_"+c]=float(df[[Y,c]].corr(method="spearman").iloc[0,1])
out["ghi_decile"]=[float(v) for v in df.assign(q=pd.qcut(df.GHI,10,duplicates="drop")).groupby("q",observed=True)[Y].mean()]
rows=[]
for h in H:
 for name,model in {"Ridge":Ridge(alpha=1.0),"RandomForest":RandomForestRegressor(n_estimators=100,min_samples_leaf=5,n_jobs=-1,random_state=2026),"LightGBM":LGBMRegressor(n_estimators=400,learning_rate=.03,num_leaves=31,subsample=.8,colsample_bytree=.8,random_state=2026,verbose=-1)}.items():
  model.fit(tr[X],tr[f"y_{h}h"]); p=model.predict(te[X]); y=te[f"y_{h}h"]; rows.append({"model":name,"horizon":h,"MAE":float(mean_absolute_error(y,p)),"RMSE":float(np.sqrt(mean_squared_error(y,p)))})
units=sorted(data[U].unique()); val=units[:4]; hold=units[-4:]; fit=[u for u in units if u not in val+hold]; Xt=[c for c in X if c!="unit_code"]; fitdf=data[data[U].isin(fit)]; valdf=data[data[U].isin(val)]; holddf=data[data[U].isin(hold)]
tm=LGBMRegressor(n_estimators=400,learning_rate=.03,num_leaves=31,subsample=.8,colsample_bytree=.8,random_state=2026,verbose=-1); tm.fit(fitdf[Xt],fitdf["y_1h"]); pv=tm.predict(valdf[Xt]); ph=tm.predict(holddf[Xt])
out["transfer"]={"fit_units":fit,"validation_units":val,"unseen_units":hold,"validation_mae":float(mean_absolute_error(valdf["y_1h"],pv)),"unseen_mae":float(mean_absolute_error(holddf["y_1h"],ph))}
pd.DataFrame(rows).to_csv(OUT/"fresh_model_results.csv",index=False); (OUT/"fresh_analysis.json").write_text(json.dumps(out,indent=2)); (OUT/"fresh_metadata.json").write_text(json.dumps({"branch":"chatgpt-fresh-rerun-fixed-20261002","md5":m["md5"],"fit_units":fit,"validation_units":val,"unseen_units":hold},indent=2)); print(json.dumps(out,indent=2))
