import argparse, json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error

def benchmark(df, mask_rate=0.1, seed=42):
    num = df.select_dtypes(include=np.number).copy()
    if num.shape[1] == 0:
        raise ValueError("No numeric columns found.")
    rng = np.random.default_rng(seed)
    original = num.copy()
    masked = num.copy()
    eval_mask = pd.DataFrame(False, index=num.index, columns=num.columns)
    for c in num.columns:
        candidates = np.flatnonzero(num[c].notna().to_numpy())
        if len(candidates) < 5:
            continue
        n = max(1, int(len(candidates) * mask_rate))
        idx = rng.choice(candidates, size=n, replace=False)
        masked.iloc[idx, masked.columns.get_loc(c)] = np.nan
        eval_mask.iloc[idx, eval_mask.columns.get_loc(c)] = True

    methods = {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "knn": KNNImputer(n_neighbors=5),
        "iterative_regression": IterativeImputer(estimator=BayesianRidge(), random_state=seed, max_iter=20),
    }
    results = {}
    for name, imp in methods.items():
        arr = imp.fit_transform(masked)
        pred = pd.DataFrame(arr, columns=num.columns, index=num.index)
        per_col = {}
        for c in num.columns:
            m = eval_mask[c]
            if m.any():
                per_col[c] = float(mean_squared_error(original.loc[m,c], pred.loc[m,c]) ** 0.5)
        results[name] = {"mean_rmse": float(np.mean(list(per_col.values()))) if per_col else None,
                         "per_column_rmse": per_col}
    valid = {k:v["mean_rmse"] for k,v in results.items() if v["mean_rmse"] is not None}
    recommendation = min(valid, key=valid.get) if valid else None
    return {"results": results, "recommended_method": recommendation}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--mask-rate",type=float,default=0.1); ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args()
    print(json.dumps(benchmark(pd.read_csv(a.csv), a.mask_rate, a.seed), indent=2))

if __name__=="__main__": main()
