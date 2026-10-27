from datasets import load_dataset
import pandas as pd

for cfg in ["up_fall", "gmdcsa24", "ur_fall", "muvim", "edf", "occu"]:
    try:
        ds = load_dataset("simplexsigil2/omnifall", cfg)
        df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
        c = df["label"].value_counts()
        print(f"{cfg:>10}: lie_down={c.get(5,0):>4}  lying={c.get(6,0):>4}  "
              f"fall={c.get(1,0):>4}  fallen={c.get(2,0):>4}  total={len(df)}")
    except Exception as e:
        print(f"{cfg:>10}: no disponible ({str(e)[:40]})")