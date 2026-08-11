import pandas as pd
df = pd.read_csv("up_fall_omnifall.csv")
df["actividad"] = df["path"].str.extract(r"Activity(\d+)").astype(int)

print(df[df["label"].isin([5, 6])].groupby("actividad").size())
print("\nDistribución de labels por actividad:")
print(pd.crosstab(df["actividad"], df["label"]))