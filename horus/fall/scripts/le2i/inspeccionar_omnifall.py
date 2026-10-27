from datasets import load_dataset
import pandas as pd

ds = load_dataset("simplexsigil2/omnifall", "le2i")
print("Splits disponibles:", list(ds.keys()))

df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
print(f"\nTotal de filas: {len(df)}")
print("\nColumnas:", list(df.columns))
print("\nPrimeras 10 filas:")
print(df.head(10).to_string())

print("\nValores únicos de label:")
print(df["label"].value_counts() if "label" in df.columns else "(no hay columna 'label')")

df.to_csv("le2i_omnifall.csv", index=False)
print("\nGuardado en le2i_omnifall.csv")
print("\nSujetos por escenario:")
df["escenario"] = df["path"].str.split("/").str[0]
print(df.groupby("escenario")["subject"].unique())

print("\nCantidad de sujetos distintos en total:", df["subject"].nunique())