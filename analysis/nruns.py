import pandas as pd

df = pd.read_csv("results/experiments.csv")
print(df["model"].value_counts())
