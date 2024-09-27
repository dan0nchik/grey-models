import pandas as pd
import numpy as np
from GreyModel.gvm import GVM
from GreyModel.gm import GM

df = pd.read_csv("Us_Euro_parity_dataset.csv", index_col="Unnamed: 0")

df.head()

# filter dates as in paper

df = df[(df["DATE"] >= "2005-01-01") & (df["DATE"] <= "2006-01-01")]


df.info()


# rename column
df.rename(columns={"US dollar/Euro (EXR.D.USD.EUR.SP00.A)": "price"}, inplace=True)


# # change in error is insignificant: either drop or fill
df = df[~df["price"].isna()]
# df["price"] = df["price"].interpolate().ffill().bfill()

# # MAF
# maf_window_size = 4
# df["price"] = df["price"].rolling(window=maf_window_size).mean()

X0 = np.array(df["price"])

model = GVM()
model.fit(X0, window_size=5)
print(model.get_arpe())

model_GM = GM()
model_GM.fit(X0, window_size=5)
print(model_GM.get_arpe())
print(model_GM.get_farpe())