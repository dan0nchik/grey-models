import pandas as pd
import numpy as np

from GreyModel.efgm import EFGM
from GreyModel.efgvm import EFGVM
from GreyModel.gvm import *
from GreyModel.gm import *
from GreyModel.tfgm import TFGM
from GreyModel.tfgvm import TFGVM

df = pd.read_csv("Us_Euro_parity_dataset.csv", index_col="Unnamed: 0")

# filter dates as in paper

df = df[(df["DATE"] >= "2006-01-01") & (df["DATE"] <= "2007-01-01")]
df = df.reset_index(drop=True)


# rename column
df.rename(columns={"US dollar/Euro (EXR.D.USD.EUR.SP00.A)": "price"}, inplace=True)


df["price"] = df["price"].interpolate().ffill().bfill()

# # UNCOMMENT FOR MAF
# maf_window_size = 4
# df["price"] = df["price"].rolling(window=maf_window_size).mean()
# df = df[~df["price"].isna()]


X0 = np.array(df["price"])

model_window = 5
for i in [GM(), EFGM(), TFGM(), GVM(), EFGVM(), TFGVM()]:
    model = i
    model.fit(X0, window_size=model_window)
    print(model.get_arpe())

# df['price'] = df["price"][model_window:].reset_index(drop=True)
# plt.plot(df["price"], label="train")
# plt.plot(model.predicted, label="predicted")
# plt.legend()
# plt.show()
