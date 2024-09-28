import numpy as np
import pandas as pd

from GreyModel.gm import GM
from GreyModel.gvm import GVM

# filter dates as in paper
for start, end in zip(["2005-01-01", "2006-01-01", "2007-01-01"], ["2006-01-01", "2007-01-01", "2008-01-01"]):
    print(start, end)
    df = pd.read_csv("Us_Euro_parity_dataset.csv", index_col="Unnamed: 0")
    df = df[(df["DATE"] >= start) & (df["DATE"] <= end)]
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
    for i in [GM(), GVM()]:
        model = i
        model.fit(X0, window_size=model_window)
        model_name = i.__str__().split('.')[-1].split()[0]
        print(model_name, model.get_arpe())
        print()

        ## UNCOMMENT FOR PLOTTING
        # df['price'] = df["price"][model_window:].reset_index(drop=True)
        # plt.plot(df["price"], label="train")
        # plt.plot(df["price"], label="predicted")
        # plt.title(f"{model_name} from {start} to {end}")
        # plt.legend()
        # plt.show()

        # plt.plot(model.residuals, label="residuals")
        # plt.title(f"Residuals of {model_name} from {start} to {end}")
        # plt.legend()
        # plt.show()