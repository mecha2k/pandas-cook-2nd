import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp

from scipy import stats
from icecream import ic

pd.set_option("max_columns", 4, "max_rows", 10, "max_colwidth", 12)


if __name__ == "__main__":
    fueleco = pd.read_csv("data/vehicles.csv.zip", low_memory=False)
    # fueleco = fueleco.select_dtypes(["object"]).fillna("")
    # fueleco.info()
    # ic(fueleco.dtypes)
    # ic(type(fueleco.dtypes))
    # ic(fueleco.dtypes.value_counts())
    # ic(fueleco.isna())
    # ic(fueleco.isna().any())
    # ic(fueleco.isna().sum())
    # ic(fueleco.isna().any().sum())
    # fueleco_cols = fueleco.columns[fueleco.isna().any()]
    # ic(fueleco_cols)
    # # fueleco = fueleco[fueleco_cols]
    #
    # ic(fueleco.mean(numeric_only=True))
    # ic(fueleco.std(numeric_only=True))
    # ic(fueleco.select_dtypes(include="number").quantile([0, 0.25, 0.5, 0.75, 1]))
    # ic(fueleco.describe())
    # ic(fueleco.describe(include=object))
    # ic(fueleco.describe().T)
    # ic(fueleco.select_dtypes("int64").describe().T)
    # ic(np.iinfo(np.int8))
    # ic(np.iinfo(np.int16))
    # fueleco[["city08", "comb08"]].info()
    # ic(
    #     fueleco[["city08", "comb08"]]
    #     .assign(city08=fueleco.city08.astype(np.int16), comb08=fueleco.comb08.astype(np.int16))
    #     .info()
    # )
    # ic(fueleco.make.nunique())
    # ic(fueleco.model.nunique())
    # fueleco[["make"]].info()
    # fueleco[["make"]].assign(make=fueleco.make.astype("category")).info()
    #
    # fueleco[["model"]].info()
    # fueleco[["model"]].assign(model=fueleco.model.astype("category")).info()
    #
    # ic(fueleco.select_dtypes(object).columns)
    # ic(fueleco.drive.nunique())
    # ic(fueleco.drive.sample(5, random_state=42))
    # ic(fueleco.drive.isna())
    # ic(fueleco.drive.isna().sum())
    # ic(fueleco.drive.isna().mean() * 100)
    # ic(fueleco.drive.value_counts())
    # top_n = fueleco.make.value_counts().index[:6]
    # ic(
    #     fueleco.assign(
    #         make=fueleco.make.where(fueleco.make.isin(top_n), "Other")
    #     ).make.value_counts()
    # )
    #
    # fig, ax = plt.subplots(figsize=(8, 6))
    # top_n = fueleco.make.value_counts().index[:6]
    # ic(
    #     fueleco.assign(make=fueleco.make.where(fueleco.make.isin(top_n), "Other"))
    #     .make.value_counts()
    #     .plot.bar(ax=ax)
    # )
    # fig.savefig("images/ch05/c5-catpan.png", dpi=300)
    #
    # fig, ax = plt.subplots(figsize=(8, 6))
    # top_n = fueleco.make.value_counts().index[:6]
    # sns.countplot(
    #     y="make",
    #     data=(fueleco.assign(make=fueleco.make.where(fueleco.make.isin(top_n), "Other"))),
    # )
    # fig.savefig("images/ch05/c5-catsns.png", dpi=300)
    #
    # ic(fueleco[fueleco.drive.isna()])
    # ic(fueleco.drive.value_counts(dropna=False))
    # ic(fueleco.rangeA.value_counts())
    # ic(
    #     fueleco.rangeA.str.extract(r"([^0-9.])")
    #     .dropna()
    #     .apply(lambda row: "".join(row), axis=1)
    #     .value_counts()
    # )
    # ic(set(fueleco.rangeA.apply(type)))
    # ic(fueleco.rangeA.isna().sum())
    # ic(
    #     fueleco.rangeA.fillna("0")
    #     .str.replace("-", "/")
    #     .str.split("/", expand=True)
    #     .astype(float)
    #     .mean(axis=1)
    # )
    # ic(
    #     fueleco.rangeA.fillna("0")
    #     .str.replace("-", "/")
    #     .str.split("/", expand=True)
    #     .astype(float)
    #     .mean(axis=1)
    #     .pipe(lambda ser_: pd.cut(ser_, 10))
    #     .value_counts()
    # )
    # # ic(
    # #     fueleco.rangeA.fillna("0")
    # #     .str.replace("-", "/")
    # #     .str.split("/", expand=True)
    # #     .astype(float)
    # #     .mean(axis=1)
    # #     .pipe(lambda ser_: pd.qcut(ser_, 10))
    # #     .value_counts()
    # # )
    # # ic(fueleco.city08.pipe(lambda ser: pd.qcut(ser, q=10)).value_counts())
    #
    # ## Continuous Data
    # ic(fueleco.select_dtypes("number"))
    # ic(fueleco.city08.sample(5, random_state=42))
    # ic(fueleco.city08.isna().sum())
    # ic(fueleco.city08.isna().mean() * 100)
    # ic(fueleco.city08.describe())
    #
    # fig, ax = plt.subplots(figsize=(10, 8))
    # fueleco.city08.hist(ax=ax)
    # fig.savefig("images/ch05/c5-conthistpan.png", dpi=300)
    #
    # fig, ax = plt.subplots(figsize=(10, 8))
    # fueleco.city08.hist(ax=ax, bins=30)
    # fig.savefig("images/ch05/c5-conthistpanbins.png", dpi=300)
    #
    # fig, ax = plt.subplots(figsize=(10, 8))
    # sns.rugplot(data=fueleco.city08, ax=ax)
    # fig.savefig("images/ch05/c5-conthistsns.png", dpi=300)
    #
    # fig, axs = plt.subplots(nrows=3, figsize=(10, 8))
    # sns.boxplot(data=fueleco.city08, ax=axs[0])
    # sns.violinplot(data=fueleco.city08, ax=axs[1])
    # sns.boxenplot(data=fueleco.city08, ax=axs[2])
    # fig.savefig("images/ch05/c5-contothersns.png", dpi=300)
    #
    # # ic(stats.kstest(fueleco.city08, cdf="norm"))
    #
    # fig, ax = plt.subplots(figsize=(10, 8))
    # stats.probplot(fueleco.city08, plot=ax)
    # fig.savefig("images/ch05/c5-conprob.png", dpi=300)
    #
    # ## Comparing Continuous Values across Categories
    # mask = fueleco.make.isin(["Ford", "Honda", "Tesla", "BMW"])
    # fueleco[mask].groupby("make").city08.agg(["mean", "std"])
    #
    # g = sns.catplot(x="make", y="city08", data=fueleco[mask], height=5, kind="box")
    # g.ax.figure.savefig("images/ch05/c5-catbox.png", dpi=300)
    #
    # # mask = fueleco.make.isin(["Ford", "Honda", "Tesla", "BMW"])
    # # ic(fueleco[mask].groupby("make").city08.count())
    #
    # # g = sns.catplot(x="make", y="city08", data=fueleco[mask], height=1, kind="box")
    # # sns.swarmplot(x="make", y="city08", data=fueleco[mask], color="k", size=1, ax=g.ax)
    # # g.ax.figure.savefig("images/ch05/c5-catbox2.png", dpi=300)
    #
    # g = sns.catplot(
    #     x="make",
    #     y="city08",
    #     data=fueleco[mask],
    #     height=1,
    #     kind="box",
    #     col="year",
    #     col_order=[2012, 2014, 2016, 2018],
    #     col_wrap=2,
    # )
    # g.axes[0].figure.savefig("images/ch05/c5-catboxcol.png", dpi=300)
    #
    # g = sns.catplot(
    #     x="make",
    #     y="city08",
    #     data=fueleco[mask],
    #     height=1,
    #     kind="box",
    #     hue="year",
    #     hue_order=[2012, 2014, 2016, 2018],
    # )
    # g.ax.figure.savefig("images/ch05/c5-catboxhue.png", dpi=300)
    #
    # # mask = fueleco.make.isin(["Ford", "Honda", "Tesla", "BMW"])
    # # ic(
    # #     fueleco[mask]
    # #     .groupby("make")
    # #     .city08.agg(["mean", "std"])
    # #     .style.background_gradient(cmap="RdBu", axis=0)
    # # )
    #
    # ## Comparing Two Continuous Columns
    # ic(fueleco.city08.cov(fueleco.highway08))
    # ic(fueleco.city08.cov(fueleco.comb08))
    # ic(fueleco.city08.cov(fueleco.cylinders))
    # ic(fueleco.city08.corr(fueleco.highway08))
    # ic(fueleco.city08.corr(fueleco.cylinders))
    #
    # fig, ax = plt.subplots(figsize=(8, 8))
    # corr = fueleco[["city08", "highway08", "cylinders"]].corr()
    # mask = np.zeros_like(corr, dtype=bool)
    # mask[np.triu_indices_from(mask)] = True
    # sns.heatmap(
    #     corr, mask=mask, fmt=".2f", annot=True, ax=ax, cmap="RdBu", vmin=-1, vmax=1, square=True
    # )
    # fig.savefig("images/ch05/c5-heatmap.png", dpi=300, bbox_inches="tight")
    #
    # fig, ax = plt.subplots(figsize=(8, 8))
    # fueleco.plot.scatter(x="city08", y="highway08", alpha=0.1, ax=ax)
    # fig.savefig("images/ch05/c5-scatpan.png", dpi=300, bbox_inches="tight")
    #
    # fig, ax = plt.subplots(figsize=(8, 8))
    # fueleco.plot.scatter(x="city08", y="cylinders", alpha=0.1, ax=ax)
    # fig.savefig("images/ch05/c5-scatpan-cyl.png", dpi=300, bbox_inches="tight")
    #
    # ic(fueleco.cylinders.isna().sum())
    #
    # fig, ax = plt.subplots(figsize=(8, 8))
    # (
    #     fueleco.assign(cylinders=fueleco.cylinders.fillna(0)).plot.scatter(
    #         x="city08", y="cylinders", alpha=0.1, ax=ax
    #     )
    # )
    # fig.savefig("images/ch05/c5-scatpan-cyl0.png", dpi=300, bbox_inches="tight")
    #
    # res = sns.lmplot(x="city08", y="highway08", data=fueleco)
    # res.fig.savefig("images/ch05/c5-lmplot.png", dpi=300, bbox_inches="tight")
    #
    # ic(fueleco.city08.corr(fueleco.highway08 * 2))
    # ic(fueleco.city08.cov(fueleco.highway08 * 2))
    #
    # res = sns.relplot(
    #     x="city08",
    #     y="highway08",
    #     data=fueleco.assign(cylinders=fueleco.cylinders.fillna(0)),
    #     hue="year",
    #     size="barrels08",
    #     alpha=0.5,
    #     height=8,
    # )
    # res.fig.savefig("images/ch05/c5-relplot2.png", dpi=300, bbox_inches="tight")
    #
    # res = sns.relplot(
    #     x="city08",
    #     y="highway08",
    #     data=fueleco.assign(cylinders=fueleco.cylinders.fillna(0)),
    #     hue="year",
    #     size="barrels08",
    #     alpha=0.5,
    #     height=8,
    #     col="make",
    #     col_order=["Ford", "Tesla"],
    # )
    # res.fig.savefig("images/ch05/c5-relplot3.png", dpi=300, bbox_inches="tight")
    #
    # ic(fueleco.city08.corr(fueleco.barrels08, method="spearman"))

    ## Comparing Categorical and Categorical Values

    def generalize(ser, match_name, default):
        seen = None
        for match, name in match_name:
            mask = ser.str.contains(match)
            if seen is None:
                seen = mask
            else:
                seen |= mask
            ser = ser.where(~mask, name)
        ser = ser.where(seen, default)
        return ser

    makes = ["Ford", "Tesla", "BMW", "Toyota"]
    data = fueleco[fueleco.make.isin(makes)].assign(
        SClass=lambda df_: generalize(
            df_.VClass,
            [
                ("Seaters", "Car"),
                ("Car", "Car"),
                ("Utility", "SUV"),
                ("Truck", "Truck"),
                ("Van", "Van"),
                ("van", "Van"),
                ("Wagon", "Wagon"),
            ],
            "other",
        )
    )
    ic(data.groupby(["make", "SClass"]).size().unstack())
    ic(pd.crosstab(data.make, data.SClass))
    ic(pd.crosstab([data.year, data.make], [data.SClass, data.VClass]))

    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    ic(cramers_v(data.make, data.SClass))

    fig, ax = plt.subplots(figsize=(10, 8))
    (data.pipe(lambda df_: pd.crosstab(df_.make, df_.SClass)).plot.bar(ax=ax))
    fig.savefig("images/ch05/c5-bar.png", dpi=300, bbox_inches="tight")

    res = sns.catplot(kind="count", x="make", hue="SClass", data=data)
    res.fig.savefig("images/ch05/c5-barsns.png", dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(10, 8))
    (
        data.pipe(lambda df_: pd.crosstab(df_.make, df_.SClass))
        .pipe(lambda df_: df_.div(df_.sum(axis=1), axis=0))
        .plot.bar(stacked=True, ax=ax)
    )
    fig.savefig("images/ch05/c5-barstacked.png", dpi=300, bbox_inches="tight")

    ic(cramers_v(data.make, data.trany))
    ic(cramers_v(data.make, data.model))

    profile = pp.ProfileReport(
        fueleco,
        title="Pandas Profiling",
        correlations={"kendall": {"calculate": False}, "cramers": {"calculate": False}},
    )
    profile.to_file("images/ch05/fuel.html")
