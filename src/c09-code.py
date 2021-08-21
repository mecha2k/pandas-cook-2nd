import pandas as pd
import numpy as np
import pandas_profiling as pp

from scipy.stats import gmean, hmean
from icecream import ic

pd.set_option("max_columns", 4, "max_rows", 10, "max_colwidth", 12)


if __name__ == "__main__":
    flights = pd.read_csv("data/flights.csv")
    flights.info()
    ic(flights.head())

    ic(flights.groupby("AIRLINE").agg({"ARR_DELAY": "mean"}))
    ic(flights.groupby("AIRLINE")["ARR_DELAY"].agg("mean"))
    ic(flights.groupby("AIRLINE")["ARR_DELAY"].agg(np.mean))
    ic(flights.groupby("AIRLINE")["ARR_DELAY"].mean())

    grouped = flights.groupby("AIRLINE")
    ic(type(grouped))
    # np.sqrt is not aggregate function
    # ic(flights.groupby("AIRLINE")["ARR_DELAY"].agg(np.sqrt))

    # Grouping and aggregating with multiple columns and functions
    ic(flights.groupby(["AIRLINE", "WEEKDAY"])["CANCELLED"].agg("sum"))
    ic(flights.groupby(["AIRLINE", "WEEKDAY"])[["CANCELLED", "DIVERTED"]].agg(["sum", "mean"]))
    ic(
        flights.groupby(["ORG_AIR", "DEST_AIR"]).agg(
            {"CANCELLED": ["sum", "mean", "size"], "AIR_TIME": ["mean", "var"]}
        )
    )
    ic(
        flights.groupby(["ORG_AIR", "DEST_AIR"]).agg(
            sum_cancelled=pd.NamedAgg(column="CANCELLED", aggfunc="sum"),
            mean_cancelled=pd.NamedAgg(column="CANCELLED", aggfunc="mean"),
            size_cancelled=pd.NamedAgg(column="CANCELLED", aggfunc="size"),
            mean_air_time=pd.NamedAgg(column="AIR_TIME", aggfunc="mean"),
            var_air_time=pd.NamedAgg(column="AIR_TIME", aggfunc="var"),
        )
    )

    res = flights.groupby(["ORG_AIR", "DEST_AIR"]).agg(
        {"CANCELLED": ["sum", "mean", "size"], "AIR_TIME": ["mean", "var"]}
    )
    res.columns = ["_".join(x) for x in res.columns.to_flat_index()]
    ic(res)

    def flatten_cols(df):
        df.columns = ["_".join(x) for x in df.columns.to_flat_index()]
        return df

    res = (
        flights.groupby(["ORG_AIR", "DEST_AIR"])
        .agg({"CANCELLED": ["sum", "mean", "size"], "AIR_TIME": ["mean", "var"]})
        .pipe(flatten_cols)
    )
    ic(res)

    res = (
        flights.assign(ORG_AIR=flights.ORG_AIR.astype("category"))
        .groupby(["ORG_AIR", "DEST_AIR"])
        .agg({"CANCELLED": ["sum", "mean", "size"], "AIR_TIME": ["mean", "var"]})
    )
    ic(res)

    res = (
        flights.assign(ORG_AIR=flights.ORG_AIR.astype("category"))
        .groupby(["ORG_AIR", "DEST_AIR"], observed=True)
        .agg({"CANCELLED": ["sum", "mean", "size"], "AIR_TIME": ["mean", "var"]})
    )
    ic(res)

    ## Removing the MultiIndex after grouping
    flights = pd.read_csv("data/flights.csv")
    airline_info = (
        flights.groupby(["AIRLINE", "WEEKDAY"])
        .agg({"DIST": ["sum", "mean"], "ARR_DELAY": ["min", "max"]})
        .astype(int)
    )
    ic(airline_info)

    airline_info.columns.get_level_values(0)
    airline_info.columns.get_level_values(1)
    airline_info.columns.to_flat_index()
    airline_info.columns = ["_".join(x) for x in airline_info.columns.to_flat_index()]
    ic(airline_info)

    airline_info.reset_index()

    ic(
        flights.groupby(["AIRLINE", "WEEKDAY"])
        .agg(
            dist_sum=pd.NamedAgg(column="DIST", aggfunc="sum"),
            dist_mean=pd.NamedAgg(column="DIST", aggfunc="mean"),
            arr_delay_min=pd.NamedAgg(column="ARR_DELAY", aggfunc="min"),
            arr_delay_max=pd.NamedAgg(column="ARR_DELAY", aggfunc="max"),
        )
        .astype(int)
        .reset_index()
    )

    ic(flights.groupby(["AIRLINE"], as_index=False)["DIST"].agg("mean").round(0))

    # Grouping with a custom aggregation function
    college = pd.read_csv("data/college.csv")
    ic(college.groupby("STABBR")["UGDS"].agg(["mean", "std"]).round(0))

    def max_deviation(s):
        std_score = (s - s.mean()) / s.std()
        return std_score.abs().max()

    ic(college.groupby("STABBR")["UGDS"].agg(max_deviation).round(1))
    ic(college.groupby("STABBR")[["UGDS", "SATVRMID", "SATMTMID"]].agg(max_deviation).round(1))
    ic(
        college.groupby(["STABBR", "RELAFFIL"])[["UGDS", "SATVRMID", "SATMTMID"]]
        .agg([max_deviation, "mean", "std"])
        .round(1)
    )

    ic(max_deviation.__name__)
    max_deviation.__name__ = "Max Deviation"
    ic(
        college.groupby(["STABBR", "RELAFFIL"])[["UGDS", "SATVRMID", "SATMTMID"]]
        .agg([max_deviation, "mean", "std"])
        .round(1)
    )

    # Customizing aggregating functions with *args and **kwargs
    def pct_between_1_3k(s):
        return s.between(1_000, 3_000).mean() * 100

    ic(college.groupby(["STABBR", "RELAFFIL"])["UGDS"].agg(pct_between_1_3k).round(1))

    def pct_between(s, low, high):
        return s.between(low, high).mean() * 100

    ic(college.groupby(["STABBR", "RELAFFIL"])["UGDS"].agg(pct_between, 1_000, 10_000).round(1))

    def between_n_m(n, m):
        def wrapper(ser):
            return pct_between(ser, n, m)

        wrapper.__name__ = f"between_{n}_{m}"
        return wrapper

    ic(
        college.groupby(["STABBR", "RELAFFIL"])["UGDS"]
        .agg([between_n_m(1_000, 10_000), "max", "mean"])
        .round(1)
    )

    ## Examining the groupby object
    college = pd.read_csv("data/college.csv")
    grouped = college.groupby(["STABBR", "RELAFFIL"])
    ic(type(grouped))
    ic([attr for attr in dir(grouped) if not attr.startswith("_")])
    ic(grouped.ngroups)

    groups = list(grouped.groups)
    ic(groups[:6])
    ic(grouped.get_group(("FL", 1)))

    # from IPython.display import display
    # for name, group in grouped:
    #     print(name)
    #     display(group.head(3))

    for name, group in grouped:
        print(name)
        print(group)
        break
    ic(grouped.head(2))
    ic(grouped.nth([1, -1]))

    ## Filtering for states with a minority majority
    college = pd.read_csv("data/college.csv", index_col="INSTNM")
    grouped = college.groupby("STABBR")
    ic(grouped.ngroups)
    ic(college["STABBR"].nunique())  # verifying the same number

    def check_minority(df, threshold):
        minority_pct = 1 - df["UGDS_WHITE"]
        total_minority = (df["UGDS"] * minority_pct).sum()
        total_ugds = df["UGDS"].sum()
        total_minority_pct = total_minority / total_ugds
        return total_minority_pct > threshold

    college_filtered = grouped.filter(check_minority, threshold=0.5)
    ic(college_filtered)
    ic(college.shape)
    ic(college_filtered.shape)
    ic(college_filtered["STABBR"].nunique())

    college_filtered_20 = grouped.filter(check_minority, threshold=0.2)
    ic(college_filtered_20.shape)
    ic(college_filtered_20["STABBR"].nunique())

    college_filtered_70 = grouped.filter(check_minority, threshold=0.7)
    ic(college_filtered_70.shape)
    ic(college_filtered_70["STABBR"].nunique())

    ## Transforming through a weight loss bet
    weight_loss = pd.read_csv("data/weight_loss.csv")
    ic(weight_loss.query('Month == "Jan"'))

    def percent_loss(s):
        return ((s - s.iloc[0]) / s.iloc[0]) * 100

    ic(weight_loss.query('Name=="Bob" and Month=="Jan"')["Weight"].pipe(percent_loss))
    ic(weight_loss.groupby(["Name", "Month"])["Weight"].transform(percent_loss))
    ic(
        weight_loss.assign(
            percent_loss=(
                weight_loss.groupby(["Name", "Month"])["Weight"].transform(percent_loss).round(1)
            )
        ).query('Name=="Bob" and Month in ["Jan", "Feb"]')
    )
    ic(
        weight_loss.assign(
            percent_loss=(
                weight_loss.groupby(["Name", "Month"])["Weight"].transform(percent_loss).round(1)
            )
        ).query('Week == "Week 4"')
    )
    ic(
        weight_loss.assign(
            percent_loss=(
                weight_loss.groupby(["Name", "Month"])["Weight"].transform(percent_loss).round(1)
            )
        )
        .query('Week == "Week 4"')
        .pivot(index="Month", columns="Name", values="percent_loss")
    )
    ic(
        weight_loss.assign(
            percent_loss=(
                weight_loss.groupby(["Name", "Month"])["Weight"].transform(percent_loss).round(1)
            )
        )
        .query('Week == "Week 4"')
        .pivot(index="Month", columns="Name", values="percent_loss")
        .assign(winner=lambda df_: np.where(df_.Amy < df_.Bob, "Amy", "Bob"))
    )
    ic(
        weight_loss.assign(
            percent_loss=(
                weight_loss.groupby(["Name", "Month"])["Weight"].transform(percent_loss).round(1)
            )
        )
        .query('Week == "Week 4"')
        .pivot(index="Month", columns="Name", values="percent_loss")
        .assign(winner=lambda df_: np.where(df_.Amy < df_.Bob, "Amy", "Bob"))
        .style.highlight_min(axis=1)
    )
    ic(
        weight_loss.assign(
            percent_loss=(
                weight_loss.groupby(["Name", "Month"])["Weight"].transform(percent_loss).round(1)
            )
        )
        .query('Week == "Week 4"')
        .pivot(index="Month", columns="Name", values="percent_loss")
        .assign(winner=lambda df_: np.where(df_.Amy < df_.Bob, "Amy", "Bob"))
        .winner.value_counts()
    )
    ic(
        weight_loss.assign(
            percent_loss=(
                weight_loss.groupby(["Name", "Month"])["Weight"].transform(percent_loss).round(1)
            )
        )
        .query('Week == "Week 4"')
        .groupby(["Month", "Name"])["percent_loss"]
        .first()
        .unstack()
    )
    ic(
        weight_loss.assign(
            percent_loss=(
                weight_loss.groupby(["Name", "Month"])["Weight"].transform(percent_loss).round(1)
            ),
            Month=pd.Categorical(
                weight_loss.Month, categories=["Jan", "Feb", "Mar", "Apr"], ordered=True
            ),
        )
        .query('Week == "Week 4"')
        .pivot(index="Month", columns="Name", values="percent_loss")
    )

    ## Calculating weighted mean SAT scores per state with apply
    college = pd.read_csv("data/college.csv")
    subset = ["UGDS", "SATMTMID", "SATVRMID"]
    college2 = college.dropna(subset=subset)
    ic(college.shape)
    ic(college2.shape)

    def weighted_math_average(df):
        weighted_math = df["UGDS"] * df["SATMTMID"]
        return int(weighted_math.sum() / df["UGDS"].sum())

    ic(college2.groupby("STABBR").apply(weighted_math_average))
    # ic(college2.groupby("STABBR").agg(weighted_math_average))
    # ic(college2.groupby("STABBR")["SATMTMID"].agg(weighted_math_average))

    def weighted_average(df):
        weight_m = df["UGDS"] * df["SATMTMID"]
        weight_v = df["UGDS"] * df["SATVRMID"]
        wm_avg = weight_m.sum() / df["UGDS"].sum()
        wv_avg = weight_v.sum() / df["UGDS"].sum()
        data = {
            "w_math_avg": wm_avg,
            "w_verbal_avg": wv_avg,
            "math_avg": df["SATMTMID"].mean(),
            "verbal_avg": df["SATVRMID"].mean(),
            "count": len(df),
        }
        return pd.Series(data)

    ic(college2.groupby("STABBR").apply(weighted_average).astype(int))
    ic(college.groupby("STABBR").apply(weighted_average))

    def calculate_means(df):
        df_means = pd.DataFrame(index=["Arithmetic", "Weighted", "Geometric", "Harmonic"])
        cols = ["SATMTMID", "SATVRMID"]
        for col in cols:
            arithmetic = df[col].mean()
            weighted = np.average(df[col], weights=df["UGDS"])
            geometric = gmean(df[col])
            harmonic = hmean(df[col])
            df_means[col] = [arithmetic, weighted, geometric, harmonic]
        df_means["count"] = len(df)
        return df_means.astype(int)

    ic(college2.groupby("STABBR").apply(calculate_means))

    ## Grouping by continuous variables
    flights = pd.read_csv("data/flights.csv")
    ic(flights)

    bins = [-np.inf, 200, 500, 1000, 2000, np.inf]
    cuts = pd.cut(flights["DIST"], bins=bins)
    ic(cuts)
    ic(cuts.value_counts())
    ic(flights.groupby(cuts)["AIRLINE"].value_counts(normalize=True).round(3))
    ic(flights.groupby(cuts)["AIR_TIME"].quantile(q=[0.25, 0.5, 0.75]).div(60).round(2))

    labels = ["Under an Hour", "1 Hour", "1-2 Hours", "2-4 Hours", "4+ Hours"]
    cuts2 = pd.cut(flights["DIST"], bins=bins, labels=labels)
    ic(flights.groupby(cuts2)["AIRLINE"].value_counts(normalize=True).round(3).unstack())

    ## Counting the total number of flights between cities
    flights = pd.read_csv("data/flights.csv")
    flights_ct = flights.groupby(["ORG_AIR", "DEST_AIR"]).size()
    ic(flights_ct)
    ic(flights_ct.loc[[("ATL", "IAH"), ("IAH", "ATL")]])
    f_part3 = flights[["ORG_AIR", "DEST_AIR"]].apply(
        lambda ser: ser.sort_values().reset_index(drop=True), axis="columns"
    )
    ic(f_part3)

    # rename_dict = {0: "AIR1", 1: "AIR2"}
    # ic(
    #     flights[["ORG_AIR", "DEST_AIR"]]
    #     .apply(lambda ser: ser.sort_values().reset_index(drop=True), axis="columns")
    #     .rename(columns=rename_dict)
    #     .groupby(["AIR1", "AIR2"])
    #     .size()
    # )
    # ic(
    #     flights[["ORG_AIR", "DEST_AIR"]]
    #     .apply(lambda ser: ser.sort_values().reset_index(drop=True), axis="columns")
    #     .rename(columns=rename_dict)
    #     .groupby(["AIR1", "AIR2"])
    #     .size()
    #     .loc[("ATL", "IAH")]
    # )
    # ic(
    #     flights[["ORG_AIR", "DEST_AIR"]]
    #     .apply(lambda ser: ser.sort_values().reset_index(drop=True), axis="columns")
    #     .rename(columns=rename_dict)
    #     .groupby(["AIR1", "AIR2"])
    #     .size()
    #     .loc[("IAH", "ATL")]
    # )

    data_sorted = np.sort(flights[["ORG_AIR", "DEST_AIR"]])
    ic(data_sorted[:10])

    flights_sort2 = pd.DataFrame(data_sorted, columns=["AIR1", "AIR2"])
    ic(flights_sort2.equals(f_part3.rename(columns={"ORG_AIR": "AIR1", "DEST_AIR": "AIR2"})))

    flights_sort = flights[["ORG_AIR", "DEST_AIR"]].apply(
        lambda ser: ser.sort_values().reset_index(drop=True), axis="columns"
    )
    # get_ipython().run_cell_magic('timeit', '', "data_sorted = np.sort(flights[['ORG_AIR', 'DEST_AIR']])"
    #                             "flights_sort2 = pd.DataFrame(data_sorted, columns=['AIR1', 'AIR2'])")

    ## Finding the longest streak of on-time flights
    s = pd.Series([0, 1, 1, 0, 1, 1, 1, 0])
    ic(s)
    s1 = s.cumsum()
    ic(s1)
    ic(s.mul(s1))
    ic(s.mul(s1).diff())
    ic(s.mul(s.cumsum()).diff().where(lambda x: x < 0))
    ic(s.mul(s.cumsum()).diff().where(lambda x: x < 0).ffill())
    ic(s.mul(s.cumsum()).diff().where(lambda x: x < 0).ffill().add(s.cumsum(), fill_value=0))

    flights = pd.read_csv("data/flights.csv")
    ic(
        flights.assign(ON_TIME=flights["ARR_DELAY"].lt(15).astype(int))[
            ["AIRLINE", "ORG_AIR", "ON_TIME"]
        ]
    )

    def max_streak(s):
        s1 = s.cumsum()
        return s.mul(s1).diff().where(lambda x: x < 0).ffill().add(s1, fill_value=0).max()

    ic(
        flights.assign(ON_TIME=flights["ARR_DELAY"].lt(15).astype(int))
        .sort_values(["MONTH", "DAY", "SCHED_DEP"])
        .groupby(["AIRLINE", "ORG_AIR"])["ON_TIME"]
        .agg(["mean", "size", max_streak])
        .round(2)
    )

    def max_delay_streak(df):
        df = df.reset_index(drop=True)
        late = 1 - df["ON_TIME"]
        late_sum = late.cumsum()
        streak = (
            late.mul(late_sum).diff().where(lambda x: x < 0).ffill().add(late_sum, fill_value=0)
        )
        last_idx = streak.idxmax()
        first_idx = last_idx - streak.max() + 1
        res = df.loc[[first_idx, last_idx], ["MONTH", "DAY"]].assign(streak=streak.max())
        res.index = ["first", "last"]
        return res

    # ic(
    #     flights.assign(ON_TIME=flights["ARR_DELAY"].lt(15).astype(int))
    #     .sort_values(["MONTH", "DAY", "SCHED_DEP"])
    #     .groupby(["AIRLINE", "ORG_AIR"])
    #     .apply(max_delay_streak)
    #     .sort_values("streak", ascending=False)
    # )

    flights = pd.read_csv("data/flights.csv")
    profile = pp.ProfileReport(
        flights,
        title="Pandas Profiling",
        minimal=True,
        correlations={"kendall": {"calculate": False}, "cramers": {"calculate": False}},
    )
    profile.to_file("data/ch09_flights.html")
