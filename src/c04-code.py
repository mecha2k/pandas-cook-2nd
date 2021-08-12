import pandas as pd
import numpy as np
from icecream import ic

pd.set_option("max_columns", 4, "max_rows", 10, "max_colwidth", 12)


if __name__ == "__main__":
    college = pd.read_csv("data/college.csv")
    college.sample(random_state=42)

    ic(college.shape)
    college.info()
    ic(college.describe(include=[np.number]).T)
    ic(college.describe(include=[object, pd.Categorical]).T)
    ic(
        college.describe(
            include=[np.number], percentiles=[0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        ).T
    )

    ## Reducing memory by changing data types
    college = pd.read_csv("data/college.csv")
    different_cols = ["RELAFFIL", "SATMTMID", "CURROPER", "INSTNM", "STABBR"]
    col2 = college.loc[:, different_cols]
    ic(col2.head())
    ic(col2.dtypes)
    original_mem = col2.memory_usage(deep=True)
    ic(original_mem)
    col2["RELAFFIL"] = col2["RELAFFIL"].astype(np.int8)
    ic(col2.dtypes)
    ic(college[different_cols].memory_usage(deep=True))
    ic(col2.select_dtypes(include=["object"]).nunique())
    col2["STABBR"] = col2["STABBR"].astype("category")
    ic(col2.dtypes)
    new_mem = col2.memory_usage(deep=True)
    ic(new_mem)
    ic(new_mem / original_mem)

    college.loc[0, "CURROPER"] = 10000000
    college.loc[0, "INSTNM"] = college.loc[0, "INSTNM"] + "a"
    ic(college[["CURROPER", "INSTNM"]].memory_usage(deep=True))
    ic(college["MENONLY"].dtype)
    # Cannot convert non-finite values (NA or inf) to integer
    # college["MENONLY"].astype(np.int8)
    college.assign(
        MENONLY=college["MENONLY"].astype("float16"), RELAFFIL=college["RELAFFIL"].astype("int8")
    )
    college.index = pd.Int64Index(college.index)
    ic(college.index.memory_usage())  # previously was just 80

    ## Selecting the smallest of the largest
    movie = pd.read_csv("data/movie.csv")
    movie2 = movie[["movie_title", "imdb_score", "budget"]]
    ic(movie2.head())
    ic(movie2.nlargest(100, "imdb_score").head())
    ic(movie2.nlargest(100, "imdb_score").nsmallest(5, "budget"))

    ## Selecting the largest of each group by sorting
    movie = pd.read_csv("data/movie.csv")
    ic(movie[["movie_title", "title_year", "imdb_score"]])
    ic(
        movie[["movie_title", "title_year", "imdb_score"]].sort_values(
            "title_year", ascending=False
        )
    )
    ic(
        movie[["movie_title", "title_year", "imdb_score"]].sort_values(
            ["title_year", "imdb_score"], ascending=False
        )
    )
    ic(
        movie[["movie_title", "title_year", "imdb_score"]]
        .sort_values(["title_year", "imdb_score"], ascending=False)
        .drop_duplicates(subset="title_year")
    )
    ic(
        movie[["movie_title", "title_year", "imdb_score"]]
        .groupby("title_year", as_index=False)
        .apply(lambda df: df.sort_values("imdb_score", ascending=False).head(1))
        .sort_values("title_year", ascending=False)
    )
    ic(
        movie[["movie_title", "title_year", "content_rating", "budget"]]
        .sort_values(["title_year", "content_rating", "budget"], ascending=[False, False, True])
        .drop_duplicates(subset=["title_year", "content_rating"])
    )

    ## Replicating nlargest with sort_values
    movie = pd.read_csv("data/movie.csv")
    ic(
        movie[["movie_title", "imdb_score", "budget"]]
        .nlargest(100, "imdb_score")
        .nsmallest(5, "budget")
    )
    ic(
        movie[["movie_title", "imdb_score", "budget"]]
        .sort_values("imdb_score", ascending=False)
        .head(100)
    )
    ic(
        movie[["movie_title", "imdb_score", "budget"]]
        .sort_values("imdb_score", ascending=False)
        .head(100)
        .sort_values("budget")
        .head(5)
    )
    ic(movie[["movie_title", "imdb_score", "budget"]].nlargest(100, "imdb_score").tail())
    ic(
        movie[["movie_title", "imdb_score", "budget"]]
        .sort_values("imdb_score", ascending=False)
        .head(100)
        .tail()
    )

    ## Calculating a trailing stop order price
    # import datetime
    # import pandas_datareader.data as web
    # import requests_cache
    #
    # session = requests_cache.CachedSession(
    #     cache_name="cache", backend="sqlite", expire_after=datetime.timedelta(days=90)
    # )
    #
    # tsla = web.DataReader("tsla", data_source="yahoo", start="2017-1-1", session=session)
    # ic(tsla.head(8))
    # tsla_close = tsla["Close"]
    # tsla_cummax = tsla_close.cummax()
    # ic(tsla_cummax.head())
    # ic(tsla["Close"].cummax().mul(0.9).head())
