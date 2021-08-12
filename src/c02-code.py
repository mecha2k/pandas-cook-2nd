import numpy as np
import pandas as pd
from icecream import ic

pd.set_option("max_columns", 4, "max_rows", 10, "max_colwidth", 12)


if __name__ == "__main__":
    movies = pd.read_csv("data/movie.csv")
    movies.info()

    ic(movies.shape)
    ic(movies.size)
    ic(movies.ndim)
    ic(len(movies))
    ic(movies.count())
    ic(movies.describe().T)
    movie_actor_director = movies[["actor_1_name", "actor_2_name", "actor_3_name", "director_name"]]
    ic(movie_actor_director.head())
    ic(type(movies[["director_name"]]))
    ic(type(movies.loc[:, "director_name"]))
    ic(type(movies.loc[:, ["director_name"]]))

    cols = ["actor_1_name", "actor_2_name", "actor_3_name", "director_name"]
    movie_actor_director = movies[cols]

    def shorten(col):
        return col.replace("facebook_likes", "fb").replace("_for_reviews", "")

    movies = movies.rename(columns=shorten)
    ic(movies.columns)
    ic(movies.dtypes.value_counts())
    ic(movies.select_dtypes(include="int").head())
    ic(movies.select_dtypes(include="number").head())
    ic(movies.select_dtypes(include=["int", "object"]).head())
    ic(movies.select_dtypes(exclude="float").head())
    ic(movies.filter(like="fb").head())
    ic(movies.filter(items=cols).head())
    ic(movies.filter(regex=r"\d").head())

    cat_core = ["movie_title", "title_year", "content_rating", "genres"]
    cat_people = ["director_name", "actor_1_name", "actor_2_name", "actor_3_name"]
    cat_other = ["color", "country", "language", "plot_keywords", "movie_imdb_link"]
    cont_fb = ["director_fb", "actor_1_fb", "actor_2_fb", "actor_3_fb", "cast_total_fb", "movie_fb"]
    cont_finance = ["budget", "gross"]
    cont_num_reviews = ["num_voted_users", "num_user", "num_critic"]
    cont_other = ["imdb_score", "duration", "aspect_ratio", "facenumber_in_poster"]

    new_col_order = (
        cat_core + cat_people + cat_other + cont_fb + cont_finance + cont_num_reviews + cont_other
    )
    ic(set(movies.columns) == set(new_col_order))
    ic(movies[new_col_order].head())
    ic(movies.describe(percentiles=[0.01, 0.3, 0.99]).T)
    # Dropping of nuisance columns in DataFrame reductions(with 'numeric_only=None') is deprecated
    # ic(movies.min(skipna=False))
    ic(movies.apply(pd.to_numeric, args=["coerce"]).min(skipna=False))

    ## Chaining DataFrame Methods
    movies = pd.read_csv("data/movie.csv")

    def shorten(col):
        return col.replace("facebook_likes", "fb").replace("_for_reviews", "")

    movies = movies.rename(columns=shorten)
    ic(movies.isna().head())
    ic(movies.isna().sum())
    ic(movies.isna().sum().sum())
    ic(movies.isnull().head())
    ic(movies.isnull().sum().head())
    ic(movies.isnull().sum().sum())
    ic(movies.isnull().any().head())
    ic(movies.isnull().any().any())
    ic(movies.isnull().dtypes.value_counts())

    # Dropping of nuisance columns in DataFrame reductions(with 'numeric_only=None') is deprecated
    # ic(movies[["color", "movie_title", "color"]].max())
    ic(movies.apply(pd.to_numeric, args=["coerce"])[["color", "movie_title", "color"]].max())

    with pd.option_context("max_colwidth", 20):
        ic(movies.select_dtypes(["object"]).fillna("").max())

    ## DataFrame Operations
    colleges = pd.read_csv("data/college.csv")
    # can only concatenate str (not "int") to str
    # ic(colleges + 5)

    colleges = pd.read_csv("data/college.csv", index_col="INSTNM")
    college_ugds = colleges.filter(like="UGDS_")
    ic(college_ugds.head())

    name = "Northwest-Shoals Community College"
    ic(college_ugds.loc[name])
    ic(college_ugds.loc[name].round(2))
    ic((college_ugds.loc[name] + 0.0001).round(2))
    ic(college_ugds + 0.00501)
    ic((college_ugds + 0.00501) // 0.01)

    college_ugds_op_round = (college_ugds + 0.00501) // 0.01 / 100
    ic(college_ugds_op_round.head())
    college_ugds_round = (college_ugds + 0.00001).round(2)
    ic(college_ugds_round)
    ic(college_ugds_op_round.equals(college_ugds_round))
    ic(0.045 + 0.005)
    college2 = college_ugds.add(0.00501).floordiv(0.01).div(100)
    ic(college2.equals(college_ugds_op_round))

    ## Comparing Missing Values
    ic(np.nan == np.nan)
    ic(None is None)
    ic(np.nan > 5)
    ic(5 > np.nan)
    ic(np.nan != 5)

    college = pd.read_csv("data/college.csv", index_col="INSTNM")
    college_ugds = college.filter(like="UGDS_")

    ic(college_ugds == 0.0019)
    college_self_compare = college_ugds == college_ugds
    ic(college_self_compare.head())
    ic(college_self_compare.all())
    ic((college_ugds == np.nan).sum())
    ic(college_ugds.isnull().sum())
    ic(college_ugds.equals(college_ugds))
    ic(college_ugds.eq(0.0019))  # same as college_ugds == .0019

    from pandas.testing import assert_frame_equal

    ic(assert_frame_equal(college_ugds, college_ugds) is None)

    ## Transposing the direction of a DataFrame operation
    college = pd.read_csv("data/college.csv", index_col="INSTNM")
    college_ugds = college.filter(like="UGDS_")
    ic(college_ugds.head())
    ic(college_ugds.count())
    ic(college_ugds.count(axis="columns").head())
    ic(college_ugds.sum(axis="columns").head())
    ic(college_ugds.median(axis="index"))

    college_ugds_cumsum = college_ugds.cumsum(axis=1)
    ic(college_ugds_cumsum.head())

    ## Determining college campus diversity
    college = pd.read_csv("data/college.csv", index_col="INSTNM")
    college_ugds = college.filter(like="UGDS_")

    ic(college_ugds.isnull().sum(axis="columns").sort_values(ascending=False).head())
    college_ugds = college_ugds.dropna(how="all")
    ic(college_ugds.isnull().sum())
    ic(college_ugds.ge(0.15))

    diversity_metric = college_ugds.ge(0.15).sum(axis="columns")
    ic(diversity_metric.head())
    ic(diversity_metric.value_counts())
    ic(diversity_metric.sort_values(ascending=False).head())
    ic(college_ugds.loc[["Regency Beauty Institute-Austin", "Central Texas Beauty College-Temple"]])

    us_news_top = [
        "Rutgers University-Newark",
        "Andrews University",
        "Stanford University",
        "University of Houston",
        "University of Nevada-Las Vegas",
    ]
    ic(diversity_metric.loc[us_news_top])
    ic(college_ugds.max(axis=1).sort_values(ascending=False).head(10))
    ic((college_ugds > 0.01).all(axis=1).any())
