import pandas as pd

from icecream import ic

pd.set_option("max_columns", 4, "max_rows", 10)

if __name__ == "__main__":
    movies = pd.read_csv("data/movie.csv")
    movies.head()

    ic(movies.index)
    ic(movies.columns)
    ic(movies.values)

    ic(type(movies.index))
    ic(type(movies.columns))
    ic(type(movies.values))

    ic(issubclass(pd.RangeIndex, pd.Index))

    ic(movies.index.values)
    ic(movies.columns.values)

    ic(movies.dtypes)
    ic(movies.info())

    ic(movies.loc[:, "director_name"])
    ic(movies.iloc[:, 1])

    ic(
        movies.director_name.index,
        movies.director_name.dtype,
        movies.director_name.size,
        movies.director_name.name,
    )

    ic(movies["director_name"].apply(type).unique())

    s_attr_methods = set(dir(pd.Series))
    ic(len(s_attr_methods))
    df_attr_methods = set(dir(pd.DataFrame))
    ic(len(df_attr_methods))
    ic(len(s_attr_methods & df_attr_methods))

    director = movies["director_name"]
    fb_likes = movies["actor_1_facebook_likes"]

    fb_likes.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    director.isna()
    fb_likes_filled = fb_likes.fillna(0)
    fb_likes_filled.count()
    fb_likes_dropped = fb_likes.dropna()
    ic(fb_likes_dropped.size)
    ic(director.value_counts(normalize=True).mul(100))
    ic(director.hasnans)
    ic(director.isna().sum())
    ic(fb_likes.fillna(0).astype(int).head())

    idx_map = {
        "Avatar": "Ratava",
        "Spectre": "Ertceps",
        "Pirates of the Caribbean: At World's End": "POC",
    }
    col_map = {"aspect_ratio": "aspect", "movie_facebook_likes": "fblikes"}
    ic(movies.set_index("movie_title").rename(index=idx_map, columns=col_map).head(3))

    ids = movies.index.tolist()
    columns = movies.columns.tolist()

    ids[0] = "Ratava"
    ids[1] = "POC"
    ids[2] = "Ertceps"
    columns[1] = "DirectoR"
    columns[-2] = "aspecT"
    columns[-1] = "fbLikes"
    movies.index = ids
    movies.columns = columns
    ic(movies.head(3))

    def to_clean(value):
        return value.strip().lower().replace(" ", "_")

    ic(movies.rename(columns=to_clean).head(3))

    cols = [col.strip().lower().replace(" ", "_") for col in movies.columns]
    movies.columns = cols
    movies.head(3)

    movies["has_seen"] = 0
    idx_map = {
        "Avatar": "Ratava",
        "Spectre": "Ertceps",
        "Pirates of the Caribbean: At World's End": "POC",
    }
    col_map = {"aspect_ratio": "aspect", "movie_facebook_likes": "fblikes"}
    ic(movies.rename(index=idx_map, columns=col_map).assign(has_seen=0).head())

    total = (
        movies["actor_1_facebook_likes"]
        + movies["actor_2_facebook_likes"]
        + movies["actor_3_facebook_likes"]
        + movies["director_facebook_likes"]
    )
    ic(total.head(5))

    cols = [
        "actor_1_facebook_likes",
        "actor_2_facebook_likes",
        "actor_3_facebook_likes",
        "director_facebook_likes",
    ]
    sum_col = movies[cols].sum(axis="columns")
    ic(sum_col.head(5))

    movies.assign(total_likes=sum_col).head(5)

    def sum_likes(df):
        return df[[c for c in df.columns if "like" in c]].sum(axis=1)

    movies.assign(total_likes=sum_likes).head(5)

    movies.assign(total_likes=sum_col)["total_likes"].isna().sum()

    ic(movies.assign(total_likes=total)["total_likes"].isna().sum())

    ic(movies.assign(total_likes=total.fillna(0))["total_likes"].isna().sum())

    def cast_like_gt_actor_director(df):
        return df["cast_total_facebook_likes"] >= df["total_likes"]

    df2 = movies.assign(total_likes=total, is_cast_likes_more=cast_like_gt_actor_director)

    df2["is_cast_likes_more"].all()

    df2 = df2.drop(columns="total_likes")

    actor_sum = movies[[c for c in movies.columns if "actor_" in c and "_likes" in c]].sum(
        axis="columns"
    )
    actor_sum.head(5)

    ic(movies["cast_total_facebook_likes"] >= actor_sum)
    movies["cast_total_facebook_likes"].ge(actor_sum)
    movies["cast_total_facebook_likes"].ge(actor_sum).all()

    pct_like = actor_sum.div(movies["cast_total_facebook_likes"])
    pct_like.describe()
    pd.Series(pct_like.values, index=movies["movie_title"].values).head()
    # profit_index = movies.columns.get_loc("gross") + 1
    # ic(profit_index)

    # movies.insert(loc=profit_index, column="profit", value=movies["gross"] - movies["budget"])

    del movies["director"]
