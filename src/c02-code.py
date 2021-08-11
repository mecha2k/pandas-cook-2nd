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
    ic(movies.min(skipna=False))

    ## Chaining DataFrame Methods
    movies = pd.read_csv("data/movie.csv")

    def shorten(col):
        return col.replace("facebook_likes", "fb").replace("_for_reviews", "")

    movies = movies.rename(columns=shorten)
    movies.isnull().head()

    ic(movies.isnull().sum().head())
    ic(movies.isnull().sum().sum())
    ic(movies.isnull().any().any())
    ic(movies.isnull().dtypes.value_counts())
    ic(movies[["color", "movie_title", "color"]].max())

    with pd.option_context("max_colwidth", 20):
        ic(movies.select_dtypes(["object"]).fillna("").max())
    with pd.option_context("max_colwidth", 20):
        ic(movies.select_dtypes(["object"]).fillna("").max())

# # ## DataFrame Operations
#

# colleges = pd.read_csv('data/college.csv')
# colleges + 5

# colleges = pd.read_csv('data/college.csv', index_col='INSTNM')
# college_ugds = colleges.filter(like='UGDS_')
# college_ugds.head()

# name = 'Northwest-Shoals Community College'
# college_ugds.loc[name]

# college_ugds.loc[name].round(2)

# (college_ugds.loc[name] + .0001).round(2)

# college_ugds + .00501

# (college_ugds + .00501) // .01

# college_ugds_op_round = (college_ugds + .00501) // .01 / 100
# college_ugds_op_round.head()

# college_ugds_round = (college_ugds + .00001).round(2)
# college_ugds_round

# college_ugds_op_round.equals(college_ugds_round)
# # ### How it works\...
#

# .045 + .005
# # ### There\'s more\...
#

# college2 = (college_ugds
#     .add(.00501)
#     .floordiv(.01)
#     .div(100)
# )
# college2.equals(college_ugds_op_round)
# # ### See also
#
# # ## Comparing Missing Values
#

# np.nan == np.nan

# None == None

# np.nan > 5

# 5 > np.nan

# np.nan != 5
# # ### Getting ready
#

# college = pd.read_csv('data/college.csv', index_col='INSTNM')
# college_ugds = college.filter(like='UGDS_')

# college_ugds == .0019

# college_self_compare = college_ugds == college_ugds
# college_self_compare.head()

# college_self_compare.all()

# (college_ugds == np.nan).sum()

# college_ugds.isnull().sum()

# college_ugds.equals(college_ugds)
# # ### How it works\...
#
# # ### There\'s more\...
#

# college_ugds.eq(.0019)    # same as college_ugds == .0019

# from pandas.testing import assert_frame_equal
# assert_frame_equal(college_ugds, college_ugds) is None
# # ## Transposing the direction of a DataFrame operation
#
# # ### How to do it\...
#

# college = pd.read_csv('data/college.csv', index_col='INSTNM')
# college_ugds = college.filter(like='UGDS_')
# college_ugds.head()

# college_ugds.count()

# college_ugds.count(axis='columns').head()

# college_ugds.sum(axis='columns').head()

# college_ugds.median(axis='index')
# # ### How it works\...
#
# # ### There\'s more\...
#

# college_ugds_cumsum = college_ugds.cumsum(axis=1)
# college_ugds_cumsum.head()
# # ### See also
#
# # ## Determining college campus diversity
#

# pd.read_csv('data/college_diversity.csv', index_col='School')
# # ### How to do it\...
#

# college = pd.read_csv('data/college.csv', index_col='INSTNM')
# college_ugds = college.filter(like='UGDS_')

# (college_ugds.isnull()
#    .sum(axis='columns')
#    .sort_values(ascending=False)
#    .head()
# )

# college_ugds = college_ugds.dropna(how='all')
# college_ugds.isnull().sum()

# college_ugds.ge(.15)

# diversity_metric = college_ugds.ge(.15).sum(axis='columns')
# diversity_metric.head()

# diversity_metric.value_counts()

# diversity_metric.sort_values(ascending=False).head()

# college_ugds.loc[['Regency Beauty Institute-Austin',
#                    'Central Texas Beauty College-Temple']]

# us_news_top = ['Rutgers University-Newark',
#                   'Andrews University',
#                   'Stanford University',
#                   'University of Houston',
#                   'University of Nevada-Las Vegas']
# diversity_metric.loc[us_news_top]
# # ### How it works\...
#
# # ### There\'s more\...
#

# (college_ugds
#    .max(axis=1)
#    .sort_values(ascending=False)
#    .head(10)
# )

# (college_ugds > .01).all(axis=1).any()
# # ### See also
