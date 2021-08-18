import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.testing import assert_frame_equal
from icecream import ic

pd.set_option("max_columns", 4, "max_rows", 10, "max_colwidth", 12)


if __name__ == "__main__":
    movie = pd.read_csv("data/movie.csv", index_col="movie_title")
    ic(movie[["duration"]].head())
    movie_2_hours = movie["duration"] > 120
    ic(movie_2_hours.head(10))
    ic(movie_2_hours.sum())
    ic(movie_2_hours.mean())
    ic(movie["duration"].dropna().gt(120).mean())
    ic(movie_2_hours.describe())
    ic(movie_2_hours.value_counts(normalize=True))
    ic(movie_2_hours.astype(int).describe())
    actors = movie[["actor_1_facebook_likes", "actor_2_facebook_likes"]].dropna()
    ic((actors["actor_1_facebook_likes"] > actors["actor_2_facebook_likes"]).mean())

    movie = pd.read_csv("data/movie.csv", index_col="movie_title")
    criteria1 = movie.imdb_score > 8
    criteria2 = movie.content_rating == "PG-13"
    criteria3 = (movie.title_year < 2000) | (movie.title_year > 2009)
    criteria_final = criteria1 & criteria2 & criteria3
    ic(criteria_final.head())

    movie = pd.read_csv("data/movie.csv", index_col="movie_title")
    crit_a1 = movie.imdb_score > 8
    crit_a2 = movie.content_rating == "PG-13"
    crit_a3 = (movie.title_year < 2000) | (movie.title_year > 2009)
    final_crit_a = crit_a1 & crit_a2 & crit_a3

    crit_b1 = movie.imdb_score < 5
    crit_b2 = movie.content_rating == "R"
    crit_b3 = (movie.title_year >= 2000) & (movie.title_year <= 2010)
    final_crit_b = crit_b1 & crit_b2 & crit_b3

    final_crit_all = final_crit_a | final_crit_b
    ic(final_crit_all.head())
    ic(movie[final_crit_all].head())
    ic(movie.loc[final_crit_all].head())
    cols = ["imdb_score", "content_rating", "title_year"]
    movie_filtered = movie.loc[final_crit_all, cols]
    ic(movie_filtered.head(10))

    # ic(movie.iloc[final_crit_all])
    ic(movie.iloc[final_crit_all.values])
    final_crit_a2 = (
        (movie.imdb_score > 8)
        & (movie.content_rating == "PG-13")
        & ((movie.title_year < 2000) | (movie.title_year > 2009))
    )
    ic(final_crit_a2.equals(final_crit_a))

    ## Comparing Row Filtering and Index Filtering
    college = pd.read_csv("data/college.csv")
    ic(college[college["STABBR"] == "TX"].head())
    college2 = college.set_index("STABBR")
    ic(college2.loc["TX"].head())
    # get_ipython().run_line_magic('timeit', "college[college['STABBR'] == 'TX']")
    # get_ipython().run_line_magic('timeit', "college2.loc['TX']")
    # get_ipython().run_line_magic('timeit', "college2 = college.set_index('STABBR')")
    ic(college[college["STABBR"] == "TX"])
    ic(college2.loc["TX"])
    college2 = college.set_index("STABBR")
    ic(college2)

    states = ["TX", "CA", "NY"]
    ic(college[college["STABBR"].isin(states)])
    ic(college2.loc[states])

    ## Selecting with unique and sorted indexes
    college = pd.read_csv("data/college.csv")
    college2 = college.set_index("STABBR")
    ic(college2.index.is_monotonic)
    college3 = college2.sort_index()
    ic(college3.index.is_monotonic)
    # get_ipython().run_line_magic('timeit', "college[college['STABBR'] == 'TX']")
    # get_ipython().run_line_magic('timeit', "college2.loc['TX']")
    # get_ipython().run_line_magic('timeit', "college3.loc['TX']")
    college_unique = college.set_index("INSTNM")
    ic(college_unique.index.is_unique)
    ic(college[college["INSTNM"] == "Stanford University"])
    ic(college_unique.loc["Stanford University"])
    ic(college_unique.loc[["Stanford University"]])
    # get_ipython().run_line_magic('timeit', "college[college['INSTNM'] == 'Stanford University']")
    # get_ipython().run_line_magic('timeit', "college_unique.loc[['Stanford University']]")

    college.index = college["CITY"] + ", " + college["STABBR"]
    college = college.sort_index()
    ic(college.head())
    ic(college.loc["Miami, FL"].head())
    # get_ipython().run_cell_magic('timeit', '', "crit1 = college['CITY'] == 'Miami'\ncrit2 = college['STABBR'] == 'FL'\ncollege[crit1 & crit2]")
    # get_ipython().run_line_magic('timeit', "college.loc['Miami, FL']")

    ## Translating SQL WHERE clauses
    employee = pd.read_csv("data/employee.csv")
    ic(employee.dtypes)
    ic(employee.DEPARTMENT.value_counts().head())
    ic(employee.GENDER.value_counts())
    ic(employee.BASE_SALARY.describe())
    depts = ["Houston Police Department-HPD", "Houston Fire Department (HFD)"]
    criteria_dept = employee.DEPARTMENT.isin(depts)
    criteria_gender = employee.GENDER == "Female"
    criteria_sal = (employee.BASE_SALARY >= 80000) & (employee.BASE_SALARY <= 120000)
    criteria_final = criteria_dept & criteria_gender & criteria_sal
    select_columns = ["UNIQUE_ID", "DEPARTMENT", "GENDER", "BASE_SALARY"]
    ic(employee.loc[criteria_final, select_columns].head())

    criteria_sal = employee.BASE_SALARY.between(80_000, 120_000)
    top_5_depts = employee.DEPARTMENT.value_counts().index[:5]
    criteria = ~employee.DEPARTMENT.isin(top_5_depts)
    ic(employee[criteria])

    ## Improving readability of boolean indexing with the query method
    employee = pd.read_csv("data/employee.csv")
    depts = ["Houston Police Department-HPD", "Houston Fire Department (HFD)"]
    select_columns = ["UNIQUE_ID", "DEPARTMENT", "GENDER", "BASE_SALARY"]
    qs = "DEPARTMENT in @depts " " and GENDER == 'Female' " " and 80000 <= BASE_SALARY <= 120000"
    emp_filtered = employee.query(qs)
    ic(emp_filtered[select_columns].head())

    top10_depts = employee.DEPARTMENT.value_counts().index[:10].tolist()
    qs = "DEPARTMENT not in @top10_depts and GENDER == 'Female'"
    employee_filtered2 = employee.query(qs)
    ic(employee_filtered2.head())

    ## Preserving Series size with the where method
    movie = pd.read_csv("data/movie.csv", index_col="movie_title")
    fb_likes = movie["actor_1_facebook_likes"].dropna()
    ic(fb_likes.head())
    ic(fb_likes.describe())

    fig, ax = plt.subplots(figsize=(10, 8))
    fb_likes.hist(ax=ax)
    fig.savefig("images/ch07/c7-hist.png", dpi=300)

    criteria_high = fb_likes < 20_000
    ic(criteria_high.mean().round(2))
    ic(fb_likes.where(criteria_high).head())
    ic(fb_likes.where(criteria_high, other=20000).head())
    criteria_low = fb_likes > 300
    fb_likes_cap = fb_likes.where(criteria_high, other=20_000).where(criteria_low, 300)
    ic(fb_likes_cap.head())
    ic(len(fb_likes), len(fb_likes_cap))

    fig, ax = plt.subplots(figsize=(10, 8))
    fb_likes_cap.hist(ax=ax)
    fig.savefig("images/ch07/c7-hist2.png", dpi=300)

    fb_likes_cap2 = fb_likes.clip(lower=300, upper=20000)
    ic(fb_likes_cap2.equals(fb_likes_cap))

    ## Masking DataFrame rows
    movie = pd.read_csv("data/movie.csv", index_col="movie_title")
    c1 = movie["title_year"] >= 2010
    c2 = movie["title_year"].isna()
    criteria = c1 | c2
    ic(movie.mask(criteria).head())
    movie_mask = movie.mask(criteria).dropna(how="all")
    ic(movie_mask.head())
    movie_boolean = movie[movie["title_year"] < 2010]
    ic(movie_mask.equals(movie_boolean))
    ic(movie_mask.shape == movie_boolean.shape)
    ic(movie_mask.dtypes == movie_boolean.dtypes)

    assert_frame_equal(movie_boolean, movie_mask, check_dtype=False)
    # get_ipython().run_line_magic('timeit', "movie.mask(criteria).dropna(how='all')")
    # get_ipython().run_line_magic('timeit', "movie[movie['title_year'] < 2010]")
    ic(movie[movie["title_year"] < 2010])
    ic(movie.mask(criteria).dropna(how="all"))

    ## Selecting with booleans, integer location, and labels
    movie = pd.read_csv("data/movie.csv", index_col="movie_title")
    c1 = movie["content_rating"] == "G"
    c2 = movie["imdb_score"] < 4
    criteria = c1 & c2
    movie_loc = movie.loc[criteria]
    ic(movie_loc.head())
    ic(movie_loc.equals(movie[criteria]))
    # iLocation based boolean indexing cannot use an indexable as a mask
    # movie_iloc = movie.iloc[criteria]
    movie_iloc = movie.iloc[criteria.values]
    ic(movie_iloc.equals(movie_loc))
    criteria_col = movie.dtypes == np.int64
    ic(criteria_col.head())
    ic(movie.loc[:, criteria_col].head())
    ic(movie.iloc[:, criteria_col.values].head())

    cols = ["content_rating", "imdb_score", "title_year", "gross"]
    movie.loc[criteria, cols].sort_values("imdb_score")
    col_index = [movie.columns.get_loc(col) for col in cols]
    ic(col_index)
    movie.iloc[criteria.values, col_index].sort_values("imdb_score")
    a = criteria.values
    ic(a[:5])
    ic(len(a), len(criteria))
    ic(movie.select_dtypes(int))
