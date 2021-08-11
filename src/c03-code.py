import pandas as pd
import numpy as np
import zipfile
import sqlite3
import sqlalchemy as sa
import json

from icecream import ic

pd.set_option("max_columns", 4, "max_rows", 10, "max_colwidth", 12)


if __name__ == "__main__":
    fname = ["Paul", "John", "Richard", "George"]
    lname = ["McCartney", "Lennon", "Starkey", "Harrison"]
    birth = [1942, 1940, 1940, 1943]
    people = {"first": fname, "last": lname, "birth": birth}
    beatles = pd.DataFrame(people)
    ic(beatles)
    ic(beatles.index)
    ic(pd.DataFrame(people, index=["a", "b", "c", "d"]))
    ic(
        pd.DataFrame(
            [
                {"first": "Paul", "last": "McCartney", "birth": 1942},
                {"first": "John", "last": "Lennon", "birth": 1940},
                {"first": "Richard", "last": "Starkey", "birth": 1940},
                {"first": "George", "last": "Harrison", "birth": 1943},
            ]
        )
    )
    ic(
        pd.DataFrame(
            [
                {"first": "Paul", "last": "McCartney", "birth": 1942},
                {"first": "John", "last": "Lennon", "birth": 1940},
                {"first": "Richard", "last": "Starkey", "birth": 1940},
                {"first": "George", "last": "Harrison", "birth": 1943},
            ],
            columns=["last", "first", "birth"],
        )
    )
    ic(beatles)

    # from io import StringIO
    #
    # fout = StringIO()
    # beatles.to_csv(fout)  # use a filename instead of fout
    # print(fout.getvalue())
    # _ = fout.seek(0)
    # pd.read_csv(fout)
    # _ = fout.seek(0)
    # pd.read_csv(fout, index_col=0)
    # fout = StringIO()
    # beatles.to_csv(fout, index=False)
    # print(fout.getvalue())

    diamonds = pd.read_csv("data/diamonds.csv", nrows=1000)
    ic(diamonds)
    diamonds.info()
    ic(diamonds.describe())

    diamonds2 = pd.read_csv(
        "data/diamonds.csv",
        nrows=1000,
        dtype={
            "carat": np.float32,
            "depth": np.float32,
            "table": np.float32,
            "x": np.float32,
            "y": np.float32,
            "z": np.float32,
            "price": np.int16,
        },
    )
    diamonds2.info()
    ic(diamonds2.describe())
    ic(diamonds2.cut.value_counts())
    ic(diamonds2.color.value_counts())
    ic(diamonds2.clarity.value_counts())

    diamonds3 = pd.read_csv(
        "data/diamonds.csv",
        nrows=1000,
        dtype={
            "carat": np.float32,
            "depth": np.float32,
            "table": np.float32,
            "x": np.float32,
            "y": np.float32,
            "z": np.float32,
            "price": np.int16,
            "cut": "category",
            "color": "category",
            "clarity": "category",
        },
    )
    diamonds3.info()
    ic(np.iinfo(np.int8))
    ic(np.finfo(np.float16))

    cols = ["carat", "cut", "color", "clarity", "depth", "table", "price"]
    diamonds4 = pd.read_csv(
        "data/diamonds.csv",
        nrows=1000,
        dtype={
            "carat": np.float32,
            "depth": np.float32,
            "table": np.float32,
            "price": np.int16,
            "cut": "category",
            "color": "category",
            "clarity": "category",
        },
        usecols=cols,
    )
    diamonds4.info()

    cols = ["carat", "cut", "color", "clarity", "depth", "table", "price"]
    diamonds_iter = pd.read_csv(
        "data/diamonds.csv",
        nrows=1000,
        dtype={
            "carat": np.float32,
            "depth": np.float32,
            "table": np.float32,
            "price": np.int16,
            "cut": "category",
            "color": "category",
            "clarity": "category",
        },
        usecols=cols,
        chunksize=200,
    )

    def process(df):
        return f"processed {df.size} items"

    for chunk in diamonds_iter:
        process(chunk)

    diamonds.price.memory_usage()
    diamonds.price.memory_usage(index=False)
    diamonds.cut.memory_usage()
    diamonds.cut.memory_usage(deep=True)
    diamonds4.to_feather("data/ch03/d.arr")
    diamonds5 = pd.read_feather("data/ch03/d.arr")
    diamonds4.to_parquet("data/ch03/d.pqt")

    # beatles.to_excel("data/ch03/beat.xls")
    beatles.to_excel("data/ch03/beat.xlsx")
    # beat2 = pd.read_excel("data/ch03/beat.xls")
    # ic(beat2)
    # beat2 = pd.read_excel("data/ch03/beat.xls", index_col=0)
    # ic(beat2)
    # ic(beat2.dtypes)

    xl_writer = pd.ExcelWriter("data/ch03/beat.xlsx")
    beatles.to_excel(xl_writer, sheet_name="All")
    beatles[beatles.birth < 1941].to_excel(xl_writer, sheet_name="1940")
    xl_writer.save()

    autos = pd.read_csv("data/vehicles.csv.zip", low_memory=False)
    ic(autos)
    ic(autos.modifiedOn.dtype)
    ic(autos.modifiedOn)
    ic(pd.to_datetime(autos.modifiedOn))

    autos = pd.read_csv("data/vehicles.csv.zip", parse_dates=["modifiedOn"])  # doctest: +SKIP
    ic(autos.modifiedOn)

    with zipfile.ZipFile("data/kaggle-survey-2018.zip") as z:
        ic("\n".join(z.namelist()))
        kag = pd.read_csv(z.open("multipleChoiceResponses.csv"))
        kag_questions = kag.iloc[0]
        survey = kag.iloc[1:]
    ic(survey.head(2).T)

    # con = sqlite3.connect('data/beat.db')
    # with con:
    #     cur = con.cursor()
    #     cur.execute("""DROP TABLE Band""")
    #     cur.execute("""CREATE TABLE Band(id INTEGER PRIMARY KEY,
    #         fname TEXT, lname TEXT, birthyear INT)""")
    #     cur.execute("""INSERT INTO Band VALUES(
    #         0, 'Paul', 'McCartney', 1942)""")
    #     cur.execute("""INSERT INTO Band VALUES(
    #         1, 'John', 'Lennon', 1940)""")
    #     _ = con.commit()

    # engine = sa.create_engine("sqlite:///data/beat.db", echo=True)
    # sa_connection = engine.connect()
    #
    # beat = pd.read_sql("Band", sa_connection, index_col="id")
    # ic(beat)
    #
    # sql = """SELECT fname, birthyear from Band"""
    # fnames = pd.read_sql(sql, con)
    # ic(fnames)

    # encoded = json.dumps(people)
    # ic(encoded)
    # ic(json.loads(encoded))
    #
    # beatles = pd.read_json(encoded)
    # ic(beatles)
    # records = beatles.to_json(orient="records")
    # ic(records)
    # ic(pd.read_json(records, orient="records"))
    # split = beatles.to_json(orient="split")
    # ic(split)
    # ic(pd.read_json(split, orient="split"))
    # index = beatles.to_json(orient="index")
    # ic(index)
    # ic(pd.read_json(index, orient="index"))
    # values = beatles.to_json(orient="values")
    # ic(values)
    # ic(pd.read_json(values, orient="values"))
    # ic(
    #     pd.read_json(values, orient="values").rename(
    #         columns=dict(enumerate(["first", "last", "birth"]))
    #     )
    # )
    # table = beatles.to_json(orient="table")
    # ic(table)
    # pd.read_json(table, orient="table")
    # output = beat.to_dict()
    # ic(output)
    # output["version"] = "0.4.1"
    # ic(json.dumps(output))
    #
    # url = "https://en.wikipedia.org/wiki/The_Beatles_discography"
    # dfs = pd.read_html(url)
    # ic(len(dfs))
    # ic(dfs[0])
    # url = "https://en.wikipedia.org/wiki/The_Beatles_discography"
    # dfs = pd.read_html(url, match="List of studio albums", na_values="—")
    # ic(len(dfs))
    # ic(dfs[0].columns)
    # url = "https://en.wikipedia.org/wiki/The_Beatles_discography"
    # dfs = pd.read_html(url, match="List of studio albums", na_values="—", header=[0, 1])
    # ic(len(dfs))
    # ic(dfs[0])
    # ic(dfs[0].columns)
    # df = dfs[0]
    # df.columns = [
    #     "Title",
    #     "Release",
    #     "UK",
    #     "AUS",
    #     "CAN",
    #     "FRA",
    #     "GER",
    #     "NOR",
    #     "US",
    #     "Certifications",
    # ]
    # ic(df)
    # res = (
    #     df.pipe(lambda df_: df_[~df_.Title.str.startswith("Released")])
    #     .iloc[:-1]
    #     .assign(
    #         release_date=lambda df_: pd.to_datetime(
    #             df_.Release.str.extract(r"Released: (.*) Label")[0].str.replace(r"\[E\]", "")
    #         ),
    #         label=lambda df_: df_.Release.str.extract(r"Label: (.*)"),
    #     )
    #     .loc[:, ["Title", "UK", "AUS", "CAN", "FRA", "GER", "NOR", "US", "release_date", "label"]]
    # )
    # ic(res)
    #
    # url = "https://github.com/mattharrison/datasets/blob/master/data/anscombes.csv"
    # dfs = pd.read_html(url, attrs={"class": "csv-data"})
    # ic(len(dfs))
    # ic(dfs[0])
