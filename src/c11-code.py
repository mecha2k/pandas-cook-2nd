import pandas as pd
import numpy as np
import glob

from icecream import ic
from IPython.display import display_html

# from sqlalchemy import create_engine

pd.set_option("max_rows", 10, "max_columns", 7, "max_colwidth", 9)
pd.set_option(
    "display.expand_frame_repr",
    True,
    "display.width",
    65,
)

if __name__ == "__main__":
    names = pd.read_csv("data/names.csv")
    ic(names)

    new_data_list = ["Aria", 1]
    names.loc[4] = new_data_list
    ic(names)
    names.loc["five"] = ["Zach", 3]
    ic(names)
    names.loc[len(names)] = {"Name": "Zayd", "Age": 2}
    ic(names)
    names.loc[len(names)] = pd.Series({"Age": 32, "Name": "Dean"})
    ic(names)

    names = pd.read_csv("data/names.csv")
    # names.append({"Name": "Aria", "Age": 1})
    names.append({"Name": "Aria", "Age": 1}, ignore_index=True)
    names.index = ["Canada", "Canada", "USA", "USA"]
    ic(names)
    s = pd.Series({"Name": "Zach", "Age": 3}, name=len(names))
    ic(s)
    names.append(s)
    ic(names)

    s1 = pd.Series({"Name": "Zach", "Age": 3}, name=len(names))
    s2 = pd.Series({"Name": "Zayd", "Age": 2}, name="USA")
    names.append([s1, s2])

    bball_16 = pd.read_csv("data/baseball16.csv")
    ic(bball_16)

    data_dict = bball_16.iloc[0].to_dict()
    ic(data_dict)

    new_data_dict = {k: "" if isinstance(v, str) else np.nan for k, v in data_dict.items()}
    ic(new_data_dict)

    random_data = []
    for i in range(1000):  # doctest: +SKIP
        d = dict()
        for k, v in data_dict.items():
            if isinstance(v, str):
                d[k] = np.random.choice(list("abcde"))
            else:
                d[k] = np.random.randint(10)
        random_data.append(pd.Series(d, name=i + len(bball_16)))
    ic(random_data[0])

    # Concatenating multiple DataFrames together
    stocks_2016 = pd.read_csv("data/stocks_2016.csv", index_col="Symbol")
    stocks_2017 = pd.read_csv("data/stocks_2017.csv", index_col="Symbol")
    ic(stocks_2016)
    ic(stocks_2017)

    s_list = [stocks_2016, stocks_2017]
    pd.concat(s_list)
    pd.concat(s_list, keys=["2016", "2017"], names=["Year", "Symbol"])
    pd.concat(s_list, keys=["2016", "2017"], axis="columns", names=["Year", None])
    pd.concat(s_list, join="inner", keys=["2016", "2017"], axis="columns", names=["Year", None])

    stocks_2016.append(stocks_2017)

    # Understanding the differences between concat, join, and merge
    years = 2016, 2017, 2018
    stock_tables = [
        pd.read_csv("data/stocks_{}.csv".format(year), index_col="Symbol") for year in years
    ]
    stocks_2016, stocks_2017, stocks_2018 = stock_tables
    ic(stocks_2016)
    ic(stocks_2017)
    ic(stocks_2018)

    pd.concat(stock_tables, keys=[2016, 2017, 2018])
    pd.concat(dict(zip(years, stock_tables)), axis="columns")
    stocks_2016.join(stocks_2017, lsuffix="_2016", rsuffix="_2017", how="outer")
    other = [stocks_2017.add_suffix("_2017"), stocks_2018.add_suffix("_2018")]
    stocks_2016.add_suffix("_2016").join(other, how="outer")
    stock_join = stocks_2016.add_suffix("_2016").join(other, how="outer")
    stock_concat = pd.concat(dict(zip(years, stock_tables)), axis="columns")
    level_1 = stock_concat.columns.get_level_values(1)
    level_0 = stock_concat.columns.get_level_values(0).astype(str)
    stock_concat.columns = level_1 + "_" + level_0
    stock_join.equals(stock_concat)

    stocks_2016.merge(stocks_2017, left_index=True, right_index=True)
    step1 = stocks_2016.merge(
        stocks_2017, left_index=True, right_index=True, how="outer", suffixes=("_2016", "_2017")
    )
    stock_merge = step1.merge(
        stocks_2018.add_suffix("_2018"), left_index=True, right_index=True, how="outer"
    )
    stock_concat.equals(stock_merge)

    names = ["prices", "transactions"]
    food_tables = [pd.read_csv("data/food_{}.csv".format(name)) for name in names]
    food_prices, food_transactions = food_tables
    ic(food_prices)
    ic(food_transactions)

    food_transactions.merge(food_prices, on=["item", "store"])
    food_transactions.merge(food_prices.query("Date == 2017"), how="left")
    food_prices_join = food_prices.query("Date == 2017").set_index(["item", "store"])
    ic(food_prices_join)
    food_transactions.join(food_prices_join, on=["item", "store"])
    pd.concat(
        [food_transactions.set_index(["item", "store"]), food_prices.set_index(["item", "store"])],
        axis="columns",
    )

    df_list = []
    for filename in glob.glob("data/gas prices/*.csv"):
        df_list.append(pd.read_csv(filename, index_col="Week", parse_dates=["Week"]))
    gas = pd.concat(df_list, axis="columns")
    ic(gas)

    # # Connecting to SQL databases
    # engine = create_engine("sqlite:///data/chinook.db")
    # tracks = pd.read_sql_table("tracks", engine)
    # ic(tracks)
    #
    # ic(
    #     pd.read_sql_table("genres", engine)
    #     .merge(tracks[["GenreId", "Milliseconds"]], on="GenreId", how="left")
    #     .drop("GenreId", axis="columns")
    # )
    # ic(
    #     pd.read_sql_table("genres", engine)
    #     .merge(tracks[["GenreId", "Milliseconds"]], on="GenreId", how="left")
    #     .drop("GenreId", axis="columns")
    #     .groupby("Name")["Milliseconds"]
    #     .mean()
    #     .pipe(lambda s_: pd.to_timedelta(s_, unit="ms"))
    #     .dt.floor("s")
    #     .sort_values()
    # )
    #
    # cust = pd.read_sql_table("customers", engine, columns=["CustomerId", "FirstName", "LastName"])
    # invoice = pd.read_sql_table("invoices", engine, columns=["InvoiceId", "CustomerId"])
    # ii = pd.read_sql_table("invoice_items", engine, columns=["InvoiceId", "UnitPrice", "Quantity"])
    # (cust.merge(invoice, on="CustomerId").merge(ii, on="InvoiceId"))
    # ic(
    #     cust.merge(invoice, on="CustomerId")
    #     .merge(ii, on="InvoiceId")
    #     .assign(Total=lambda df_: df_.Quantity * df_.UnitPrice)
    #     .groupby(["CustomerId", "FirstName", "LastName"])["Total"]
    #     .sum()
    #     .sort_values(ascending=False)
    # )

    # sql_string1 = """
    # SELECT
    #     Name,
    #     time(avg(Milliseconds) / 1000, 'unixepoch') as avg_time
    # FROM (
    #       SELECT
    #           g.Name,
    #           t.Milliseconds
    #       FROM
    #           genres as g
    #       JOIN
    #           tracks as t on
    #           g.genreid == t.genreid
    #      )
    # GROUP BY Name
    # ORDER BY avg_time"""
    # pd.read_sql_query(sql_string1, engine)

    # sql_string2 = '''
    #    SELECT
    #          c.customerid,
    #          c.FirstName,
    #          c.LastName,
    #          sum(ii.quantity * ii.unitprice) as Total
    #    FROM
    #         customers as c
    #    JOIN
    #         invoices as i
    #         on c.customerid = i.customerid
    #    JOIN
    #        invoice_items as ii
    #        on i.invoiceid = ii.invoiceid
    #    GROUP BY
    #        c.customerid, c.FirstName, c.LastName
    #    ORDER BY
    #        Total desc'''
    # pd.read_sql_query(sql_string2, engine)
