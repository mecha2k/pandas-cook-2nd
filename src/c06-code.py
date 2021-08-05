import pandas as pd
import numpy as np
from icecream import ic

pd.set_option("max_columns", 4, "max_rows", 10, "max_colwidth", 12)


if __name__ == "__main__":
    college = pd.read_csv("data/college.csv", index_col="INSTNM")
    city = college["CITY"]
    ic(city)
    ic(city["Alabama A & M University"])
    ic(city.loc["Alabama A & M University"])
    ic(city.iloc[0])
    ic(city[["Alabama A & M University", "Alabama State University"]])
    ic(city.loc[["Alabama A & M University", "Alabama State University"]])
    ic(city.iloc[[0, 4]])
    ic(city["Alabama A & M University":"Alabama State University"])
    ic(city[0:5])
    ic(city.loc["Alabama A & M University":"Alabama State University"])
    ic(city.iloc[0:5])

    alabama_mask = city.isin(["Birmingham", "Montgomery"])
    ic(city[alabama_mask])

    ic(college.loc["Alabama A & M University", "CITY"])
    ic(college.iloc[0, 0])
    ic(college.loc[["Alabama A & M University", "Alabama State University"], "CITY"])
    ic(college.iloc[[0, 4], 0])
    ic(college.loc["Alabama A & M University":"Alabama State University", "CITY"])
    ic(college.iloc[0:5, 0])
    ic(city.loc["Reid State Technical College":"Alabama State University"])

    college = pd.read_csv("data/college.csv", index_col="INSTNM")
    college.sample(5, random_state=42)
    ic(college.iloc[60])
    ic(college.loc["University of Alaska Anchorage"])
    ic(college.iloc[[60, 99, 3]])
    labels = [
        "University of Alaska Anchorage",
        "International Academy of Hair Design",
        "University of Alabama in Huntsville",
    ]
    ic(college.loc[labels])
    ic(college.iloc[99:102])
    start = "International Academy of Hair Design"
    stop = "Mesa Community College"
    ic(college.loc[start:stop])

    ic(college.iloc[[60, 99, 3]].index.tolist())

    ic(college=pd.read_csv("data/ic(college.csv", index_col="INSTNM"))
    ic(college.iloc[:3, :4])
    ic(college.loc[:"Amridge University", :"MENONLY"])
    ic(college.iloc[:, [4, 6]].head())
    ic(college.loc[:, ["WOMENONLY", "SATVRMID"]].head())
    ic(college.iloc[[100, 200], [7, 15]])

    rows = ["GateWay Community ic(college", "American Baptist Seminary of the West"]
    columns = ["SATMTMID", "UGDS_NHPI"]
    ic(college.loc[rows, columns])
    ic(college.iloc[5, -4])
    ic(college.loc["The University of Alabama", "PCTFLOAN"])
    ic(college.iloc[90:80:-2, 5])

    start = "Empire Beauty School-Flagstaff"
    stop = "Arizona State University-Tempe"
    ic(college.loc[start:stop:-2, "RELAFFIL"])

    college = pd.read_csv("data/college.csv", index_col="INSTNM")
    col_start = college.columns.get_loc("UGDS_WHITE")
    col_end = college.columns.get_loc("UGDS_UNKN") + 1
    ic(col_start, col_end)
    ic(college.iloc[:5, col_start:col_end])

    row_start = college.index[10]
    row_end = college.index[15]
    ic(college.loc[row_start:row_end, "UGDS_WHITE":"UGDS_UNKN"])
    ic(college.ix[10:16, "UGDS_WHITE":"UGDS_UNKN"])
    ic(college.iloc[10:16].loc[:, "UGDS_WHITE":"UGDS_UNKN"])

    college = pd.read_csv("data/college.csv", index_col="INSTNM")

    ic(college.loc["Sp":"Su"])
    ic(college=college.sort_index())
    ic(college.loc["Sp":"Su"])

    college = college.sort_index(ascending=False)
    ic(college.index.is_monotonic_decreasing)
    ic(college.loc["E":"B"])
