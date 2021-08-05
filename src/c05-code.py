import pandas as pd
import numpy as np
import scipy.stats as ss
from icecream import ic

pd.set_option("max_columns", 4, "max_rows", 10, "max_colwidth", 12)


if __name__ == "__main__":
    fueleco = pd.read_csv("data/vehicles.csv.zip")
    ic(fueleco)
    fueleco.mean()
    fueleco.std()
    fueleco.quantile([0, 0.25, 0.5, 0.75, 1])
    fueleco.describe()
    fueleco.describe(include=object)
    ic(fueleco.describe().T)
    ic(fueleco.dtypes)
    fueleco.dtypes.value_counts()
    ic(fueleco.select_dtypes("int64").describe().T)
    np.iinfo(np.int8)
    np.iinfo(np.int16)
    fueleco[["city08", "comb08"]].info()
    ic(
        fueleco[["city08", "comb08"]]
        .assign(city08=fueleco.city08.astype(np.int16), comb08=fueleco.comb08.astype(np.int16))
        .info()
    )
    fueleco.make.nunique()
    fueleco.model.nunique()
    fueleco[["make"]].info()
    ic(fueleco[["make"]].assign(make=fueleco.make.astype("category")).info())

    fueleco[["model"]].info()
    ic(fueleco[["model"]].assign(model=fueleco.model.astype("category")).info())

    ic(fueleco.select_dtypes(object).columns)
    fueleco.drive.nunique()
    fueleco.drive.sample(5, random_state=42)
    fueleco.drive.isna().sum()
    fueleco.drive.isna().mean() * 100
    fueleco.drive.value_counts()
    top_n = fueleco.make.value_counts().index[:6]
    ic(
        fueleco.assign(
            make=fueleco.make.where(fueleco.make.isin(top_n), "Other")
        ).make.value_counts()
    )

    # In[29]:
    #
    #
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(10, 8))
    # top_n = fueleco.make.value_counts().index[:6]
    # (fueleco     # doctest: +SKIP
    #    .assign(make=fueleco.make.where(
    #               fueleco.make.isin(top_n),
    #               'Other'))
    #    .make
    #    .value_counts()
    #    .plot.bar(ax=ax)
    # )
    # fig.savefig('/tmp/c5-catpan.png', dpi=300)     # doctest: +SKIP
    #
    #
    # # In[30]:
    #
    #
    # import seaborn as sns
    # fig, ax = plt.subplots(figsize=(10, 8))
    # top_n = fueleco.make.value_counts().index[:6]
    # sns.countplot(y='make',     # doctest: +SKIP
    #   data= (fueleco
    #    .assign(make=fueleco.make.where(
    #               fueleco.make.isin(top_n),
    #               'Other'))
    #   )
    # )
    # fig.savefig('/tmp/c5-catsns.png', dpi=300)    # doctest: +SKIP
    #
    #
    # # ### How it works...
    #
    # # In[31]:
    #
    #
    # fueleco[fueleco.drive.isna()]
    #
    #
    # # In[32]:
    #
    #
    # fueleco.drive.value_counts(dropna=False)
    #
    #
    # # ### There's more...
    #
    # # In[33]:
    #
    #
    # fueleco.rangeA.value_counts()
    #
    #
    # # In[34]:
    #
    #
    # (fueleco
    #  .rangeA
    #  .str.extract(r'([^0-9.])')
    #  .dropna()
    #  .apply(lambda row: ''.join(row), axis=1)
    #  .value_counts()
    # )
    #
    #
    # # In[35]:
    #
    #
    # set(fueleco.rangeA.apply(type))
    #
    #
    # # In[36]:
    #
    #
    # fueleco.rangeA.isna().sum()
    #
    #
    # # In[37]:
    #
    #
    # (fueleco
    #   .rangeA
    #   .fillna('0')
    #   .str.replace('-', '/')
    #   .str.split('/', expand=True)
    #   .astype(float)
    #   .mean(axis=1)
    # )
    #
    #
    # # In[38]:
    #
    #
    # (fueleco
    #   .rangeA
    #   .fillna('0')
    #   .str.replace('-', '/')
    #   .str.split('/', expand=True)
    #   .astype(float)
    #   .mean(axis=1)
    #   .pipe(lambda ser_: pd.cut(ser_, 10))
    #   .value_counts()
    # )
    #
    #
    # # In[39]:
    #
    #
    # (fueleco
    #   .rangeA
    #   .fillna('0')
    #   .str.replace('-', '/')
    #   .str.split('/', expand=True)
    #   .astype(float)
    #   .mean(axis=1)
    #   .pipe(lambda ser_: pd.qcut(ser_, 10))
    #   .value_counts()
    # )
    #
    #
    # # In[40]:
    #
    #
    # (fueleco
    #   .city08
    #   .pipe(lambda ser: pd.qcut(ser, q=10))
    #   .value_counts()
    # )
    #
    #
    # # ## Continuous Data
    #
    # # ### How to do it...
    #
    # # In[41]:
    #
    #
    # fueleco.select_dtypes('number')
    #
    #
    # # In[42]:
    #
    #
    # fueleco.city08.sample(5, random_state=42)
    #
    #
    # # In[43]:
    #
    #
    # fueleco.city08.isna().sum()
    #
    #
    # # In[44]:
    #
    #
    # fueleco.city08.isna().mean() * 100
    #
    #
    # # In[45]:
    #
    #
    # fueleco.city08.describe()
    #
    #
    # # In[46]:
    #
    #
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(10, 8))
    # fueleco.city08.hist(ax=ax)
    # fig.savefig('/tmp/c5-conthistpan.png', dpi=300)     # doctest: +SKIP
    #
    #
    # # In[47]:
    #
    #
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(10, 8))
    # fueleco.city08.hist(ax=ax, bins=30)
    # fig.savefig('/tmp/c5-conthistpanbins.png', dpi=300)     # doctest: +SKIP
    #
    #
    # # In[48]:
    #
    #
    # fig, ax = plt.subplots(figsize=(10, 8))
    # sns.distplot(fueleco.city08, rug=True, ax=ax)
    # fig.savefig('/tmp/c5-conthistsns.png', dpi=300)     # doctest: +SKIP
    #
    #
    # # ### How it works...
    #
    # # ### There's more...
    #
    # # In[49]:
    #
    #
    # fig, axs = plt.subplots(nrows=3, figsize=(10, 8))
    # sns.boxplot(fueleco.city08, ax=axs[0])
    # sns.violinplot(fueleco.city08, ax=axs[1])
    # sns.boxenplot(fueleco.city08, ax=axs[2])
    # fig.savefig('/tmp/c5-contothersns.png', dpi=300)
    #
    #
    # # In[50]:
    #
    #
    # from scipy import stats
    # stats.kstest(fueleco.city08, cdf='norm')
    #
    #
    # # In[51]:
    #
    #
    # from scipy import stats
    # fig, ax = plt.subplots(figsize=(10, 8))
    # stats.probplot(fueleco.city08, plot=ax)
    # fig.savefig('/tmp/c5-conprob.png', dpi=300)
    #
    #
    # # ## Comparing Continuous Values across Categories
    #
    # # ### How to do it...
    #
    # # In[52]:
    #
    #
    # mask = fueleco.make.isin(['Ford', 'Honda', 'Tesla', 'BMW'])
    # fueleco[mask].groupby('make').city08.agg(['mean', 'std'])
    #
    #
    # # In[53]:
    #
    #
    # g = sns.catplot(x='make', y='city08',
    #   data=fueleco[mask], kind='box')
    # g.ax.figure.savefig('/tmp/c5-catbox.png', dpi=300)
    #
    #
    # # ### How it works...
    #
    # # ### There's more...
    #
    # # In[54]:
    #
    #
    # mask = fueleco.make.isin(['Ford', 'Honda', 'Tesla', 'BMW'])
    # (fueleco
    #   [mask]
    #   .groupby('make')
    #   .city08
    #   .count()
    # )
    #
    #
    # # In[55]:
    #
    #
    # g = sns.catplot(x='make', y='city08',
    #   data=fueleco[mask], kind='box')
    # sns.swarmplot(x='make', y='city08',    # doctest: +SKIP
    #   data=fueleco[mask], color='k', size=1, ax=g.ax)
    # g.ax.figure.savefig('/tmp/c5-catbox2.png', dpi=300)    # doctest: +SKIP
    #
    #
    # # In[56]:
    #
    #
    # g = sns.catplot(x='make', y='city08',
    #   data=fueleco[mask], kind='box',
    #   col='year', col_order=[2012, 2014, 2016, 2018],
    #   col_wrap=2)
    # g.axes[0].figure.savefig('/tmp/c5-catboxcol.png', dpi=300)    # doctest: +SKIP
    #
    #
    # # In[57]:
    #
    #
    # g = sns.catplot(x='make', y='city08', # doctest: +SKIP
    #   data=fueleco[mask], kind='box',
    #   hue='year', hue_order=[2012, 2014, 2016, 2018])
    # g.ax.figure.savefig('/tmp/c5-catboxhue.png', dpi=300)    # doctest: +SKIP
    #
    #
    # # In[58]:
    #
    #
    # mask = fueleco.make.isin(['Ford', 'Honda', 'Tesla', 'BMW'])
    # (fueleco
    #   [mask]
    #   .groupby('make')
    #   .city08
    #   .agg(['mean', 'std'])
    #   .style.background_gradient(cmap='RdBu', axis=0)
    # )
    #
    #
    # # ## Comparing Two Continuous Columns
    #
    # # ### How to do it...
    #
    # # In[59]:
    #
    #
    # fueleco.city08.cov(fueleco.highway08)
    #
    #
    # # In[60]:
    #
    #
    # fueleco.city08.cov(fueleco.comb08)
    #
    #
    # # In[61]:
    #
    #
    # fueleco.city08.cov(fueleco.cylinders)
    #
    #
    # # In[62]:
    #
    #
    # fueleco.city08.corr(fueleco.highway08)
    #
    #
    # # In[63]:
    #
    #
    # fueleco.city08.corr(fueleco.cylinders)
    #
    #
    # # In[64]:
    #
    #
    # import seaborn as sns
    # fig, ax = plt.subplots(figsize=(8,8))
    # corr = fueleco[['city08', 'highway08', 'cylinders']].corr()
    # mask = np.zeros_like(corr, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True
    # sns.heatmap(corr, mask=mask,
    #     fmt='.2f', annot=True, ax=ax, cmap='RdBu', vmin=-1, vmax=1,
    #     square=True)
    # fig.savefig('/tmp/c5-heatmap.png', dpi=300, bbox_inches='tight')
    #
    #
    # # In[65]:
    #
    #
    # fig, ax = plt.subplots(figsize=(8,8))
    # fueleco.plot.scatter(x='city08', y='highway08', alpha=.1, ax=ax)
    # fig.savefig('/tmp/c5-scatpan.png', dpi=300, bbox_inches='tight')
    #
    #
    # # In[66]:
    #
    #
    # fig, ax = plt.subplots(figsize=(8,8))
    # fueleco.plot.scatter(x='city08', y='cylinders', alpha=.1, ax=ax)
    # fig.savefig('/tmp/c5-scatpan-cyl.png', dpi=300, bbox_inches='tight')
    #
    #
    # # In[67]:
    #
    #
    # fueleco.cylinders.isna().sum()
    #
    #
    # # In[68]:
    #
    #
    # fig, ax = plt.subplots(figsize=(8,8))
    # (fueleco
    #  .assign(cylinders=fueleco.cylinders.fillna(0))
    #  .plot.scatter(x='city08', y='cylinders', alpha=.1, ax=ax))
    # fig.savefig('/tmp/c5-scatpan-cyl0.png', dpi=300, bbox_inches='tight')
    #
    #
    # # In[69]:
    #
    #
    # res = sns.lmplot(x='city08', y='highway08', data=fueleco)
    # res.fig.savefig('/tmp/c5-lmplot.png', dpi=300, bbox_inches='tight')
    #
    #
    # # ### How it works...
    #
    # # In[70]:
    #
    #
    # fueleco.city08.corr(fueleco.highway08*2)
    #
    #
    # # In[71]:
    #
    #
    # fueleco.city08.cov(fueleco.highway08*2)
    #
    #
    # # ### There's more...
    #
    # # In[72]:
    #
    #
    # res = sns.relplot(x='city08', y='highway08',
    #    data=fueleco.assign(
    #        cylinders=fueleco.cylinders.fillna(0)),
    #    hue='year', size='barrels08', alpha=.5, height=8)
    # res.fig.savefig('/tmp/c5-relplot2.png', dpi=300, bbox_inches='tight')
    #
    #
    # # In[73]:
    #
    #
    # res = sns.relplot(x='city08', y='highway08',
    #   data=fueleco.assign(
    #   cylinders=fueleco.cylinders.fillna(0)),
    #   hue='year', size='barrels08', alpha=.5, height=8,
    #   col='make', col_order=['Ford', 'Tesla'])
    # res.fig.savefig('/tmp/c5-relplot3.png', dpi=300, bbox_inches='tight')
    #
    #
    # # In[74]:
    #
    #
    # fueleco.city08.corr(fueleco.barrels08, method='spearman')
    #
    #
    # # ## Comparing Categorical and Categorical Values
    #
    # # ### How to do it...
    #
    # # In[75]:
    #
    #
    # def generalize(ser, match_name, default):
    #     seen = None
    #     for match, name in match_name:
    #         mask = ser.str.contains(match)
    #         if seen is None:
    #             seen = mask
    #         else:
    #             seen |= mask
    #         ser = ser.where(~mask, name)
    #     ser = ser.where(seen, default)
    #     return ser
    #
    #
    # # In[76]:
    #
    #
    # makes = ['Ford', 'Tesla', 'BMW', 'Toyota']
    # data = (fueleco
    #    [fueleco.make.isin(makes)]
    #    .assign(SClass=lambda df_: generalize(df_.VClass,
    #     [('Seaters', 'Car'), ('Car', 'Car'), ('Utility', 'SUV'),
    #      ('Truck', 'Truck'), ('Van', 'Van'), ('van', 'Van'),
    #      ('Wagon', 'Wagon')], 'other'))
    # )
    #
    #
    # # In[77]:
    #
    #
    # data.groupby(['make', 'SClass']).size().unstack()
    #
    #
    # # In[78]:
    #
    #
    # pd.crosstab(data.make, data.SClass)
    #
    #
    # # In[79]:
    #
    #
    # pd.crosstab([data.year, data.make], [data.SClass, data.VClass])

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

    #
    # cramers_v(data.make, data.SClass)

    # fig, ax = plt.subplots(figsize=(10,8))
    # (data
    #  .pipe(lambda df_: pd.crosstab(df_.make, df_.SClass))
    #  .plot.bar(ax=ax)
    # )
    # fig.savefig('/tmp/c5-bar.png', dpi=300, bbox_inches='tight')

    # res = sns.catplot(kind='count',
    #    x='make', hue='SClass', data=data)
    # res.fig.savefig('/tmp/c5-barsns.png', dpi=300, bbox_inches='tight')

    # fig, ax = plt.subplots(figsize=(10,8))
    # (data
    #  .pipe(lambda df_: pd.crosstab(df_.make, df_.SClass))
    #  .pipe(lambda df_: df_.div(df_.sum(axis=1), axis=0))
    #  .plot.bar(stacked=True, ax=ax)
    # )
    # fig.savefig('/tmp/c5-barstacked.png', dpi=300, bbox_inches='tight')

    # cramers_v(data.make, data.trany)
    # cramers_v(data.make, data.model)

    # import pandas_profiling as pp
    # pp.ProfileReport(fueleco)
    # report = pp.ProfileReport(fueleco)
    # report.to_file('/tmp/fuel.html')
