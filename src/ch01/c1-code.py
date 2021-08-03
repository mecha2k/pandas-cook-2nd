#!/usr/bin/env python
# coding: utf-8

# # Chapter 1: Pandas Foundations

# In[4]:


import pandas as pd
import numpy as np


# ## Introduction

# ## Dissecting the anatomy of a DataFrame

# In[5]:


import os
print(os.getcwd())

pd.set_option('max_columns', 4, 'max_rows', 10)


# In[6]:


movies = pd.read_csv('../data/movie.csv')
movies.head()


# ### How it works...

# ## DataFrame Attributes

# ### How to do it... {#how-to-do-it-1}

# In[9]:


movies = pd.read_csv('../data/movie.csv')
columns = movies.columns
index = movies.index
data = movies.values


# In[10]:


columns


# In[11]:


index


# In[12]:


data


# In[13]:


type(index)


# In[14]:


type(columns)


# In[15]:


type(data)


# In[16]:


issubclass(pd.RangeIndex, pd.Index)


# ### How it works...

# ### There's more

# In[ ]:


index.values


# In[ ]:


columns.values


# ## Understanding data types

# ### How to do it... {#how-to-do-it-2}

# In[ ]:


movies = pd.read_csv('data/movie.csv')


# In[ ]:


movies.dtypes


# In[ ]:


movies.get_dtype_counts()


# In[ ]:


movies.info()


# ### How it works...

# In[ ]:


pd.Series(['Paul', np.nan, 'George']).dtype


# ### There's more...

# ### See also

# ## Selecting a Column

# ### How to do it... {#how-to-do-it-3}

# In[ ]:


movies = pd.read_csv('data/movie.csv')
movies['director_name']


# In[ ]:


movies.director_name


# In[ ]:


movies.loc[:, 'director_name']


# In[ ]:


movies.iloc[:, 1]


# In[ ]:


movies['director_name'].index


# In[ ]:


movies['director_name'].dtype


# In[ ]:


movies['director_name'].size


# In[ ]:


movies['director_name'].name


# In[ ]:


type(movies['director_name'])


# In[ ]:


movies['director_name'].apply(type).unique()


# ### How it works...

# ### There's more

# ### See also

# ## Calling Series Methods

# In[ ]:


s_attr_methods = set(dir(pd.Series))
len(s_attr_methods)


# In[ ]:


df_attr_methods = set(dir(pd.DataFrame))
len(df_attr_methods)


# In[ ]:


len(s_attr_methods & df_attr_methods)


# ### How to do it... {#how-to-do-it-4}

# In[ ]:


movies = pd.read_csv('data/movie.csv')
director = movies['director_name']
fb_likes = movies['actor_1_facebook_likes']


# In[ ]:


director.dtype


# In[ ]:


fb_likes.dtype


# In[ ]:


director.head()


# In[ ]:


director.sample(n=5, random_state=42)


# In[ ]:


fb_likes.head()


# In[ ]:


director.value_counts()


# In[ ]:


fb_likes.value_counts()


# In[ ]:


director.size


# In[ ]:


director.shape


# In[ ]:


len(director)


# In[ ]:


director.unique()


# In[ ]:


director.count()


# In[ ]:


fb_likes.count()


# In[ ]:


fb_likes.quantile()


# In[ ]:


fb_likes.min()


# In[ ]:


fb_likes.max()


# In[ ]:


fb_likes.mean()


# In[ ]:


fb_likes.median()


# In[ ]:


fb_likes.std()


# In[ ]:


fb_likes.describe()


# In[ ]:


director.describe()


# In[ ]:


fb_likes.quantile(.2)


# In[ ]:


fb_likes.quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9])


# In[ ]:


director.isna()


# In[ ]:


fb_likes_filled = fb_likes.fillna(0)
fb_likes_filled.count()


# In[ ]:


fb_likes_dropped = fb_likes.dropna()
fb_likes_dropped.size


# ### How it works...

# ### There's more...

# In[ ]:


director.value_counts(normalize=True)


# In[ ]:


director.hasnans


# In[ ]:


director.notna()


# ### See also

# ## Series Operations

# In[ ]:


5 + 9    # plus operator example. Adds 5 and 9


# ### How to do it... {#how-to-do-it-5}

# In[ ]:


movies = pd.read_csv('data/movie.csv')
imdb_score = movies['imdb_score']
imdb_score


# In[ ]:


imdb_score + 1


# In[ ]:


imdb_score * 2.5


# In[ ]:


imdb_score // 7


# In[ ]:


imdb_score > 7


# In[ ]:


director = movies['director_name']
director == 'James Cameron'


# ### How it works...

# ### There's more...

# In[ ]:


imdb_score.add(1)   # imdb_score + 1


# In[ ]:


imdb_score.gt(7)   # imdb_score > 7


# ### See also

# ## Chaining Series Methods

# ### How to do it... {#how-to-do-it-6}

# In[ ]:


movies = pd.read_csv('data/movie.csv')
fb_likes = movies['actor_1_facebook_likes']
director = movies['director_name']


# In[ ]:


director.value_counts().head(3)


# In[ ]:


fb_likes.isna().sum()


# In[ ]:


fb_likes.dtype


# In[ ]:


(fb_likes.fillna(0)
         .astype(int)
         .head()
)


# ### How it works...

# ### There's more...

# In[ ]:


(fb_likes.fillna(0)
         #.astype(int)
         #.head()
)


# In[ ]:


(fb_likes.fillna(0)
         .astype(int)
         #.head()
)


# In[ ]:


fb_likes.isna().mean()


# In[ ]:


fb_likes.fillna(0)         .astype(int)         .head()


# In[ ]:


def debug_df(df):
    print("BEFORE")
    print(df)
    print("AFTER")
    return df


# In[ ]:


(fb_likes.fillna(0)
         .pipe(debug_df)
         .astype(int) 
         .head()
)


# In[ ]:


intermediate = None
def get_intermediate(df):
    global intermediate
    intermediate = df
    return df


# In[ ]:


res = (fb_likes.fillna(0)
         .pipe(get_intermediate)
         .astype(int) 
         .head()
)


# In[ ]:


intermediate


# ## Renaming Column Names

# ### How to do it...

# In[ ]:


movies = pd.read_csv('data/movie.csv')


# In[ ]:


col_map = {'director_name':'Director Name', 
             'num_critic_for_reviews': 'Critical Reviews'} 


# In[ ]:


movies.rename(columns=col_map).head()


# ### How it works... {#how-it-works-8}

# ### There's more {#theres-more-7}

# In[ ]:


idx_map = {'Avatar':'Ratava', 'Spectre': 'Ertceps',
  "Pirates of the Caribbean: At World's End": 'POC'}
col_map = {'aspect_ratio': 'aspect',
  "movie_facebook_likes": 'fblikes'}
(movies
   .set_index('movie_title')
   .rename(index=idx_map, columns=col_map)
   .head(3)
)


# In[ ]:


movies = pd.read_csv('data/movie.csv', index_col='movie_title')
ids = movies.index.tolist()
columns = movies.columns.tolist()


# # rename the row and column labels with list assignments

# In[ ]:


ids[0] = 'Ratava'
ids[1] = 'POC'
ids[2] = 'Ertceps'
columns[1] = 'director'
columns[-2] = 'aspect'
columns[-1] = 'fblikes'
movies.index = ids
movies.columns = columns


# In[ ]:


movies.head(3)


# In[ ]:


def to_clean(val):
    return val.strip().lower().replace(' ', '_')


# In[ ]:


movies.rename(columns=to_clean).head(3)


# In[ ]:


cols = [col.strip().lower().replace(' ', '_')
        for col in movies.columns]
movies.columns = cols
movies.head(3)


# ## Creating and Deleting columns

# ### How to do it... {#how-to-do-it-9}

# In[ ]:


movies = pd.read_csv('data/movie.csv')
movies['has_seen'] = 0


# In[ ]:


idx_map = {'Avatar':'Ratava', 'Spectre': 'Ertceps',
  "Pirates of the Caribbean: At World's End": 'POC'}
col_map = {'aspect_ratio': 'aspect',
  "movie_facebook_likes": 'fblikes'}
(movies
   .rename(index=idx_map, columns=col_map)
   .assign(has_seen=0)
)


# In[ ]:


total = (movies['actor_1_facebook_likes'] +
         movies['actor_2_facebook_likes'] + 
         movies['actor_3_facebook_likes'] + 
         movies['director_facebook_likes'])


# In[ ]:


total.head(5)


# In[ ]:


cols = ['actor_1_facebook_likes','actor_2_facebook_likes',
    'actor_3_facebook_likes','director_facebook_likes']
sum_col = movies[cols].sum(axis='columns')
sum_col.head(5)


# In[ ]:


movies.assign(total_likes=sum_col).head(5)


# In[ ]:


def sum_likes(df):
   return df[[c for c in df.columns
              if 'like' in c]].sum(axis=1)


# In[ ]:


movies.assign(total_likes=sum_likes).head(5)


# In[ ]:


(movies
   .assign(total_likes=sum_col)
   ['total_likes']
   .isna()
   .sum()
)


# In[ ]:


(movies
   .assign(total_likes=total)
   ['total_likes']
   .isna()
   .sum()
)


# In[ ]:


(movies
   .assign(total_likes=total.fillna(0))
   ['total_likes']
   .isna()
   .sum()
)


# In[ ]:


def cast_like_gt_actor_director(df):
    return df['cast_total_facebook_likes'] >=            df['total_likes']


# In[ ]:


df2 = (movies
   .assign(total_likes=total,
           is_cast_likes_more = cast_like_gt_actor_director)
)


# In[ ]:


df2['is_cast_likes_more'].all()


# In[ ]:


df2 = df2.drop(columns='total_likes')


# In[ ]:


actor_sum = (movies
   [[c for c in movies.columns if 'actor_' in c and '_likes' in c]]
   .sum(axis='columns')
)


# In[ ]:


actor_sum.head(5)


# In[ ]:


movies['cast_total_facebook_likes'] >= actor_sum


# In[ ]:


movies['cast_total_facebook_likes'].ge(actor_sum)


# In[ ]:


movies['cast_total_facebook_likes'].ge(actor_sum).all()


# In[ ]:


pct_like = (actor_sum
    .div(movies['cast_total_facebook_likes'])
)


# In[ ]:


pct_like.describe()


# In[ ]:


pd.Series(pct_like.values,
    index=movies['movie_title'].values).head()


# ### How it works... {#how-it-works-9}

# ### There's more... {#theres-more-8}

# In[ ]:


profit_index = movies.columns.get_loc('gross') + 1
profit_index


# In[ ]:


movies.insert(loc=profit_index,
              column='profit',
              value=movies['gross'] - movies['budget'])


# In[ ]:


del movies['director_name']


# ### See also
