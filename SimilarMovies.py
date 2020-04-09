import numpy as np
import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
m_cols = ['movie_id', 'title']
ratings = pd.read_csv('u.data', sep = '\t', names = r_cols, usecols = range(3), encoding = "ISO-8859-1")
movies = pd.read_csv('u.item', sep = '|', names = m_cols, usecols = range(2), encoding = "ISO-8859-1")
ratings = pd.merge(movies, ratings)
print(ratings.head())

# Matrix (user x movie rating)
movieRatings = ratings.pivot_table(index = ['user_id'], columns = ['title'], values = 'rating')
print(movieRatings.head())

starWarsRatings = movieRatings['Star Wars (1977)']
print(starWarsRatings.head())

# Pairwise correlation
similarMovies = movieRatings.corrwith(starWarsRatings)
similarMovies = similarMovies.dropna()
df = pd.DataFrame(similarMovies)
print(df.head(10))
print(similarMovies.sort_values(ascending = False))

movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
print(movieStats.head())
# Filter movies rated by at least 100 people
popularMovies = movieStats['rating']['size'] >= 100
print(movieStats[popularMovies].sort_values([('rating', 'mean')], ascending = False)[:15])

# Popular movies joined with similar movies
df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns = ['similarity']))
print(df.head())
# Sorted by similar movies
print(df.sort_values(['similarity'], ascending = False)[:15])
