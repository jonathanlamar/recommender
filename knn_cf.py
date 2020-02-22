import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

# A dumb model.  Simply return nearest neighbors by cosine distance in ratings
# space as recommendations.  I just wanted to get something as a baseline.

class KnnContentFiltering:
    def __init__(self, movies_df, ratings_df):
        # Pivot ratings_df to form matrix.  Also create two lookup tables for
        # grabbing row number from movie id and vice versa
        features_df = ratings_df.pivot(
            index='movieId',
            columns='userId',
            values='rating'
        ).fillna(0)

        # Sparse matrix representation of user-rating matrix
        self.features_mat = csr_matrix(features_df.values)

        self.movies_df = movies_df.set_index('movieId')[['title']]

        self.movie_lookup = {
            mov_id : ind for ind, mov_id in enumerate(features_df.index)
        }
        self.reverse_lookup = dict(enumerate(features_df.index))

        # Use cosine distance of ratings vectors as metric in space of movies
        self.model = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=20,
            n_jobs=-1)

        # Necessary? We aren't using this model to predict...
        self.model.fit(self.features_mat)


    def get_recommendations(self, search_term, num_recs=10):
        _, mov_id = self.search_db(search_term)
        ind = self.movie_lookup[mov_id]

        distances, indices = self.model.kneighbors(
            self.features_mat[ind],
            n_neighbors=num_recs+1)

        # Cast to list and ignore the first element
        distances = list(distances.reshape(-1))[1:]
        indices = list(indices.reshape(-1))[1:]

        mov_ids = [self.reverse_lookup[i] for i in indices]

        print('Recommending the following %d movies' % num_recs)
        for i, d in zip(mov_ids, distances):
            title = self.movies_df.loc[i, 'title']
            print('%s: dist = %.4f.' % (title, d))


    def search_db(self, search_term):
        df = self.movies_df.copy()
        df['ratio'] = df['title'].apply(lambda x : fuzz.ratio(search_term.lower(), x.lower()))
        df = (df.query('ratio >= 60')
              .sort_values('ratio', ascending=False))

        print('Found %d possible matches.' % df.shape[0])
        for title in df['title']:
            print(title)

        return df['title'].values[0], df.index[0]
