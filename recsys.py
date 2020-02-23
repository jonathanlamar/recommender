import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors

class RecSys:
    r"""
    Base class for all of my recommender systems in this repo.

    Sets up data pipeline and otherwise does nothing.
    """

    def __init__(self, dataPath='./data'):
        moviesDf = pd.read_csv(dataPath + '/movies.csv').set_index('movieId')
        self.moviesDf = moviesDf.drop('genres', axis=1)

        # Build movies-genres dataframe
        genreList = moviesDf['genres'].apply(lambda x : x.split('|'))
        genresTall = pd.DataFrame(genreList.explode())
        genresTall.columns = ['genre']
        self.genresDf = (genresTall
                         .pivot_table(
                             index=genresTall.index,
                             columns='genre',
                             aggfunc=np.size)
                         .fillna(0)
                         .join(moviesDf['title']))

        # For classic content and collaborative filtering, want a users by
        # movies dataframe of ratings.  Keeping tall and wide versions for
        # different uses.
        self.ratingsTallDf = pd.read_csv(dataPath + '/ratings.csv')
        self.ratingsWideDf = (self.ratingsTallDf
                              .pivot(
                                  index='userId',
                                  columns='movieId',
                                  values='rating')
                              .fillna(0))

        # Sparse matrix representation of ratings matrix
        self.featuresMat = csr_matrix(self.ratingsWideDf.values)

        # Sparse matrix is not indexed by movie id, so need lookup and reverse
        # lookup for both users and movies.
        self.movieIdFromIndex = dict(enumerate(self.ratingsWideDf))
        self.movieIndexFromId = dict(map(lambda tup : (tup[1], tup[0]),
                                         enumerate(self.ratingsWideDf)))

        self.userIdFromIndex = dict(enumerate(self.ratingsWideDf.T))
        self.userIndexFromId = dict(map(lambda tup : (tup[1], tup[0]),
                                        enumerate(self.ratingsWideDf.T)))

        # Using knn with cosine distance for user-user and movie-movie
        # similarity and lookup
        self.userKnn = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=20, # Not sure about this
            n_jobs=-1).fit(self.featuresMat)

        self.movieKnn = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=20, # Not sure about this
            n_jobs=-1).fit(self.featuresMat.transpose())


    def searchMovieTitles(self, searchTerm):
        r"""
        Find closest matching movie title based on fuzzy text search.

        Returns title and movieId as a tuple.
        """
        searchTerm = searchTerm.lower()
        matchRatios = self.moviesDf[['title']]
        matchRatios['ratio'] = (matchRatios['title']
                                .apply(lambda x : x.lower())
                                .apply(lambda x : fuzz.ratio(searchTerm, x)))

        possibleMatches = (matchRatios
                           .query('ratio >= 60')
                           .sort_values('ratio', ascending=False))

        print('Found %d possible matches.' % possibleMatches.shape[0])
        for title in possibleMatches['title']:
            print(title)

        return possibleMatches['title'].values[0], possibleMatches.index[0]
