import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors

class InitConfig:
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
                         .fillna(0))

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


    def searchMovieTitles(self, searchTerm, showMatches=True):
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
                           .sort_values('ratio', ascending=False)
                           .iloc[:10])

        if showMatches:
            print('Found %d possible matches.' % possibleMatches.shape[0])
            for title in possibleMatches['title']:
                print(title)

        return possibleMatches['title'].values[0], possibleMatches.index[0]


    def showUserTaste(self, userId):
        r"""
        Prints a selectiong of the user's top-rated movies.

        Returns nothing.
        """
        moviesRatedFive = (self.ratingsTallDf
                           .query('userId == %d' % userId)
                           .query('rating == 5')
                           .set_index('movieId'))

        topMoviesForUser = (self.ratingsTallDf
                            .query('userId == %d' % userId)
                            .query('rating > 0')
                            .sort_values('rating', ascending=False)
                            .set_index('movieId'))

        if moviesRatedFive.shape[0] >= 10:
            df = moviesRatedFive.sample(10)
            print('Here are some movies with 5 star ratings by user %d.'
                  % userId)
        elif topMoviesForUser.shape[0] >= 10:
            df = topMoviesForUser.iloc[:10]
            print('Here are the top 10 rated movies for user %d.' % userId)
        else:
            df = topMoviesForUser
            print('Here are the only movies user %d has rated' % userId)

        for movId in df.index:
            rating = df.loc[movId, 'rating']
            title = self.moviesDf.loc[movId, 'title']
            print('%s, rating = %d' % (title, rating))


    def getFavoriteGenres(self, userId):
        r"""
        Returns a pandas Series representing the user's genre affinity.

        The affinity vector is a weighted average of all the movies rated by the
        user, by the rating.
        """

        genresAndRating = (self.ratingsTallDf
                           .query('userId == %d' % userId)
                           .loc[:, ['movieId', 'rating']]
                           .merge(self.genresDf,
                                  left_on='movieId',
                                  right_index=True,
                                  how='left'))

        # Scale by rating for weighted average.
        for c in self.genresDf.columns:
            genresAndRating[c] *= genresAndRating['rating']

        weightedAvg = (genresAndRating
                       .drop(['rating', 'movieId'], axis=1)
                       .mean()
                       .sort_values(ascending=False))

        return weightedAvg
