import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors
from datetime import datetime, timedelta, timezone

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


    def showUserTaste(self, userId, numToShow=10):
        r"""
        Prints a selectiong of the user's top-rated movies.

        Returns nothing.
        """
        topMoviesForUser = (self.ratingsTallDf
                            .query('userId == %d' % userId)
                            .query('rating > 0')
                            .drop(['userId', 'timestamp'], axis=1)
                            .sort_values('rating', ascending=False)
                            .set_index('movieId')
                            .join(self.moviesDf)
                            .iloc[:numToShow])

        if topMoviesForUser.shape[0] >= numToShow:
            print('Here are the top 10 rated movies for user %d.' % userId)
        else:
            print('Here are the only movies user %d has rated' % userId)

        for movId in topMoviesForUser.index:
            rating = topMoviesForUser.loc[movId, 'rating']
            title = self.moviesDf.loc[movId, 'title']

            print('%s, rating = %d' % (title, rating))

        return topMoviesForUser


    def getFavoriteGenres(self, userId):
        r"""
        Returns a pandas Series representing the user's genre affinity.

        The affinity vector is a weighted average of all the movies rated by the
        user, by the rating.
        """

        allRatingsByUser = self.ratingsTallDf.query('userId == %d' % userId)

        mu = allRatingsByUser['rating'].mean()
        std = allRatingsByUser['rating'].std()
        criticalRating = min(5, mu + std)
        print('Ratings by user %d have mean %1.2f and standard deviation %1.2f'
              % (userId, mu, std))
        print('Critical rating: %1.2f' % criticalRating)

        allRatingsByUser['high_rating'] = (allRatingsByUser['rating'] >= criticalRating)

        genresAndRating = (allRatingsByUser
                           .query('high_rating == True')
                           .loc[:, ['movieId']]
                           .merge(self.genresDf,
                                  left_on='movieId',
                                  right_index=True,
                                  how='left'))

        print('User %d has %d high rated movies.  Averaging their genre information.'
              % (userId, genresAndRating.shape[0]))

        genreProfile = (genresAndRating
                        .drop('movieId', axis=1)
                        .mean()
                        .sort_values(ascending=False))

        return genreProfile


    def _parseTimestamp(self, timestamp):
        return datetime.fromtimestamp(timestamp, timezone.utc)


    # TODO: Cast this to units that make sense - maybe years
    def _getSecondsSinceRating(self, timestamp, unit='Days'):
        dt = self._parseTimestamp(timestamp)
        maxTs = self._parseTimestamp(self.ratingsTallDf['timestamp'].max())

        diff = maxTs - dt

        return float(diff.total_seconds())


    def _parseTimestampAsStr(self, timestamp, fmt="%Y-%m-%d %H:%M:%S"):
        d = self._parseTimestamp(timestamp)

        return d.strftime(fmt)
