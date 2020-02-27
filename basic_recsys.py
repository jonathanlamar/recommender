import numpy as np
import pandas as pd

# My stuff
from init_config import InitConfig

class BasicRecSys(InitConfig):
    r"""
    Provides item-item knn-based content filtering and user-item top
    recommendations based on best among unreviewed for similar users.
    """

    def __init__(self, dataPath='./data'):

        # Data wrangling happens here.
        super().__init__(dataPath)


    def recSimilarMovies(self, searchTerm, numRecs=10):
        r"""
        Find most similar movies to searchTerm based on cosine similarity of
        item-user ratings vectors.
        """
        title, movId = self.searchMovieTitles(searchTerm, showMatches=False)

        print('I think you meant \"%s\".' % title)
        print('Searching for movies similar to this.\n')
        ind = self.movieIndexFromId[movId]
        ratingsVec = self.featuresMat.transpose()[ind]

        # Get recs
        distances, indexes = (self.movieKnn
                              .kneighbors(ratingsVec, n_neighbors=numRecs+1))

        # Top recommendation is the movie we're searching, so exclude it.
        distances = list(distances.flatten())[1:]
        indexes = list(indexes.flatten())[1:]
        topMovies = [self.movieIdFromIndex[i] for i in indexes]

        print('Top %d matches:' % numRecs)
        for n, (d, movId) in enumerate(zip(distances, topMovies)):
            title = self.moviesDf.loc[movId, 'title']
            print('%d: %s (distance = %4f).' % (n+1, title, d))

        return pd.Series(distances, index=topMovies, name='cosine similarity')


    def topUnratedMovies(self, userId, numRecs=10, numUsers=5):
        r"""
        Find most similar users by cosine similarity of user-item ratings
        vectors and recommend the top rated movies among those users which
        have not been reviewed by userId.
        """
        ind = self.userIndexFromId[userId]
        ratingsVec = self.featuresMat[ind]

        distances, indexes = (self.userKnn
                              .kneighbors(ratingsVec, n_neighbors=numUsers+1))

        # Most similar user is the one we're searching for, so exclude it.
        distances = list(distances.flatten())[1:]
        indexes = list(indexes.flatten())[1:]

        userIds = [self.userIdFromIndex[i] for i in indexes]

        # Get predictions of unrated movies.
        # 2 steps: transpose, query for unrated movies by userId, then
        # transpose back and filter for user ids and take mean
        df = self.ratingsWideDf.transpose()
        topUnrated = (df
                      .loc[df[userId] == 0]
                      .transpose()
                      .loc[userIds]
                      .mean()
                      .sort_values(ascending=False)
                      .iloc[:numRecs])

        print('Top %d matches based on top %d similar users:'
              % (numRecs, numUsers))
        for n, movId in enumerate(topUnrated.index):
            title = self.moviesDf.loc[movId, 'title']
            avgRating = topUnrated.loc[movId]
            print('%d: %s (predicted rating: %1.2f).'
                  % (n+1, title, avgRating))

        return topUnrated
