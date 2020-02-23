import numpy as np
import pandas as pd

# My stuff
from recsys import RecSys

class BasicRecommendations(RecSys):
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

        # Top recommendation is the movie we're searching, so exclude it
        distances = list(distances.flatten())[1:]
        indexes = list(indexes.flatten())[1:]

        print('Top %d matches:' % numRecs)
        for n, (d, i) in enumerate(zip(distances, indexes)):
            # Get title
            movId = self.movieIdFromIndex[i]
            title = self.moviesDf.loc[movId, 'title']
            print('%d: %s (distance = %4f).' % (n+1, title, d))


    def topUnratedMovies(self, userId, numRecs=10):
        r"""
        Find most similar users by cosine similarity of user-item ratings
        vectors and recommend the top rated movies among those users which have
        not been reviewed by userId.
        """
        pass
