import numpy as np
import pandas as pd

class BasicRecommendations:
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
        pass


    def topUnratedMovies(self, userId, numRecs=10):
        r"""
        Find most similar users by cosine similarity of user-item ratings
        vectors and recommend the top rated movies among those users which have
        not been reviewed by userId.
        """
        pass
