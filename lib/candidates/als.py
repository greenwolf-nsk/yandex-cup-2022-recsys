from _operator import itemgetter
from typing import List

from collections import defaultdict

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from more_itertools import chunked

from lib.utils import spm, get_seqential_user_ids


class AlsRecommender:

    def __init__(self, implicit_model: AlternatingLeastSquares):
        self.implicit_model = implicit_model

    def recommend(self, data: List[List[int]], n_recs: int = 100) -> pd.DataFrame:
        all_recs = []
        all_scores = []
        for batch in chunked(data, 100000):
            recs, scores = self.implicit_model.recommend(
                range(len(batch)),
                spm(batch),
                N=n_recs,
                recalculate_user=True,
            )
            all_recs.append(recs)
            all_scores.append(scores)

        recs, scores = np.vstack(all_recs), np.vstack(all_scores)

        return pd.DataFrame({
            'user_id': get_seqential_user_ids(recs),
            'item_id': [item for rec in recs for item in rec],
            'als_score': [item for rec in scores for item in rec],
            'als_rank': [rnk for rec in recs for rnk, _ in enumerate(rec)],
        })



def train_als_model(data: List[List[int]], factors: int = 100, iterations: int = 30):
    train_matrix = spm(data)
    recommender = AlternatingLeastSquares(factors=factors, iterations=iterations)
    recommender.fit(train_matrix)

    return recommender



