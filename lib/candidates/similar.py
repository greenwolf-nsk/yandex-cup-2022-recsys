from _operator import itemgetter
from typing import List

from collections import defaultdict

import pandas as pd
from implicit.nearest_neighbours import BM25Recommender, TFIDFRecommender, CosineRecommender, ItemItemRecommender

from lib.utils import spm, get_seqential_user_ids

recommenders = {
    'bm25': BM25Recommender,
    'tfidf': TFIDFRecommender,
    'cosine': CosineRecommender,
}


class SimilarRecommender:

    def __init__(self, implicit_model: ItemItemRecommender):
        self.implicit_model = implicit_model

    def recommend(self, data: List[List[int]]) -> pd.DataFrame:
        recs = [
            self._recommend_one_user(user_data)
            for user_data in data
        ]

        return pd.DataFrame({
            'user_id': get_seqential_user_ids(recs),
            'item_id': [item[0] for rec in recs for item in rec],
            'similar_score': [item[1] for rec in recs for item in rec],
            'similar_rank': [rnk for rec in recs for rnk, _ in enumerate(rec)],
        })

    def _recommend_one_user(self, user_data: List[int]):
        user_recs = defaultdict(float)
        for item in user_data[:-2:-1]:
            recs, scores = self.implicit_model.similar_items(item, N=200)
            for rec, score in zip(recs, scores):
                user_recs[rec] += score

        seen = set(user_data)

        return [x for x in sorted(user_recs.items(), key=itemgetter(1), reverse=True) if x not in seen]


def train_similar_model(data: List[List[int]], recommeder_cls: str, max_candidates: int = 500):
    train_matrix = spm(data)
    recommender = recommenders[recommeder_cls](K=max_candidates)
    recommender.fit(train_matrix)

    return SimilarRecommender(recommender)



