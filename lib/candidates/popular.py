from typing import List

from collections import Counter

import pandas as pd

from lib.utils import get_seqential_user_ids


class PopularRecommender:

    def __init__(
            self,
            top_items: List[int],
    ):
        self.top_items = top_items

    def recommend(self, data: List[List[int]]) -> pd.DataFrame:
        recs = [
            self._recommend_one_user(user_data)
            for user_data in data
        ]

        return pd.DataFrame({
            'user_id': get_seqential_user_ids(recs),
            'item_id': [item for rec in recs for item in rec],
            'popular_rank': [rnk for rec in recs for rnk, _ in enumerate(rec)],
        })

    def _recommend_one_user(self, user_data: List[int]):
        us = set(user_data)

        return [item for item in self.top_items if item not in us]


def train_popular_model(data: List[List[int]], max_candidates: int = 200):
    counts = Counter([item for user in data for item in user])
    top = counts.most_common(max_candidates)
    top_items = [x[0] for x in top]


    return PopularRecommender(top_items)


