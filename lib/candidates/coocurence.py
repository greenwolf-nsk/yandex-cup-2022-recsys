from typing import List, Tuple, Dict
from collections import Counter, defaultdict

from const import USE_CUDF
from lib.utils import get_seqential_user_ids

import polars as pl

if USE_CUDF:
    import cudf as pd
else:
    import pandas as pd


class CoocurenceRecommender:

    def __init__(
            self,
            top_pairs: Dict[int, List[Tuple[int, int]]],
            item_index: int = 0,
            num_recs: int = 300,
    ):
        self.top_pairs = top_pairs
        self.item_index = item_index
        self.num_recs = num_recs

    def recommend(self, data: List[List[int]]) -> pd.DataFrame:
        recs = [
            self._recommend_one_user(user_data)
            for user_data in data
        ]

        return pl.DataFrame({
            'user_id': get_seqential_user_ids(recs),
            'item_id': [item[0] for rec in recs for item in rec],
            f'cooc_score_{self.item_index}': [item[1] for rec in recs for item in rec],
            f'cooc_rank_{self.item_index}': [rnk for rec in recs for rnk, _ in enumerate(rec)],
        })

    def _recommend_one_user(self, user_data: List[int]):
        if len(user_data) <= self.item_index:
            return []
        top_pair_items = self.top_pairs.get(user_data[-self.item_index - 1], [])
        us = set(user_data)

        return [item for item in top_pair_items if item[0] not in us][:self.num_recs]


class CoocurenceTripletRecommender:

    def __init__(
            self,
            top_triples: Dict[Tuple[int, int], List[Tuple[int, int]]],
            item_index: int = 0
    ):
        self.top_triples = top_triples
        self.item_index = item_index

    def recommend(self, data: List[List[int]]) -> pd.DataFrame:
        recs = [
            self._recommend_one_user(user_data)
            for user_data in data
        ]

        return pl.DataFrame({
            'user_id': get_seqential_user_ids(recs),
            'item_id': [item[0] for rec in recs for item in rec],
            f'cooc_triplet_score_{self.item_index}': [item[1] for rec in recs for item in rec],
            f'cooc_triplet_rank_{self.item_index}': [rnk for rec in recs for rnk, _ in enumerate(rec)],
        })

    def _recommend_one_user(self, user_data: List[int]):
        last_items = (user_data[-self.item_index - 2], user_data[-self.item_index - 1])
        top_triplet_items = self.top_triples.get(
            last_items,
            []
        )
        us = set(user_data)

        return [item for item in top_triplet_items if item[0] not in us]


def train_cooc_model(data: List[List[int]], max_candidates: int = 500):
    pairs = defaultdict(Counter)
    top_pairs = {}

    for record in data:
        for i in range(len(record) - 1):
            last, ans = record[i], record[i + 1]
            pairs[last][ans] += 1

    for key in pairs:
        top_pairs[key] = pairs[key].most_common(max_candidates)

    return top_pairs


def train_triplet_model(data: List[List[int]], max_candidates: int = 300):
    triples = defaultdict(Counter)
    top_triples = {}

    for record in data:
        for i in range(len(record) - 2):
            prev, last, ans = record[i], record[i + 1], record[i + 2]
            triples[(prev, last)][ans] += 1

    for key in triples:
        top_triples[key] = triples[key].most_common(max_candidates)

    return top_triples


