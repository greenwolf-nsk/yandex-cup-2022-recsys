from functools import partial
from typing import List

import tqdm
import polars as pl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from const import pd, USE_CUDF

DEFAULT_AGGS_MAP = {
    'als_similarity_mean': partial(np.mean, axis=1),
    'als_similarity_min': partial(np.min, axis=1),
    'als_similarity_max': partial(np.max, axis=1),
    'als_similarity_std': partial(np.std, axis=1),

}


def create_als_similarity_features(
        als_item_factors: np.ndarray,
        candidates: List[np.ndarray],
        history: List[List[int]],
        aggs_map: dict
) -> pl.DataFrame:
    features = {
        agg_name: []
        for agg_name in list(aggs_map) + ['item_id', 'user_id']
    }
    for user_id, (user_candidates, user_history) in tqdm.tqdm(enumerate(zip(candidates, history))):
        similarities = calculate_similarities(als_item_factors, user_candidates, user_history)
        features['item_id'].extend(user_candidates)
        features['user_id'].extend(np.ones_like(user_candidates) * user_id)
        for agg_name, agg_fn in aggs_map.items():
            features[agg_name].extend(agg_fn(similarities))


    return pl.DataFrame(features)


def get_candidates_lists_from_df(candidates_df_sorted: pd.DataFrame, key: str = 'item_id') -> List[np.ndarray]:
    candidate_items = []
    for _, items in candidates_df_sorted[['user_id', key]].groupby('user_id', sort=False)[key]:
        if USE_CUDF:
            candidate_items.append(items.values.get())
        else:
            candidate_items.append(items.values)

    return candidate_items


def calculate_similarities(item_factors: np.ndarray, user_candidates: np.ndarray, user_history: List[int]):
    return cosine_similarity(item_factors[user_candidates], item_factors[user_history])
