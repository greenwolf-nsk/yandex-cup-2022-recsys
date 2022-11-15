from functools import partial

import tqdm
import polars as pl
import numpy as np


SIMILARITY_AGGS_MAP = {
    'similarity_mean': partial(np.mean, axis=1),
    'similarity_min': partial(np.min, axis=1),
    'similarity_max': partial(np.max, axis=1),
    'similarity_std': partial(np.std, axis=1),

}


def calculate_similarity_stats(model, candidates, history):
    res = model.similarity[candidates] * model.similarity[history].T
    return res.toarray()


def create_similarity_features(
        model,
        candidates,
        history,
        aggs_map: dict
) -> pl.DataFrame:
    features = {
        agg_name: []
        for agg_name in list(aggs_map) + ['item_id', 'user_id']
    }
    for user_id, (user_candidates, user_history) in tqdm.tqdm(enumerate(zip(candidates, history))):
        similarities = calculate_similarity_stats(model, user_candidates, user_history)
        features['item_id'].extend(user_candidates)
        features['user_id'].extend(np.ones_like(user_candidates) * user_id)
        for agg_name, agg_fn in aggs_map.items():
            features[agg_name].extend(agg_fn(similarities))


    return pl.DataFrame(features)
