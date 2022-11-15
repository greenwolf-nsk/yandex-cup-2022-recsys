from functools import partial
from typing import List

import tqdm
import polars as pl
import numpy as np
from implicit.nearest_neighbours import ItemItemRecommender
from numba import jit

SIMILARITY_AGGS_MAP = {
    'artist_similarity_mean': partial(np.mean, axis=0),
    'artist_similarity_min': partial(np.min, axis=0),
    'artist_similarity_max': partial(np.max, axis=0),
    'artist_similarity_std': partial(np.std, axis=0),

}


def calculate_artist_last_item_similarity_stats(
    model: ItemItemRecommender,
    history_artists: np.ndarray,
    candidates_artists: np.ndarray
):
    res = model.similarity[history_artists[-1:]] * model.similarity[candidates_artists].T
    return res.toarray()[0]


def calculate_artist_similarity_stats(
    model: ItemItemRecommender,
    history_artists: np.ndarray,
    candidates_artists: np.ndarray
):
    res = model.similarity[history_artists] * model.similarity[candidates_artists].T
    return res.toarray()


# user artist share
# same artist current streak +
# last artist action +
# artist similarity stats +-

@jit(nopython=True)
def get_last_seen_index(history_artists: np.ndarray, candidates_artists: np.ndarray):
    results = []
    for artist in candidates_artists:
        for i in range(len(history_artists) - 1, -1, -1):
            if artist == history_artists[i]:
                results.append(i)
                break
        else:
            results.append(-1)

    return results


@jit(nopython=True)
def get_artist_streak(history_artists: np.ndarray, candidates_artists: np.ndarray):
    results = []
    for artist in candidates_artists:
        streak = 0
        for i in range(len(history_artists) - 1, -1, -1):
            if artist == history_artists[i]:
                streak += 1
            else:
                results.append(streak)
                break
        else:
            results.append(streak)

    return results



def create_artist_history_features(
        model: ItemItemRecommender,
        candidates: List[np.ndarray],
        history: List[np.ndarray],
) -> pl.DataFrame:
    base_aggs = ['artist_id', 'user_id', 'artist_last_seen', 'artist_current_streak', 'last_artist_similarity']
    features = {
        agg_name: []
        for agg_name in base_aggs + list(SIMILARITY_AGGS_MAP)
    }
    for user_id, (user_candidates, user_history) in tqdm.tqdm(enumerate(zip(candidates, history))):
        last_item_similarities = calculate_artist_last_item_similarity_stats(model, user_history, user_candidates)
        similarities = calculate_artist_similarity_stats(model, user_history, user_candidates)

        features['artist_id'].extend(user_candidates)
        features['user_id'].extend(np.ones_like(user_candidates) * user_id)
        features['artist_last_seen'].extend(get_last_seen_index(np.array(user_history), np.array(user_candidates)))
        features['artist_current_streak'].extend(get_artist_streak(np.array(user_history), np.array(user_candidates)))
        features['last_artist_similarity'].extend(last_item_similarities)
        for agg_name, agg_fn in SIMILARITY_AGGS_MAP.items():
            features[agg_name].extend(agg_fn(similarities))


    return pl.DataFrame(features)
