from typing import List

import tqdm
from more_itertools import chunked

from const import USE_CUDF
from lib.features.stats import get_stats

if USE_CUDF:
    import cudf as pd
    import pandas
else:
    import pandas as pd

from lib.utils import create_df_from_sessions


def create_cooc_stats(df: pd.DataFrame, min_rank_diff: int = -3, max_rank_diff: int = 3, items_per_item: int = 10):
    cooc = df.merge(df, on=['user_id']).query('rank_x != rank_y')
    coocf = cooc[(cooc.rank_y - cooc.rank_x).between(min_rank_diff, max_rank_diff)]
    groups = coocf.groupby(['item_id_x', 'item_id_y'])['user_id'].count().to_frame().reset_index()
    groups.columns = ['item_id', 'rec_item_id', 'count']
    groups = groups.sort_values(['item_id', 'count'], ascending=[True, False])
    groups['rnk'] = groups.groupby('item_id').cumcount()
    groups_filtered = groups[groups['rnk'] < items_per_item]

    return groups_filtered.set_index(['item_id', 'rec_item_id'])


def get_item_popularity_stats(df: pd.DataFrame, popularity_last_items: int):
    pop = df[df['rank'] <= popularity_last_items].groupby('item_id')['rank'].count().reset_index().sort_values('item_id')
    pop.columns = ['rec_item_id', 'item_popularity']
    return pop


def create_cooc_stats_by_chunks(
        train_sessions: List[List[int]],
        chunk_size: int = 5_000_000,
        popularity_last_items: int = 100,
        min_rank_diff: int = 1,
        max_rank_diff: int = 3,
        items_per_item: int = 200,
):
    df = create_df_from_sessions(train_sessions)
    item_popularity = get_item_popularity_stats(df, popularity_last_items)
    cooc_stats = None
    for part in tqdm.tqdm(range(df.shape[0] // chunk_size + 1)):
        chunk_stats = create_cooc_stats(
            df[part * chunk_size: (part + 1) * chunk_size],
            min_rank_diff,
            max_rank_diff,
            items_per_item,
        )
        if cooc_stats is None:
            cooc_stats = chunk_stats
        else:
            cooc_stats = cooc_stats.add(chunk_stats, fill_value=0)

    cooc_stats = cooc_stats.reset_index()
    cooc_stats = cooc_stats.sort_values(['item_id', 'count'], ascending=[True, False])
    cooc_stats['rnk'] = cooc_stats.groupby('item_id').cumcount()
    cooc_stats = cooc_stats.merge(item_popularity)
    cooc_stats['cooc_norm'] = cooc_stats['count'] / cooc_stats['item_popularity']
    del cooc_stats['item_popularity']

    cooc_stats.columns = ['item_id', 'rec_item_id', 'cooc_count', 'cooc_rnk', 'cooc_norm']

    return cooc_stats


def get_user_history_cooc_features(
        val_or_test_candidates: pd.DataFrame,
        val_or_test_history: pd.DataFrame,
        cooc_stats: pd.DataFrame,
        chunk_size: int = 1000,
) -> pd.DataFrame:
    num_users = val_or_test_history.user_id.nunique()
    all_cooc = []
    for user_ids in tqdm.tqdm(chunked(range(num_users), chunk_size), total=num_users // chunk_size):
        user_candidates = val_or_test_candidates[val_or_test_candidates.user_id.isin(user_ids)]
        user_history = val_or_test_history[val_or_test_history.user_id.isin(user_ids)]
        uhm = user_history.merge(cooc_stats)
        uhm['ccw'] = uhm['cooc_count'] / (uhm['rank'] + 1)
        user_history_cooc = (
            uhm
            .groupby(['user_id', 'rec_item_id'])[['ccw', 'cooc_count', 'cooc_rnk', 'cooc_norm']]
            .agg(['sum', 'mean', 'max', 'min', 'std'])
            .reset_index()
        )
        user_history_cooc.columns = ['_'.join(x).strip('_') for x in user_history_cooc.columns]
        user_history_cooc.rename(columns={'rec_item_id': 'item_id'}, inplace=True)
        user_cooc_features = user_candidates.merge(user_history_cooc)
        if USE_CUDF:
            all_cooc.append(user_cooc_features.to_pandas())
        else:
            all_cooc.append(user_cooc_features)

    return pandas.concat(all_cooc)
