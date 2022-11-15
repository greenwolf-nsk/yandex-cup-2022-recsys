from typing import List

from const import USE_CUDF
from lib.features.stats import get_stats

if USE_CUDF:
    import cudf as pd
else:
    import pandas as pd

from lib.utils import create_df_from_sessions


def get_user_history_features(
        train_sessions: List[List[int]],
        test_or_val_sessions: List[List[int]],
        artists: pd.DataFrame
) -> pd.DataFrame:
    train_df = create_df_from_sessions(train_sessions)
    train_df_with_artists = train_df.merge(artists)

    item_stats = get_stats(train_df_with_artists, [1, 100], 'item_id')
    artist_stats = get_stats(train_df_with_artists, [1, 100], 'artist_id')

    df = create_df_from_sessions(test_or_val_sessions)
    df = df.merge(artists).merge(item_stats).merge(artist_stats)

    agg_columns = ['user_id'] + [col for col in df.columns if 'cnt_rnk' in col]
    aggs = df[agg_columns].groupby('user_id').agg(['mean', 'max', 'min']).reset_index()
    aggs.columns = ['_'.join(col).strip('_') for col in aggs.columns]


    return aggs.sort_values('user_id')


def get_user_last_item_stats(
        train_sessions: List[List[int]],
        test_or_val_sessions: List[List[int]],
) -> pd.DataFrame:
    train_df = create_df_from_sessions(train_sessions)
    item_stats = get_stats(train_df, [255], 'item_id')
    full_df = pd.DataFrame()
    for i in range(5):
        df = create_df_from_sessions([[x[-1 - i]] if len(x) > i else [] for x in test_or_val_sessions])
        df = df.merge(item_stats)[['user_id', 'item_id_cnt_rnk_255']].sort_values('user_id')
        full_df['user_id'] = df['user_id']
        full_df[f'item_rank_{i}_popularity'] = df['item_id_cnt_rnk_255']

    return full_df

