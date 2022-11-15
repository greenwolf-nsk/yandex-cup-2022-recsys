from typing import List

from const import USE_CUDF

if USE_CUDF:
    import cudf as pd
else:
    import pandas as pd

from lib.utils import create_df_from_sessions


def get_user_stats(test_or_val_sessions: List[List[int]], artists: pd.DataFrame) -> pd.DataFrame:
    df = create_df_from_sessions(test_or_val_sessions)
    df = df.merge(artists)

    user_stats = df[['user_id', 'artist_id']].groupby(['user_id'])['artist_id'].agg(
        ['count', 'nunique']
    ).reset_index()
    user_stats.columns = ['user_id', 'total_likes', 'uniq_artists']
    user_stats['likes_per_artist'] = user_stats['total_likes'] / user_stats['uniq_artists']

    return user_stats
