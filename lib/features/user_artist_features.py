from typing import List

from const import USE_CUDF

if USE_CUDF:
    import cudf as pd
else:
    import pandas as pd

from lib.utils import create_df_from_sessions


def get_user_artist_stats(test_or_val_sessions: List[List[int]], artist_data: pd.DataFrame) -> pd.DataFrame:
    df = create_df_from_sessions(test_or_val_sessions)
    df = df.merge(artist_data)

    user_artist_stats = df[['user_id', 'artist_id', 'item_id']].groupby(['user_id', 'artist_id']).count().reset_index()
    user_artist_stats.columns = ['user_id', 'artist_id', 'artist_like_count']

    return user_artist_stats
