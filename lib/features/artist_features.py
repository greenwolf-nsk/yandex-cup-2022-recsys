from typing import List

from const import USE_CUDF

if USE_CUDF:
    import cudf as pd
else:
    import pandas as pd

from lib.features.stats import get_stats
from lib.utils import flatten, get_seqential_user_ids


def get_artists_popularity_stats(data: List[List[int]], artists_df: pd.DataFrame, offsets: List[int]) -> pd.DataFrame:
    train_df = pd.DataFrame({
        'user_id': get_seqential_user_ids(data),
        'item_id': flatten(data),
        'rank': flatten(reversed(range(len(r))) for r in data)
    })
    train_df = train_df.merge(artists_df)
    artists_stats = get_stats(train_df, offsets, 'artist_id')
    return artists_stats


def get_artist_item_stats(data: List[List[int]], artists_df: pd.DataFrame) -> pd.DataFrame:
    train_df = pd.DataFrame({
        'user_id': get_seqential_user_ids(data),
        'item_id': flatten(data),
        'rank': flatten(reversed(range(len(r))) for r in data)
    })
    train_df = train_df.merge(artists_df)
    ais = train_df.groupby(['artist_id', 'item_id'])['user_id'].count().reset_index()
    ais.columns = ['artist_id', 'item_id', 'artist_item_count']
    ais = ais.sort_values(['artist_id', 'artist_item_count'], ascending=[True, False])
    ais['artist_item_rank'] = ais.groupby('artist_id', sort=False).cumcount()
    ais['artist_item_count_norm'] = ais['artist_item_count'] / ais.groupby('artist_id')['artist_item_count'].transform('sum')
    del ais['artist_item_count']

    return ais

