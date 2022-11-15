from typing import List

import cudf as pd

from lib.utils import get_seqential_user_ids, flatten


def create_df_from_sessions(sessions: List[List[int]]):
    df = pd.DataFrame({
        'user_id': get_seqential_user_ids(sessions),
        'item_id': flatten(sessions),
        'rank': flatten(reversed(range(len(r))) for r in sessions)
    }, dtype='int32')
    return df



def calculate_artists_stats(data: List[List[int]], artists: pd.DataFrame) -> pd.DataFrame:
    df = create_df_from_sessions(data)
    df_with_artists = df.merge(artists)
    grouped = df_with_artists.groupby(['artist_id', 'item_id'])['user_id'].count().reset_index()
    grouped.columns = ['artist_id', 'rec_item_id', 'cnt']
    return grouped


def create_last_artists_recs(
        train_data: List[List[int]],
        val_or_test_data: List[List[int]],
        artists: pd.DataFrame,
        n_items: int = 1,
        num_candidates: int = 100,
) -> pd.DataFrame:
    val_df = create_df_from_sessions([x[-n_items:] for x in val_or_test_data])
    val_df = val_df.merge(artists)

    seen_df = create_df_from_sessions([x for x in val_or_test_data])

    artist_stats = calculate_artists_stats(train_data, artists)

    recs = val_df.merge(artist_stats, on=['artist_id'])
    recs.columns = ['user_id', 'item_id', 'rank', 'artist_id', 'rec_item_id', 'count']

    recs_notseen = recs.merge(
        seen_df,
        left_on=['user_id', 'rec_item_id'],
        right_on=['user_id', 'item_id'],
        how='leftanti'
    ).sort_values(['user_id', 'count'], ascending=[True, False])

    recs_notseen['similar_artist_rank'] = recs_notseen.groupby('user_id')['rec_item_id'].cumcount()
    recs_notseen.sort_values(['user_id', 'similar_artist_rank'])
    top = recs_notseen[recs_notseen['similar_artist_rank'] < num_candidates]
    top = top[['user_id', 'rec_item_id', 'similar_artist_rank', 'count']]
    top.columns = ['user_id', 'item_id', 'last_artist_rank', 'last_artist_score']

    return top


