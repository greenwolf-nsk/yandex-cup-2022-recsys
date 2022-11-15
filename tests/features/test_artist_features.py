import pandas as pd

from lib.features.artist_features import get_artist_item_stats


def test_get_item_ranks():
    data = [[1, 2, 3], [2, 3, 4], [1, 2, 4]]
    artists_df = pd.DataFrame({'item_id': [1, 2, 3, 4], 'artist_id': [1, 1, 2, 2]})
    artist_stats = get_artist_item_stats(data, artists_df)
    assert artist_stats.columns.tolist() == ['artist_id', 'item_id', 'artist_item_rank', 'artist_item_count_norm']
    assert artist_stats['artist_item_rank'].values.tolist() == [0, 1, 0, 1]
    assert artist_stats['artist_item_count_norm'].values.tolist() == [0.6, 0.4, 0.5, 0.5]
