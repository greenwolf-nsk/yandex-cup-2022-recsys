import argparse

from implicit.nearest_neighbours import CosineRecommender

from const import pd, N_ARTISTS

from lib.features.user_history_als_features import get_candidates_lists_from_df
from lib.features.user_history_artist_similarity_features import create_artist_history_features
from lib.utils import read_data, create_item_to_artist_map, map_item_sequences_to_artists, spm

parser = argparse.ArgumentParser()
parser.add_argument('--train_data')
parser.add_argument('--val_or_test_data')
parser.add_argument('--artists')
parser.add_argument('--candidates')

parser.add_argument('--out')

if __name__ == '__main__':
    args = parser.parse_args()
    artists = pd.read_parquet(args.artists)
    item_to_artists_map = create_item_to_artist_map(artists)
    sorted_candidates = (
        pd.read_parquet(args.candidates)[['user_id', 'item_id']]
        .merge(artists)
        .sort_values('user_id')
        .drop_duplicates(['user_id', 'artist_id'])
    )
    train_artists = map_item_sequences_to_artists(read_data(args.train_data), item_to_artists_map)
    cand_artists = get_candidates_lists_from_df(sorted_candidates, 'artist_id')
    history_artists = map_item_sequences_to_artists(read_data(args.val_or_test_data), item_to_artists_map)

    similarity_model = CosineRecommender(K=500)
    similarity_model.fit(spm(train_artists, N_ARTISTS))
    features = create_artist_history_features(
        similarity_model,
        cand_artists,
        history_artists,
    )
    features.write_parquet(args.out)
