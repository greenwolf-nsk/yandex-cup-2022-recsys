import argparse

from lib.features.user_history_cooc_features import create_cooc_stats_by_chunks
from lib.utils import read_data

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--popularity_last_items', type=int)
parser.add_argument('--min_rank_diff', type=int)
parser.add_argument('--max_rank_diff', type=int)
parser.add_argument('--items_per_item', type=int)
parser.add_argument('--out')

if __name__ == '__main__':
    args = parser.parse_args()
    data = read_data(args.data)
    cooc_stats = create_cooc_stats_by_chunks(
        data,
        popularity_last_items=args.popularity_last_items,
        min_rank_diff=args.min_rank_diff or -args.max_rank_diff,
        max_rank_diff=args.max_rank_diff,
        items_per_item=args.items_per_item
    )
    cooc_stats.to_parquet(args.out)



