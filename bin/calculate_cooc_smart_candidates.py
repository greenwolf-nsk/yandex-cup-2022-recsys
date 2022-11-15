import argparse

from const import pd

from lib.candidates.cooc_smart import get_smart_cooc_candidates_by_chunks
from lib.utils import read_data, create_df_from_sessions

parser = argparse.ArgumentParser()
parser.add_argument('--val_or_test_data')
parser.add_argument('--cooc_stats')
parser.add_argument('--history_last_items', type=int)
parser.add_argument('--num_candidates', type=int)
parser.add_argument('--out')


if __name__ == '__main__':
    args = parser.parse_args()
    val_or_test_data = read_data(args.val_or_test_data)
    history_df = create_df_from_sessions(val_or_test_data)
    cooc_stats = pd.read_parquet(args.cooc_stats)
    cooc_candidates = get_smart_cooc_candidates_by_chunks(
        history_df,
        cooc_stats,
        args.history_last_items,
        args.num_candidates,
    )
    cooc_candidates.to_parquet(args.out)



