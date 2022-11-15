import pandas
import tqdm
from more_itertools import chunked

from const import pd


def get_smart_cooc_candidates(
        user_history_chunk: pd.DataFrame,
        cooc_stats: pd.DataFrame,
        history_last_items: int,
        max_candidates: int = 1500
) -> pd.DataFrame:
    recent_history = user_history_chunk[user_history_chunk['rank'] < history_last_items]
    user_history_cooc = (
        recent_history
        .merge(cooc_stats)
        .groupby(['user_id', 'rec_item_id'])[['cooc_count']]
        .agg(['sum'])
        .reset_index()
    )
    user_history_cooc.columns = ['user_id', 'item_id', 'cooc_smart_score']  # 'cooc_count_sum'
    cooc_candidates = user_history_cooc.merge(user_history_chunk, how='leftanti')
    cooc_candidates = cooc_candidates.sort_values(['user_id', 'cooc_smart_score'], ascending=[True, False])
    cooc_candidates['cooc_smart_rank'] = cooc_candidates.groupby('user_id')['item_id'].cumcount()

    return cooc_candidates[cooc_candidates['cooc_smart_rank'] < max_candidates]


def get_smart_cooc_candidates_by_chunks(
        val_or_test_history: pd.DataFrame,
        cooc_stats: pd.DataFrame,
        history_last_items: int,
        max_candidates: int,
        chunk_size: int = 1000,
) -> pd.DataFrame:
    num_users = val_or_test_history.user_id.nunique()
    all_candidates = []
    for user_ids in tqdm.tqdm(chunked(range(num_users), chunk_size)):
        user_history_chunk = val_or_test_history[val_or_test_history.user_id.isin(user_ids)]
        candidates = get_smart_cooc_candidates(user_history_chunk, cooc_stats, history_last_items, max_candidates)
        all_candidates.append(candidates.to_pandas())

    return pandas.concat(all_candidates)


