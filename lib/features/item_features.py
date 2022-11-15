from typing import List

from const import USE_CUDF

if USE_CUDF:
    import cudf as pd
else:
    import pandas as pd

from lib.features.stats import get_stats
from lib.utils import flatten, get_seqential_user_ids


def get_items_popularity_stats(data: List[List[int]], offsets: List[int]) -> pd.DataFrame:
    train_df = pd.DataFrame({
        'user_id': get_seqential_user_ids(data),
        'item_id': flatten(data),
        'rank': flatten(reversed(range(len(r))) for r in data)
    })
    item_stats = get_stats(train_df, offsets, 'item_id')
    return item_stats

