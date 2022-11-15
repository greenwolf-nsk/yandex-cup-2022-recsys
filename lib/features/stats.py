from const import USE_CUDF

if USE_CUDF:
    import cudf as pd
else:
    import pandas as pd


def get_stats(train_df: pd.DataFrame, ranks: list, column: str):
    stats_dfs = None
    for rank in ranks:
        train_part = train_df[train_df['rank'] <= rank]
        stats = train_part[['user_id', column]].groupby(column).count().reset_index()
        colname = f'{column}_cnt_rnk_{rank}'
        stats.columns = [column, colname]
        if stats_dfs is not None:
            stats_dfs = stats_dfs.merge(stats, how='right')
        else:
            stats_dfs = stats

    return stats_dfs.fillna(0)
