import argparse
import json

import polars as pl
import pandas as pd
import lightgbm
from lightgbm.callback import early_stopping

from lib.utils import read_data, get_group_for_lgb, downsample_negative, polars_left_merge, lgb_mrr_wrapper

parser = argparse.ArgumentParser()
parser.add_argument('--candidates')
parser.add_argument('--targets')
parser.add_argument('--downsample', type=float)
parser.add_argument('--metric', type=str)
parser.add_argument('--learning_rate', type=float)


def filter_zero_groups(df: pd.DataFrame):
    target_dict = df.groupby('user_id')['target'].sum().to_dict()
    nz_users = [uid for uid in target_dict if target_dict[uid] > 0]
    return df[df.user_id.isin(nz_users)]


if __name__ == '__main__':
    args = parser.parse_args()
    candidates = pl.read_parquet(args.candidates)

    recs_full = pd.read_parquet(args.candidates)
    num_users = recs_full.user_id.nunique()

    ttrain = recs_full[recs_full.user_id < int(num_users * 0.6)]
    ttest = recs_full[recs_full.user_id.between(int(num_users * 0.6), int(num_users * 0.8))]
    tval = recs_full[int(num_users * 0.8) < recs_full.user_id]

    columns = [c for c in ttrain.columns if c not in ['target', 'user_id', 'item_id', 'artist_id']]
    ttraind = downsample_negative(filter_zero_groups(ttrain), args.downsample)

    X_train, y_train, train_groups = ttraind[columns], ttraind['target'], get_group_for_lgb(ttraind.user_id.values)
    X_val, y_val, val_groups = tval[columns], tval['target'], get_group_for_lgb(tval.user_id.values)
    X_test, y_test, test_groups = ttest[columns], ttest['target'], get_group_for_lgb(ttest.user_id.values)

    lgb_train = lightgbm.Dataset(
        X_train, y_train,
        group=train_groups, free_raw_data=False
    )
    lgb_eval = lightgbm.Dataset(
        X_val, y_val, reference=lgb_train,
        group=val_groups, free_raw_data=False
    )

    params = {
        "objective": "lambdarank",
        "metric": args.metric,
        "eval_at": 100,
        "n_estimators": 1000,
        "boosting_type": "gbdt",
        "is_unbalance": False,
        "learning_rate": args.learning_rate,
        'lambda_l1': 0.9822808040431535,
        'lambda_l2': 7.840380968160073,
        'feature_fraction': 0.6118362309488043,
        'bagging_fraction': 0.6133731265749973,
        'bagging_freq': 1,
    }

    ranker = lightgbm.train(
        params,
        lgb_train,
        valid_sets=[lgb_eval],
        feval=lgb_mrr_wrapper,
        callbacks=[
            early_stopping(50),
            lightgbm.log_evaluation(10)
        ],
    )

    test_pred = ranker.predict(X_test)
    ttest['pred'] = test_pred
    ttest_sorted = ttest.sort_values(['user_id', 'pred'], ascending=[True, False])
    ttest_sorted['rnk'] = ttest_sorted.groupby('user_id').cumcount()

    mrr = (1 / (ttest_sorted[ttest_sorted['target'] == 1].rnk + 1)).sum() / int(num_users * 0.2)

    metrics = {'mrr': mrr, 'candidates_recall': recs_full.target.sum() / num_users}
    with open('train_metrics.json', 'w') as f:
        json.dump(metrics, f)

    ranker.save_model('model.lgb')

