import argparse
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm.callback import early_stopping
from sklearn.model_selection import GroupKFold

from lib.utils import get_group_for_lgb, downsample_negative, lgb_mrr_wrapper

parser = argparse.ArgumentParser()
parser.add_argument('--candidates')
parser.add_argument('--downsample', type=float)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--binary_target', type=int)


def downsample_negative_lgb_dataset(train: lgb.Dataset, keep: float = 0.3):
    np.random.seed(33)
    train.construct()
    label = train.get_label()
    negatives = np.where(label == 0)[0]
    positives = np.where(label == 1)[0]
    num_negatives = int(np.ceil(len(negatives) * keep))
    np.random.shuffle(negatives)
    new_negatives = negatives[:num_negatives]
    index = np.concatenate([positives, new_negatives])
    index.sort()
    return train.subset(index)


def preproc(train: lgb.Dataset, test: lgb.Dataset, params: dict):
    return downsample_negative_lgb_dataset(train, 0.3), test, params


if __name__ == '__main__':
    args = parser.parse_args()
    train = pd.read_parquet(args.candidates)

    columns = [c for c in train.columns if c not in ['target', 'user_id', 'item_id', 'artist_id']]
    X_train = train[columns].to_numpy(dtype=np.float32)
    y_train = train['target'].astype(np.int32) if args.binary_target else train['target']
    train_groups = get_group_for_lgb(train.user_id.values)

    lgb_train = lgb.Dataset(
        X_train, y_train,
        feature_name=columns,
        group=train_groups,
    )

    params = {
        "objective": "lambdarank",
        "metric": "None",
        "eval_at": 100,
        "boosting_type": "gbdt",
        "is_unbalance": False,
        "learning_rate": 0.04,
        'lambda_l1': 0.9822808040431535,
        'lambda_l2': 7.840380968160073,
        'feature_fraction': 0.6118362309488043,
        'bagging_fraction': 0.6133731265749973,
        'bagging_freq': 1,
    }

    cv_results = lgb.cv(
        params,
        lgb_train,
        num_boost_round=1000,
        folds=GroupKFold(3),
        shuffle=False,
        stratified=False,
        feval=lgb_mrr_wrapper,
        fpreproc=preproc,
        callbacks=[
            early_stopping(50),
            lgb.log_evaluation(10)
        ],
        return_cvbooster=True,
    )

    for i, booster in enumerate(cv_results['cvbooster'].boosters):
        booster.save_model(f'booster_fold{i}.lgb')

    mean_mrr = cv_results['valid mrr-mean'][-1]
    std_mrr = cv_results['valid mrr-stdv'][-1]

    metrics = {'mrr': mean_mrr, 'std': std_mrr, 'candidates_recall': train.target.sum() / train.user_id.nunique()}
    with open('cv_metrics.json', 'w') as f:
        json.dump(metrics, f)

