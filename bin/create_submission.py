import argparse

import tqdm
import cudf
import pandas as pd
import lightgbm as lgb

from lib.utils import cast

parser = argparse.ArgumentParser()
parser.add_argument('--candidates')
parser.add_argument('--model')
parser.add_argument('--out')


if __name__ == '__main__':

    args = parser.parse_args()
    test_candidates = pd.read_parquet(args.candidates)
    cast(test_candidates)
    model = lgb.Booster(model_file=args.model)
    preds = model.predict(test_candidates[model.feature_name()])

    prediction_df = test_candidates[['user_id', 'item_id']]
    prediction_df['pred'] = preds

    df = pd.DataFrame(prediction_df)
    df = df.sort_values(['user_id', 'pred'], ascending=[True, False])

    user_recs = []
    for user_id, recs in tqdm.tqdm(df.groupby('user_id')['item_id']):
        user_recs.append(recs.values.tolist()[:100])

    with open(args.out, 'w') as f:
        result = [
            ' '.join(map(str, row)) + '\n'
            for row in user_recs
        ]
        f.writelines(result)
