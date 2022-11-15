from collections import defaultdict
from operator import itemgetter
from typing import List


def get_list_of_recs(rec_lines: List[str]):
    return [[x for x in s.strip().split(' ')] for s in rec_lines]


def blend(*preds, weights: list, topk: int = 100) -> str:
    res = defaultdict(float)
    for i in range(len(preds)):
        for rank, item_id in enumerate(preds[i]):
            res[item_id] += weights[i] / (rank + 1)


    res = list(dict(sorted(res.items(), key=itemgetter(1))).keys())

    return ' '.join(res[:topk])


if __name__ == '__main__':
    with open('submission_a.csv', 'r') as f:
        sub_a = get_list_of_recs(f.readlines())

    with open('submission_b.csv', 'r') as f:
        sub_b = get_list_of_recs(f.readlines())

    with open('submission_ab.csv', 'w') as f:
        for i in range(len(sub_a)):
            f.write(blend(sub_a, sub_b, [0.55, 0.45]) + '\n')
