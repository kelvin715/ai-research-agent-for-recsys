"""候选 pipeline —— agent 唯一可以修改的文件。初始内容 = 官方 FM baseline。

结构约定：源码被 `# <<<BLOCK:name>>> ... # <<<END:name>>>` 切成 7 段。
patch_scope 只能引用这些 block 名；一轮最多改 3 个，且必须声明一个 primary_block。
这让 metric delta 的归因是结构性保证，而不是靠 prompt 约束。

数据只能经 /task/dataview.py 取。禁止直接 open CSV —— G2 会在执行前用 AST 拦下。

用法（由 orchestrator 调用，不手动跑）：
    python3 pipeline.py --split test --seed 0 --out pred.npy
"""
import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, '/task')
import dataview as DV

# <<<BLOCK:data_view>>>
# Stable operator adapter. Configuration is controller-generated.
import stable_ops as SO
OP_CONFIG = {"dur_buckets": 10, "features": [], "hp": {"aux_weight": 0.2, "batch": 8192, "epochs": 8, "k": 16, "l2": 1e-06, "lr": 0.001, "n_neg": 1, "num_boost_round": 100}, "model_family": "lightgbm_rank", "objective": "pointwise"}


def build_data_view():
    return SO.build_data_view()
# <<<END:data_view>>>


# <<<BLOCK:features>>>
def build_features(splits, train_idx):
    base_Xs, base_dim = SO.build_features(splits, train_idx, OP_CONFIG)
    train_idx = np.asarray(train_idx, dtype=np.int64)

    def pair_exposure_feature(entity_name):
        features = {
            name: np.zeros(rowset.n, dtype=np.float32)
            for name, rowset in splits.items()
        }
        if train_idx.size == 0:
            return features

        train = splits['train']
        users = np.asarray(train.user_id)[train_idx]
        entities = np.asarray(getattr(train, entity_name))[train_idx]
        dates = np.asarray(train.date)[train_idx]

        order = np.lexsort((dates, entities, users))
        sorted_users = users[order]
        sorted_entities = entities[order]
        sorted_dates = dates[order]
        n = order.size
        positions = np.arange(n, dtype=np.int64)

        pair_start = np.empty(n, dtype=bool)
        pair_start[0] = True
        pair_start[1:] = ((sorted_users[1:] != sorted_users[:-1]) |
                          (sorted_entities[1:] != sorted_entities[:-1]))

        date_start = np.empty(n, dtype=bool)
        date_start[0] = True
        date_start[1:] = (pair_start[1:] |
                          (sorted_dates[1:] != sorted_dates[:-1]))

        pair_origins = np.maximum.accumulate(
            np.where(pair_start, positions, 0)
        )
        date_origins = np.maximum.accumulate(
            np.where(date_start, positions, 0)
        )
        prior_sorted = date_origins - pair_origins
        prior = np.empty(n, dtype=np.float32)
        prior[order] = np.log1p(prior_sorted).astype(np.float32)
        features['train'][train_idx] = prior

        key_dtype = np.dtype([
            ('user', sorted_users.dtype),
            ('entity', sorted_entities.dtype),
        ])
        sorted_keys = np.empty(n, dtype=key_dtype)
        sorted_keys['user'] = sorted_users
        sorted_keys['entity'] = sorted_entities

        starts = np.flatnonzero(pair_start)
        ends = np.empty_like(starts)
        ends[:-1] = starts[1:]
        ends[-1] = n
        unique_keys = sorted_keys[starts]
        totals = (ends - starts).astype(np.float32)

        for split_name, rowset in splits.items():
            if split_name == 'train':
                continue
            query_users = np.asarray(rowset.user_id)
            query_entities = np.asarray(getattr(rowset, entity_name))
            query_keys = np.empty(rowset.n, dtype=key_dtype)
            query_keys['user'] = query_users
            query_keys['entity'] = query_entities

            locations = np.searchsorted(unique_keys, query_keys)
            valid = locations < unique_keys.size
            matched = np.zeros(rowset.n, dtype=bool)
            valid_rows = np.flatnonzero(valid)
            if valid_rows.size:
                matched[valid_rows] = (
                    unique_keys[locations[valid_rows]] == query_keys[valid_rows]
                )
            values = np.zeros(rowset.n, dtype=np.float32)
            matched_rows = np.flatnonzero(matched)
            if matched_rows.size:
                values[matched_rows] = totals[locations[matched_rows]]
            features[split_name] = np.log1p(values).astype(np.float32)

        return features

    video_affinity = pair_exposure_feature('video_id')
    author_affinity = pair_exposure_feature('author_id')

    Xs = {}
    for split_name, base in base_Xs.items():
        extra = np.column_stack((
            video_affinity[split_name],
            author_affinity[split_name],
        )).astype(np.float32, copy=False)
        Xs[split_name] = np.concatenate(
            (np.asarray(base, dtype=np.float32), extra), axis=1
        )

    return Xs, base_dim + 2
# <<<END:features>>>


# <<<BLOCK:target>>>
def build_target(splits, train_idx):
    return SO.build_target(splits, train_idx, OP_CONFIG)
# <<<END:target>>>


# <<<BLOCK:model>>>
sigmoid = SO.sigmoid
FM = SO.FM
# <<<END:model>>>


# <<<BLOCK:loss>>>
loss_and_step = SO.loss_and_step
# <<<END:loss>>>


# <<<BLOCK:train>>>
HP = dict(OP_CONFIG['hp'])


def train(splits, train_idx, Xs, dim, y, seed, verbose=False):
    config = dict(OP_CONFIG)
    config['hp'] = dict(HP)
    return SO.train(splits, train_idx, Xs, dim, y, seed, config, verbose)
# <<<END:train>>>


# <<<BLOCK:predict>>>
def predict(model, Xs, split):
    return SO.predict(model, Xs, split)
# <<<END:predict>>>


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='valid', choices=['valid', 'test'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='pred.npy')
    ap.add_argument('--meta', default='meta.json')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--smoke', type=int, default=0,
                    help='冒烟模式：只用这么多训练行、最多 1 轮。用于在正式多种子评测前'
                         '几秒钟内暴露 NameError/形状错误，不产出可用分数。')
    a = ap.parse_args()

    t0 = time.time()
    splits, train_idx = build_data_view()
    if a.smoke:
        # 脚手架级降配：对 block 内容不做任何假设，只截训练行；若 HP 存在则顺带压 epoch。
        rng = np.random.default_rng(0)
        train_idx = np.sort(rng.choice(train_idx, size=min(a.smoke, len(train_idx)),
                                       replace=False))
        if isinstance(globals().get('HP'), dict):
            HP['epochs'] = 1
            # Common non-NumPy model budgets. This is only a compile/runtime smoke,
            # never a scored experiment, so expensive estimators get a tiny fit.
            for key in ('num_boost_round', 'n_estimators', 'max_iter', 'steps'):
                if key in HP:
                    HP[key] = min(int(HP[key]), 5)
    Xs, dim = build_features(splits, train_idx)
    y = build_target(splits, train_idx)
    model, info = train(splits, train_idx, Xs, dim, y, a.seed, a.verbose)
    pred = predict(model, Xs, a.split)

    n_expect = splits[a.split].n
    assert np.isfinite(pred).all(), '预测含 NaN/Inf'
    np.save(a.out, pred)

    info.update(split=a.split, seed=a.seed, n_rows=int(n_expect),
                feature_dim=dim, wall_s=round(time.time() - t0, 1))
    with open(a.meta, 'w') as fh:
        json.dump(info, fh, indent=2)
    print(json.dumps(info))


if __name__ == '__main__':
    main()
