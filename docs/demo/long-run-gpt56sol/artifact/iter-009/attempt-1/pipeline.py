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
OP_CONFIG = {"dur_buckets": 10, "features": ["hour", "user_gap"], "hp": {"aux_weight": 0.1, "batch": 8192, "epochs": 3, "k": 32, "l2": 1e-06, "lr": 0.0005, "n_neg": 4}, "objective": "bpr_censored_watch"}


def build_data_view():
    return SO.build_data_view()
# <<<END:data_view>>>


# <<<BLOCK:features>>>
def build_features(splits, train_idx):
    return SO.build_features(splits, train_idx, OP_CONFIG)
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
    train_idx = np.asarray(train_idx, dtype=np.int64)
    train_pos = np.arange(train_idx.size, dtype=np.int64)
    tab6_mask = np.asarray(splits['train'].tab)[train_idx] == 6

    raw_idx = np.concatenate([train_idx, train_idx[tab6_mask]])
    raw_source_pos = np.concatenate([train_pos, train_pos[tab6_mask]])
    order = np.argsort(raw_idx, kind='stable')
    expanded_train_idx = raw_idx[order]
    expanded_source_pos = raw_source_pos[order]

    def replicate_target(obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim > 0 and obj.shape[0] == train_idx.size:
                return obj[expanded_source_pos]
            return obj
        if isinstance(obj, dict):
            return {key: replicate_target(value) for key, value in obj.items()}
        if isinstance(obj, tuple):
            values = [replicate_target(value) for value in obj]
            if hasattr(obj, '_fields'):
                return type(obj)(*values)
            return tuple(values)
        if isinstance(obj, list):
            if len(obj) == train_idx.size and all(np.isscalar(value) for value in obj):
                return [obj[int(pos)] for pos in expanded_source_pos]
            return [replicate_target(value) for value in obj]
        return obj

    expanded_y = replicate_target(y)
    config = dict(OP_CONFIG)
    config['hp'] = dict(HP)
    return SO.train(splits, expanded_train_idx, Xs, dim, expanded_y,
                    seed, config, verbose)
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
