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
    Xs, dim = SO.build_features(splits, train_idx, OP_CONFIG)

    major_tabs = np.array([0, 1, 2, 4, 6], dtype=np.int64)
    out = {}
    add_dim = 1 + len(major_tabs) + len(major_tabs)

    for split, X in Xs.items():
        rs = splits[split]
        tab = np.asarray(rs.tab)
        dur = np.asarray(rs.duration_ms, dtype=np.float32)
        log_dur = np.log1p(np.clip(dur, 0.0, None)).astype(np.float32)

        extra = np.zeros((rs.n, add_dim), dtype=np.float32)
        extra[:, 0] = log_dur
        for j, t in enumerate(major_tabs):
            flag = (tab == t).astype(np.float32)
            extra[:, 1 + j] = flag
            extra[:, 1 + len(major_tabs) + j] = flag * log_dur

        out[split] = np.concatenate([X, extra], axis=1).astype(np.float32, copy=False)

    return out, dim + add_dim
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
    base_model, info = SO.train(splits, train_idx, Xs, dim, y, seed, config, verbose)

    from sklearn.linear_model import LogisticRegression

    n_extra = 11
    base_dim = dim - n_extra

    y_arr = np.asarray(y)
    if y_arr.shape[0] == len(train_idx):
        y_train = np.asarray(y_arr, dtype=np.float32)
    else:
        y_train = np.asarray(y_arr[train_idx], dtype=np.float32)
    DV.assert_trainable(y_train, 'residual_head')

    X_train_extra = np.asarray(Xs['train'][train_idx, base_dim:base_dim + n_extra], dtype=np.float32)

    mean = X_train_extra.mean(axis=0).astype(np.float32)
    scale = np.maximum(X_train_extra.std(axis=0), 1e-6).astype(np.float32)

    residual = {
        'mean': mean,
        'scale': scale,
        'coef': np.zeros(n_extra, dtype=np.float32),
        'intercept': 0.0,
        'weight': 0.15,
    }

    if X_train_extra.shape[0] > 0 and np.unique(y_train).size >= 2:
        Xz = (X_train_extra - mean) / scale
        clf = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='liblinear',
            max_iter=200,
            random_state=seed,
        )
        clf.fit(Xz, y_train.astype(np.int32))
        residual['coef'] = clf.coef_.reshape(-1).astype(np.float32)
        residual['intercept'] = float(clf.intercept_[0])
        head_status = 'logreg'
    else:
        head_status = 'degenerate_zero'

    info = dict(info)
    info['residual_head'] = head_status
    info['residual_weight'] = residual['weight']
    info['residual_extra_dim'] = n_extra
    return {'base_model': base_model, 'residual': residual, 'base_dim': base_dim}, info
# <<<END:train>>>


# <<<BLOCK:predict>>>
def predict(model, Xs, split):
    if not isinstance(model, dict) or 'base_model' not in model:
        return SO.predict(model, Xs, split)

    base_pred = SO.predict(model['base_model'], Xs, split)
    residual = model['residual']
    n_extra = residual['coef'].shape[0]
    start = int(model['base_dim'])
    X_extra = np.asarray(Xs[split][:, start:start + n_extra], dtype=np.float32)
    Xz = (X_extra - residual['mean']) / residual['scale']
    logits = Xz @ residual['coef'] + residual['intercept']
    return np.asarray(base_pred, dtype=np.float32) + residual['weight'] * logits.astype(np.float32)
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
