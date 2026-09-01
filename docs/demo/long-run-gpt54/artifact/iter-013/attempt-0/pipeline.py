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
    config = dict(OP_CONFIG)
    config['hp'] = dict(HP)
    return SO.train(splits, train_idx, Xs, dim, y, seed, config, verbose)
# <<<END:train>>>


# <<<BLOCK:predict>>>
def predict(model, Xs, split):
    raw = np.asarray(SO.predict(model, Xs, split), dtype=np.float32)
    rows = DV.load(split)
    tabs = np.asarray(rows.tab)
    assert raw.shape[0] == tabs.shape[0], 'tab/score length mismatch'

    pred = raw.copy()
    k1, k2 = -1.0, 1.0

    def apply_pwl(x, slope_lo, slope_mid, slope_hi, bias):
        y = np.empty_like(x, dtype=np.float32)
        lo = x < k1
        hi = x > k2
        mid = ~(lo | hi)
        y[mid] = slope_mid * x[mid] + bias
        y[lo] = slope_mid * k1 + bias + slope_lo * (x[lo] - k1)
        y[hi] = slope_mid * k2 + bias + slope_hi * (x[hi] - k2)
        return y

    # Fixed tab-conditional monotone calibrations, kept close to identity.
    # Buckets follow the accepted coarse grouping: {0,1,2,4,6,other}.
    params = {
        0: (0.96, 1.05, 1.12, 0.055),
        1: (0.98, 1.02, 1.06, 0.020),
        2: (0.95, 1.06, 1.14, 0.070),
        4: (1.00, 0.99, 0.97, -0.015),
        6: (0.97, 1.03, 1.08, 0.035),
    }

    seen = np.zeros(tabs.shape[0], dtype=bool)
    for tab_value, p in params.items():
        m = tabs == tab_value
        if np.any(m):
            pred[m] = apply_pwl(raw[m], *p)
            seen |= m

    other = ~seen
    if np.any(other):
        pred[other] = apply_pwl(raw[other], 1.0, 1.0, 1.0, 0.0)

    return pred
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
