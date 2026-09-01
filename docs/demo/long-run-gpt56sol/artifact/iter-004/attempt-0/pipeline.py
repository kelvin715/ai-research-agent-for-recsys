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
    long_view = np.asarray(
        SO.build_target(splits, train_idx, OP_CONFIG)
    ).reshape(-1)
    train_idx_arr = np.asarray(train_idx, dtype=np.int64).reshape(-1)

    if long_view.shape[0] != train_idx_arr.shape[0]:
        raise ValueError(
            'SO.build_target returned %d rows for %d train indices'
            % (long_view.shape[0], train_idx_arr.shape[0])
        )

    click_raw = DV.train_targets(['is_click'])
    if isinstance(click_raw, dict):
        click_raw = click_raw['is_click']
    click_all = np.asarray(click_raw).reshape(-1)
    if train_idx_arr.size and (
        train_idx_arr.min() < 0 or train_idx_arr.max() >= click_all.shape[0]
    ):
        raise IndexError('train_idx is outside the train-only is_click array')
    is_click = click_all[train_idx_arr]

    DV.assert_trainable(long_view, 'ordinal_reward.long_view')
    DV.assert_trainable(is_click, 'ordinal_reward.is_click')

    if not np.all((long_view == 0) | (long_view == 1)):
        raise ValueError('long_view must be binary')
    if not np.all((is_click == 0) | (is_click == 1)):
        raise ValueError('is_click must be binary')

    long_view = long_view.astype(np.int32, copy=False)
    is_click = is_click.astype(np.int32, copy=False)
    grade = 2 * long_view + is_click
    if not np.all((grade >= 0) & (grade <= 3)):
        raise AssertionError('ordinal relevance must be in {0,1,2,3}')
    return grade.astype(np.int32, copy=False)
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
    grade = np.asarray(y, dtype=np.int32).reshape(-1)
    if grade.shape[0] != len(train_idx):
        raise ValueError('graded target length does not match train_idx')
    if not np.all((grade >= 0) & (grade <= 3)):
        raise ValueError('graded target must contain only {0,1,2,3}')

    config = dict(OP_CONFIG)
    config['hp'] = dict(HP)
    model, info = SO.train(
        splits, train_idx, Xs, dim, grade, seed, config, verbose
    )

    counts = np.bincount(grade, minlength=4)
    info['target_kind'] = '2*long_view+is_click'
    info['target_grade_counts'] = {
        str(i): int(counts[i]) for i in range(4)
    }
    return model, info
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
