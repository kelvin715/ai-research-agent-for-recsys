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
OP_CONFIG = {"dur_buckets": 10, "features": [], "hp": {"aux_weight": 0.2, "batch": 8192, "epochs": 12, "hidden": [128, 64], "k": 16, "l2": 0.0001, "lr": 0.001, "n_neg": 1, "torch_threads": 1}, "model_family": "torch_deepfm_mtl", "objective": "pointwise_engagement_mtl"}


def build_data_view():
    return SO.build_data_view()
# <<<END:data_view>>>


# <<<BLOCK:features>>>
def build_features(splits, train_idx):
    return SO.build_features(splits, train_idx, OP_CONFIG)
# <<<END:features>>>


# <<<BLOCK:target>>>
def build_target(splits, train_idx):
    parent_target = SO.build_target(splits, train_idx, OP_CONFIG)
    main = np.asarray(parent_target['main'])

    train_idx = np.asarray(train_idx, dtype=np.int64)
    n_full = int(splits['train'].n)
    n_fit = int(train_idx.shape[0])
    assert main.shape[0] == n_fit
    assert np.all((train_idx >= 0) & (train_idx < n_full))

    def load_target_column(name):
        values = np.asarray(DV.train_targets([name]))
        assert values.size == n_full, (
            name, values.shape, n_full
        )
        values = values.reshape(-1)
        DV.assert_trainable(values, 'build_target.' + name + '_full')
        return values

    click_full = load_target_column('is_click')
    like_full = load_target_column('is_like')
    profile_full = load_target_column('is_profile_enter')

    watch_ratio_full, watch_valid_full = DV.watch_ratio()
    watch_ratio_full = np.asarray(watch_ratio_full)
    watch_valid_full = np.asarray(watch_valid_full, dtype=bool)
    assert watch_ratio_full.size == n_full
    assert watch_valid_full.size == n_full
    watch_ratio_full = watch_ratio_full.reshape(-1)
    watch_valid_full = watch_valid_full.reshape(-1)

    click = click_full[train_idx]
    like = like_full[train_idx]
    profile = profile_full[train_idx]
    watch_ratio = watch_ratio_full[train_idx]
    watch_valid = watch_valid_full[train_idx]

    assert click.shape == (n_fit,)
    assert like.shape == (n_fit,)
    assert profile.shape == (n_fit,)
    assert watch_ratio.shape == (n_fit,)
    assert watch_valid.shape == (n_fit,)

    DV.assert_trainable(main, 'build_target.main')
    DV.assert_trainable(click, 'build_target.is_click')
    DV.assert_trainable(like, 'build_target.is_like')
    DV.assert_trainable(profile, 'build_target.is_profile_enter')
    DV.assert_trainable(watch_ratio, 'build_target.watch_ratio')

    soft_completion = np.full(n_fit, 0.5, dtype=np.float32)
    if np.any(watch_valid):
        soft_completion[watch_valid] = np.clip(
            watch_ratio[watch_valid], 0.0, 1.0
        ).astype(np.float32, copy=False)

    engagement_aux = np.column_stack((
        click,
        like,
        profile,
        soft_completion,
    )).astype(np.float32, copy=False)

    assert engagement_aux.shape == (n_fit, 4)
    assert np.isfinite(engagement_aux).all()
    assert np.all((soft_completion >= 0.0) & (soft_completion <= 1.0))
    assert np.all(soft_completion[~watch_valid] == 0.5)

    DV.assert_trainable(soft_completion, 'build_target.soft_completion')
    DV.assert_trainable(engagement_aux, 'build_target.engagement_aux')

    return {'main': main, 'engagement_aux': engagement_aux}
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
