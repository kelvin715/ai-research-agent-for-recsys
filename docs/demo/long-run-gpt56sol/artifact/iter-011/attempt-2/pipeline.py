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
    target = SO.build_target(splits, train_idx, OP_CONFIG)
    if not isinstance(target, dict) or 'main' not in target or 'engagement_aux' not in target:
        raise AssertionError('pointwise_engagement_mtl target contract is missing required entries')

    def load_single_feedback(name):
        raw = DV.train_targets([name])
        if isinstance(raw, dict):
            if name not in raw:
                raise AssertionError('missing train-only feedback target: %s' % name)
            value = raw[name]
        elif hasattr(raw, 'columns') and name in raw.columns:
            value = raw[name]
        elif isinstance(raw, (list, tuple)) and len(raw) == 1:
            value = raw[0]
        else:
            value = raw

        arr = np.asarray(value)
        if arr.dtype.names is not None:
            if name not in arr.dtype.names:
                raise AssertionError('missing structured feedback field: %s' % name)
            arr = np.asarray(arr[name])
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        elif arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise AssertionError('%s must resolve to one train-only feedback column' % name)
        if not np.isfinite(arr.astype(np.float64, copy=False)).all():
            raise AssertionError('%s contains non-finite values' % name)
        return arr

    hate = load_single_feedback('is_hate')
    profile_enter = load_single_feedback('is_profile_enter')
    DV.assert_trainable(hate, 'build_target.is_hate')
    DV.assert_trainable(profile_enter, 'build_target.is_profile_enter')

    n_train = int(splits['train'].n)
    if hate.shape[0] != n_train or profile_enter.shape[0] != n_train:
        raise AssertionError('feedback target length does not match the train view')

    aux = np.asarray(target['engagement_aux'])
    if aux.ndim != 2:
        raise AssertionError('engagement_aux must be a two-dimensional array')

    idx = np.asarray(train_idx, dtype=np.int64)
    if aux.shape[0] == n_train:
        aux_check = aux
        hate_check = hate
        replacement = profile_enter
    elif aux.shape[0] == idx.shape[0]:
        aux_check = aux
        hate_check = hate[idx]
        replacement = profile_enter[idx]
    else:
        raise AssertionError('engagement_aux row count is incompatible with train_idx')

    hate_binary = np.asarray(hate_check, dtype=np.float64) > 0.5
    direct_matches = []
    complemented_matches = []
    for column in range(aux.shape[1]):
        candidate = np.asarray(aux_check[:, column], dtype=np.float64)
        if not np.isfinite(candidate).all():
            continue
        candidate_binary = candidate > 0.5
        if np.array_equal(candidate_binary, hate_binary):
            direct_matches.append(column)
        if np.array_equal(candidate_binary, np.logical_not(hate_binary)):
            complemented_matches.append(column)

    matches = sorted(set(direct_matches + complemented_matches))
    if len(matches) != 1:
        raise AssertionError(
            'expected exactly one engagement_aux column encoding is_hate or its complement; found %d'
            % len(matches))

    aux_new = aux.copy()
    column = matches[0]
    replacement_binary = (np.asarray(replacement, dtype=np.float64) > 0.5)
    aux_new[:, column] = replacement_binary.astype(aux_new.dtype, copy=False)
    DV.assert_trainable(aux_new[:, column],
                        'build_target.engagement_aux.is_profile_enter')

    result = dict(target)
    result['engagement_aux'] = aux_new
    return result
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
