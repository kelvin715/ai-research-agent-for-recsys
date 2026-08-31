"""开发期可信评测 —— 沙箱外唯一读取 official-valid 标签和官方 evaluate.py 的地方。

候选进程只输出 pred.npy；分数由这里算，写到候选**不可达**的路径。
所以「候选自行决定指标 / 篡改 metric」（计划 §11）在结构上不可能发生。

两条不变量：
  1. 聚合分数必须与官方 evaluate() 逐位一致 —— 本模块自己算 per-user 明细是为了
     bootstrap，但每次都断言它重新聚合后 == evaluate() 的输出。口径不允许有第二种实现。
  2. 本模块在结构上只支持 official-valid。test 预测只用于生成提交，由组委会评分；
     本地 test scorer 不属于开发代码，也没有可绕过的命令行开关。

bootstrap 的做法：per-user 统计只算一次，重采样用户即可精确重算两个聚合量 ——
不需要重跑 evaluate，1000 次重采样是毫秒级。
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, '..', 'kuairand-starter-kit'))

import columns as C
from evaluate import evaluate, auc, ndcg_at_k       # 官方口径，冻结不改

TRUSTED_CACHE = os.path.join(ROOT, 'trusted_cache_dev')   # 仅 valid；绝不挂进沙箱
K = 5
DEV_SPLITS = ('valid',)


# --------------------------------------------------------------------------
# 真标签：从原始 CSV 建缓存。这份数据不在 views/agent 里，沙箱够不着。
# --------------------------------------------------------------------------
def build_truth_cache(src_dir, out_dir=TRUSTED_CACHE):
    """只构建 official-valid 缓存；循环在访问标签前排除 train/test 行。"""
    import csv
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = C.SPLITS['valid']
    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv',
              'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(src_dir, f), newline='') as fh:
            for r in csv.DictReader(fh):
                date = int(r['date'])
                if lo <= date <= hi:
                    rows.append((int(r['user_id']), 1 if r[C.LABEL] != '0' else 0))
    np.save(os.path.join(out_dir, 'valid.user_id.npy'),
            np.array([x[0] for x in rows], dtype=np.int32))
    np.save(os.path.join(out_dir, 'valid.label.npy'),
            np.array([x[1] for x in rows], dtype=np.int64))
    os.chmod(out_dir, 0o700)
    os.chmod(os.path.join(out_dir, 'valid.user_id.npy'), 0o400)
    os.chmod(os.path.join(out_dir, 'valid.label.npy'), 0o400)
    return out_dir


def _truth(split, cache=TRUSTED_CACHE):
    if split not in DEV_SPLITS:
        raise PermissionError(
            f'开发 evaluator 只允许 {DEV_SPLITS}，收到 {split!r}；test 只能由组委会评分。')
    u = os.path.join(cache, f'{split}.user_id.npy')
    if not os.path.exists(u):
        raise FileNotFoundError(
            f'真标签缓存不存在，先跑: python3 trusted/evaluator.py --build-cache')
    return (np.load(u), np.load(os.path.join(cache, f'{split}.label.npy')))


# --------------------------------------------------------------------------
# per-user 明细 + 聚合（聚合结果对官方 evaluate() 做一致性断言）
# --------------------------------------------------------------------------
def per_user_stats(user_ids, labels, scores, k=K):
    """返回 (uids, npos, auc_u, ndcg_u)，与官方 evaluate 用同一组函数。"""
    import collections
    byu = collections.defaultdict(list)
    for u, y, s in zip(user_ids, labels, scores):
        byu[u].append((s, y))
    uids, npos, aucs, ndcgs = [], [], [], []
    for u, lst in byu.items():
        lst.sort(key=lambda x: -x[0])
        labs = [y for _, y in lst]
        p = sum(labs)
        uids.append(u)
        npos.append(p)
        # GAUC 只统计 0 < 正例数 < 曝光数 的用户；其余记 nan 表示不参与
        aucs.append(auc(labs, [s for s, _ in lst]) if 0 < p < len(labs) else np.nan)
        ndcgs.append(ndcg_at_k(labs, k))
    return (np.array(uids), np.array(npos, dtype=np.float64),
            np.array(aucs, dtype=np.float64), np.array(ndcgs, dtype=np.float64))


def aggregate(npos, aucs, ndcgs):
    """由 per-user 明细重算 GAUC / nDCG / primary，口径与 evaluate() 相同。"""
    m = ~np.isnan(aucs)
    gden = npos[m].sum()
    gauc = float((npos[m] * aucs[m]).sum() / gden) if gden else 0.5
    ndcg = float(ndcgs.mean()) if len(ndcgs) else 0.0
    return {'GAUC': gauc, f'nDCG@{K}': ndcg, 'primary': (gauc + ndcg) / 2.0}


def bootstrap_ci(npos, aucs, ndcgs, n=1000, seed=0, alpha=0.05):
    """用户级 bootstrap。表征的是 validation 用户抽样的不确定性 ——
    与 seed 方差（训练随机性）是两回事，报告里不得合称 '3 sigma'（计划 §7.1）。

    ⚠️ 这是**边际** CI，只能用来描述「这个模型的分数有多不确定」。
    **不能**用它当接受阈值 —— 比较两个模型时用 paired_bootstrap_delta()。
    实测量级：本题 seed_sd ≈ 0.00004，边际 bootstrap_sd ≈ 0.00215，差 54 倍。
    """
    if n <= 0:
        # n_boot=0 = 显式跳过（例如 block 消融只要点估计，不需要区间）
        return [float('nan'), float('nan')], float('nan')
    rng = np.random.default_rng(seed)
    n_u = len(npos)
    out = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_u, n_u)
        out[i] = aggregate(npos[idx], aucs[idx], ndcgs[idx])['primary']
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)], float(out.std())


def paired_bootstrap_delta(stats_a, stats_b, n=1000, seed=0, alpha=0.05):
    """配对用户级 bootstrap：重采样同一组用户，同时算 A 与 B 的分差。

    为什么必须配对：A 和 B 在同一批用户上评测，用户抽样带来的波动对两者是**共同的**，
    做差时会抵消。用边际 CI（±0.0043）当接受门槛会把几乎所有真实改进都判成噪声；
    用配对 delta 的 CI 才是正确的尺子。

    stats_* 是 per_user_stats() 的返回值。两者的 uid 顺序必须一致 —— 由调用方保证
    （同一 split、同一行序 ⇒ 首次出现顺序相同），这里会断言。
    """
    uid_a, npos_a, auc_a, ndcg_a = stats_a
    uid_b, npos_b, auc_b, ndcg_b = stats_b
    assert np.array_equal(uid_a, uid_b), 'per-user 统计的用户顺序不一致，无法配对'
    assert np.array_equal(npos_a, npos_b), '同一 split 的每用户正例数应相同'

    point = (aggregate(npos_a, auc_a, ndcg_a)['primary']
             - aggregate(npos_b, auc_b, ndcg_b)['primary'])
    if n <= 0:
        return {'delta_primary': float(point),
                'paired_ci95': [float('nan'), float('nan')],
                'paired_sd': float('nan'), 'excludes_zero': False}

    rng = np.random.default_rng(seed)
    n_u = len(uid_a)
    out = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_u, n_u)
        out[i] = (aggregate(npos_a[idx], auc_a[idx], ndcg_a[idx])['primary']
                  - aggregate(npos_b[idx], auc_b[idx], ndcg_b[idx])['primary'])
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return {'delta_primary': float(point),
            'paired_ci95': [float(lo), float(hi)],
            'paired_sd': float(out.std()),
            'excludes_zero': bool(lo > 0 or hi < 0)}


def bootstrap_seed_mean(stats, n=1000, seed=0, alpha=0.05):
    """对同一批用户重采样，并在每次重采样内对多个训练 seed 的指标取均值。"""
    if not stats:
        raise ValueError('至少需要一个 seed')
    uid0, npos0, _, _ = stats[0]
    for uids, npos, _, _ in stats[1:]:
        if not np.array_equal(uid0, uids) or not np.array_equal(npos0, npos):
            raise ValueError('多个 seed 的用户或标签不一致')
    if n <= 0:
        return [float('nan'), float('nan')], float('nan')
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(uid0), len(uid0))
        out[i] = np.mean([aggregate(s[1][idx], s[2][idx], s[3][idx])['primary']
                          for s in stats])
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)], float(out.std())


def paired_bootstrap_seed_mean(stats_a, stats_b, n=1000, seed=0, alpha=0.05):
    """多个匹配 seed 的用户级配对 bootstrap，返回 seed-mean primary 差。"""
    if len(stats_a) != len(stats_b) or not stats_a:
        raise ValueError('candidate/incumbent 必须提供相同且非零的 seed 数')
    uid0, npos0, _, _ = stats_a[0]
    for sa, sb in zip(stats_a, stats_b):
        if (not np.array_equal(uid0, sa[0]) or not np.array_equal(uid0, sb[0])
                or not np.array_equal(npos0, sa[1]) or not np.array_equal(npos0, sb[1])):
            raise ValueError('candidate/incumbent 的用户或标签不一致')
    point = float(np.mean([aggregate(s[1], s[2], s[3])['primary'] for s in stats_a])
                  - np.mean([aggregate(s[1], s[2], s[3])['primary'] for s in stats_b]))
    if n <= 0:
        return {'delta_primary': point,
                'paired_ci95': [float('nan'), float('nan')],
                'paired_sd': float('nan'), 'excludes_zero': False}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(uid0), len(uid0))
        da = [aggregate(s[1][idx], s[2][idx], s[3][idx])['primary'] for s in stats_a]
        db = [aggregate(s[1][idx], s[2][idx], s[3][idx])['primary'] for s in stats_b]
        out[i] = float(np.mean(da) - np.mean(db))
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return {'delta_primary': point, 'paired_ci95': [float(lo), float(hi)],
            'paired_sd': float(out.std()), 'excludes_zero': bool(lo > 0 or hi < 0)}


def _validated_pred(pred, labels, name='预测'):
    arr = np.load(pred, allow_pickle=False) if isinstance(pred, str) else np.asarray(pred)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != labels.shape:
        raise ValueError(f'{name} {arr.shape} 行，应为 {labels.shape} 行')
    if not np.isfinite(arr).all():
        raise ValueError(f'{name}含 NaN/Inf（{int((~np.isfinite(arr)).sum())} 个）')
    return arr


def compare(pred_a, pred_b, split, n=1000, seed=0, cache=TRUSTED_CACHE):
    """候选 A 相对 incumbent B 的配对比较。这是 ACCEPT/ROLLBACK 的正确依据。"""
    uids, labels = _truth(split, cache)
    u_list, y_list = uids.tolist(), labels.tolist()
    a = _validated_pred(pred_a, labels, '候选预测')
    b = _validated_pred(pred_b, labels, 'incumbent 预测')
    sa = per_user_stats(u_list, y_list, a)
    sb = per_user_stats(u_list, y_list, b)
    return paired_bootstrap_delta(sa, sb, n=n, seed=seed)


# --------------------------------------------------------------------------
# 对外入口
# --------------------------------------------------------------------------
def score(pred, split, n_boot=1000, boot_seed=0,
          cache=TRUSTED_CACHE, verify_against_official=True):
    """给一份预测打分。pred 可以是 ndarray 或 .npy 路径。"""
    uids, labels = _truth(split, cache)
    pred = _validated_pred(pred, labels)

    u_list, y_list = uids.tolist(), labels.tolist()
    upu, npos, aucs, ndcgs = per_user_stats(u_list, y_list, pred)
    res = aggregate(npos, aucs, ndcgs)

    if verify_against_official:
        # 不变量 1：本模块的聚合必须与官方 evaluate() 逐位一致
        off = evaluate(u_list, y_list, pred)
        for key in ('GAUC', f'nDCG@{K}', 'primary'):
            assert abs(off[key] - res[key]) < 1e-12, \
                f'口径不一致 {key}: 官方 {off[key]!r} vs 本模块 {res[key]!r}'
        res['users'], res['rows'] = off['users'], off['rows']

    ci, boot_sd = bootstrap_ci(npos, aucs, ndcgs, n=n_boot, seed=boot_seed)
    res['bootstrap_ci95_primary'] = ci
    res['bootstrap_sd_primary'] = boot_sd
    res['n_users_in_gauc'] = int((~np.isnan(aucs)).sum())
    res['split'] = split
    res['pred_sha256'] = hashlib.sha256(
        np.asarray(pred, dtype=np.float32).tobytes()).hexdigest()[:16]
    return res


def score_seeds(preds, split, incumbent_preds=None, **kw):
    """多种子评测。返回 per-seed、均值、seed 方差，以及与 incumbent 的配对 delta。

    配对 delta 用同一组 seed 逐个相减，比比较两个均值更敏感（计划 §7.1）。
    """
    rs = [score(p, split, **kw) for p in preds]
    prim = np.array([r['primary'] for r in rs])
    cache = kw.get('cache', TRUSTED_CACHE)
    n_boot = kw.get('n_boot', 1000)
    boot_seed = kw.get('boot_seed', 0)
    uids, labels = _truth(split, cache)
    u_list, y_list = uids.tolist(), labels.tolist()
    stats = [per_user_stats(u_list, y_list, _validated_pred(p, labels)) for p in preds]
    mean_ci, mean_boot_sd = bootstrap_seed_mean(
        stats, n=n_boot, seed=boot_seed)
    out = {
        'per_seed': [{'primary': r['primary'], 'GAUC': r['GAUC'],
                      f'nDCG@{K}': r[f'nDCG@{K}'], 'pred_sha256': r['pred_sha256']}
                     for r in rs],
        'primary_mean': float(prim.mean()),
        'primary_seed_sd': float(prim.std(ddof=1)) if len(prim) > 1 else 0.0,
        'GAUC_mean': float(np.mean([r['GAUC'] for r in rs])),
        f'nDCG@{K}_mean': float(np.mean([r[f'nDCG@{K}'] for r in rs])),
        'bootstrap_ci95_primary_mean': mean_ci,
        'bootstrap_sd_primary_mean': mean_boot_sd,
        'n_seeds': len(rs),
        'split': split,
    }
    if incumbent_preds is not None:
        if len(incumbent_preds) != len(preds):
            raise ValueError('incumbent_preds 必须与 preds 的 seed 数相同')
        inc_stats = [per_user_stats(u_list, y_list, _validated_pred(p, labels))
                     for p in incumbent_preds]
        out['paired_vs_incumbent'] = paired_bootstrap_seed_mean(
            stats, inc_stats, n=n_boot, seed=boot_seed)
    return out


def directional_compare(pred_a, pred_b, split, feature_dir, min_tab_rows=1000,
                        cache=TRUSTED_CACHE):
    """Return fixed aggregate slices for one already-scored A/B prediction pair.

    This is diagnostic feedback, not an alternative validation metric.  It reuses the
    same two prediction vectors and the same official-valid labels; it never trains or
    selects another candidate.  Only aggregate deltas leave the trusted evaluator.
    """
    if split != 'valid':
        raise PermissionError('directional feedback is development-only official-valid output')
    uids, labels = _truth(split, cache)
    a = _validated_pred(pred_a, labels, 'candidate ensemble prediction')
    b = _validated_pred(pred_b, labels, 'incumbent ensemble prediction')

    def feature(name):
        arr = np.load(os.path.join(feature_dir, name + '.npy'), mmap_mode='r')
        if len(arr) != len(labels) and name.startswith('valid.'):
            raise ValueError(f'{name} row count differs from official-valid labels')
        return np.asarray(arr)

    valid_users = feature('valid.user_id')
    if not np.array_equal(valid_users, uids):
        raise ValueError('feature-only valid user order differs from trusted truth order')
    valid_videos = feature('valid.video_id')
    valid_tabs = feature('valid.tab')
    train_users = feature('train.user_id')
    train_videos = feature('train.video_id')

    def delta_for_mask(mask):
        mask = np.asarray(mask, dtype=bool)
        sa = per_user_stats(uids[mask].tolist(), labels[mask].tolist(), a[mask])
        sb = per_user_stats(uids[mask].tolist(), labels[mask].tolist(), b[mask])
        ma = aggregate(sa[1], sa[2], sa[3])
        mb = aggregate(sb[1], sb[2], sb[3])
        return {
            'rows': int(mask.sum()),
            'users': int(len(sa[0])),
            'delta_primary': float(ma['primary'] - mb['primary']),
        }

    def count_lookup(values):
        keys, counts = np.unique(values, return_counts=True)
        return {int(key): int(count) for key, count in zip(keys, counts)}

    def quartile_slices(row_values):
        cuts = np.quantile(row_values, [0.25, 0.5, 0.75]).astype(np.int64)
        bounds = [(None, int(cuts[0])),
                  (int(cuts[0]) + 1, int(cuts[1])),
                  (int(cuts[1]) + 1, int(cuts[2])),
                  (int(cuts[2]) + 1, None)]
        slices = []
        for lo, hi in bounds:
            mask = np.ones(len(row_values), dtype=bool)
            if lo is not None:
                mask &= row_values >= lo
            if hi is not None:
                mask &= row_values <= hi
            item = {'min_inclusive': lo, 'max_inclusive': hi, **delta_for_mask(mask)}
            slices.append(item)
        return {'cutpoints': cuts.tolist(), 'slices': slices}

    # Assign whole users to history strata so each slice retains the official within-user task.
    history_lookup = count_lookup(train_users)
    history = np.array([history_lookup.get(int(uid), 0) for uid in uids], dtype=np.int64)
    unique_users, first = np.unique(uids, return_index=True)
    user_history = history[first]
    history_cuts = np.quantile(user_history, [0.25, 0.5, 0.75]).astype(np.int64)
    history_bounds = [(None, int(history_cuts[0])),
                      (int(history_cuts[0]) + 1, int(history_cuts[1])),
                      (int(history_cuts[1]) + 1, int(history_cuts[2])),
                      (int(history_cuts[2]) + 1, None)]
    history_slices = []
    for lo, hi in history_bounds:
        mask = np.ones(len(history), dtype=bool)
        if lo is not None:
            mask &= history >= lo
        if hi is not None:
            mask &= history <= hi
        history_slices.append(
            {'min_inclusive': lo, 'max_inclusive': hi, **delta_for_mask(mask)})

    popularity_lookup = count_lookup(train_videos)
    popularity = np.array(
        [popularity_lookup.get(int(video), 0) for video in valid_videos], dtype=np.int64)
    tabs = []
    for tab in np.unique(valid_tabs):
        mask = valid_tabs == tab
        if int(mask.sum()) >= min_tab_rows:
            tabs.append({'tab': int(tab), **delta_for_mask(mask)})

    return {
        'status': 'diagnostic_only_not_official_metric',
        'candidate_vs_incumbent': True,
        'fixed_dimensions': ['train_user_history_quartile',
                             'train_item_popularity_quartile', 'tab'],
        'train_user_history_quartile': {
            'cutpoints': history_cuts.tolist(), 'slices': history_slices},
        'train_item_popularity_quartile': quartile_slices(popularity),
        'tab': tabs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pred', nargs='*', help='一个或多个 pred.npy（多个 = 多种子）')
    ap.add_argument('--split', default='valid', choices=DEV_SPLITS)
    ap.add_argument('--out', help='metric JSON 落盘路径（必须在沙箱外）')
    ap.add_argument('--build-cache', action='store_true')
    ap.add_argument('--src', default=os.path.join(ROOT, '..', 'KuaiRand-Pure', 'data'))
    ap.add_argument('--n-boot', type=int, default=1000)
    a = ap.parse_args()

    if a.build_cache:
        d = build_truth_cache(a.src)
        print(f'✓ 真标签缓存已建于 {d}（0700，绝不挂进沙箱）')
        for s in DEV_SPLITS:
            u, y = _truth(s)
            print(f'  {s:5s} {len(y):>9,d} 行  正例率 {y.mean():.4f}')
        return

    if not a.pred:
        ap.error('需要至少一个 pred.npy')

    if len(a.pred) == 1:
        res = score(a.pred[0], a.split, n_boot=a.n_boot)
    else:
        res = score_seeds(a.pred, a.split, n_boot=a.n_boot)
    txt = json.dumps(res, indent=2, ensure_ascii=False)
    print(txt)
    if a.out:
        with open(a.out, 'w') as fh:
            fh.write(txt)


if __name__ == '__main__':
    main()
