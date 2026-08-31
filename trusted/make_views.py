"""构建挂进沙箱的脱敏数据视图。一次性运行，产物受 manifest hash 保护。

这是 hidden-test 物理隔离的**安全边界**（计划 §2.2）：test 真标签不进 views/agent/，
所以候选进程无论做什么都取不到 —— 不依赖 prompt 约束，也不依赖源码扫描。

  views/agent/    挂进沙箱。official-valid/test 段的 11 个反馈列全部置空串，且不落
                  label.npy；候选只拿到 train 标签。
  KuaiRand-Pure/  原始数据，只有沙箱外的 trusted/evaluator.py 读它（不额外复制一份，
                  避免两份真标签互相漂移；完整性由 manifest 的 source hash 保证）。

隔离的两个搭档文件也在这里被排除：
  video_features_statistic_pure.csv  统计窗口未知，有时间泄漏嫌疑 → 不挂载
  log_random_4_22_to_5_08_pure.csv   日期落在评测窗口 → 不挂载

用法:
    python3 trusted/make_views.py --src ../KuaiRand-Pure/data --out views/agent
"""
import argparse
import csv
import datetime as datetime_module
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import columns as C

# 原样复制的文件（不含曝光后信号）
COPY_AS_IS = [
    'video_features_basic_pure.csv',
    'user_features_pure.csv',
]
# 需要按行脱敏的日志文件
LOG_FILES = [
    'log_standard_4_08_to_4_21_pure.csv',
    'log_standard_4_22_to_5_08_pure.csv',
]

TRAIN_LO, TRAIN_HI = C.SPLITS['train']


def _safe_float(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _categorical_codes(rows, fields):
    """Encode a label-free static table into compact deterministic ids."""
    encoded = np.zeros((len(rows), len(fields)), dtype=np.int32)
    for j, field in enumerate(fields):
        values = [(row.get(field) or '__missing__') for row in rows]
        vocab = {value: index + 1 for index, value in enumerate(sorted(set(values)))}
        encoded[:, j] = [vocab[value] for value in values]
    return encoded


def _static_lookups(dst_dir):
    """Build dense user/video lookup matrices without interaction feedback."""
    with open(os.path.join(dst_dir, 'user_features_pure.csv'), newline='') as fh:
        user_rows = list(csv.DictReader(fh))
    with open(os.path.join(dst_dir, 'video_features_basic_pure.csv'), newline='') as fh:
        video_rows = list(csv.DictReader(fh))

    def dense_lookup(rows, key, values, width, dtype):
        max_id = max(int(row[key]) for row in rows)
        lookup = np.zeros((max_id + 2, width), dtype=dtype)
        for row, value in zip(rows, values):
            lookup[int(row[key])] = value
        return lookup

    user_cat_values = _categorical_codes(user_rows, C.USER_CATEGORICAL_FEATURE)
    user_num_values = np.asarray([
        [_safe_float(row.get(field)) for field in C.USER_NUMERIC_FEATURE]
        for row in user_rows
    ], dtype=np.float32)
    video_cat_values = _categorical_codes(video_rows, C.VIDEO_CATEGORICAL_FEATURE)
    video_num_values = []
    epoch = datetime_module.date(2022, 1, 1)
    for row in video_rows:
        try:
            upload_day = (datetime_module.date.fromisoformat(row['upload_dt']) - epoch).days
        except (KeyError, TypeError, ValueError):
            upload_day = 0
        video_num_values.append([
            _safe_float(row.get('video_duration')),
            _safe_float(row.get('server_width')),
            _safe_float(row.get('server_height')),
            float(upload_day),
        ])
    video_num_values = np.asarray(video_num_values, dtype=np.float32)

    return {
        'user_categorical': dense_lookup(
            user_rows, 'user_id', user_cat_values,
            len(C.USER_CATEGORICAL_FEATURE), np.int32),
        'user_numeric': dense_lookup(
            user_rows, 'user_id', user_num_values,
            len(C.USER_NUMERIC_FEATURE), np.float32),
        'video_categorical': dense_lookup(
            video_rows, 'video_id', video_cat_values,
            len(C.VIDEO_CATEGORICAL_FEATURE), np.int32),
        'video_numeric': dense_lookup(
            video_rows, 'video_id', video_num_values,
            len(C.VIDEO_NUMERIC_FEATURE), np.float32),
    }


def sanitize_log(src_path, dst_path):
    """把 train 之外行的 TRAIN_ONLY_TARGET 列置空串，其余原样。

    返回 (总行数, 被脱敏行数)。列顺序与行顺序严格保持不变 —— row_id 依赖它。
    """
    n_total = n_blanked = 0
    with open(src_path, newline='') as fin, open(dst_path, 'w', newline='') as fout:
        reader = csv.DictReader(fin)
        fields = reader.fieldnames
        assert fields is not None, f"{src_path} 没有表头"
        missing = [c for c in C.TRAIN_ONLY_TARGET if c not in fields]
        assert not missing, f"{src_path} 缺少预期的反馈列: {missing}"

        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            n_total += 1
            date = int(row['date'])
            if not (TRAIN_LO <= date <= TRAIN_HI):
                for col in C.TRAIN_ONLY_TARGET:
                    row[col] = ''
                n_blanked += 1
            writer.writerow(row)
    return n_total, n_blanked


def verify(dst_dir):
    """回读视图，断言 official-valid/test 段没有任何反馈值残留。

    这是最后一道自检 —— 如果它过不了，后面所有隔离承诺都是空的。
    """
    leaked = 0
    checked = 0
    for name in LOG_FILES:
        path = os.path.join(dst_dir, name)
        with open(path, newline='') as fh:
            for row in csv.DictReader(fh):
                date = int(row['date'])
                if TRAIN_LO <= date <= TRAIN_HI:
                    continue
                checked += 1
                for col in C.TRAIN_ONLY_TARGET:
                    if row[col] != '':
                        leaked += 1
    assert leaked == 0, f"脱敏失败：valid/test 段仍有 {leaked} 个反馈值"
    # 隔离文件不得出现在视图里
    for bad in C.QUARANTINED_FILES + C.FORBIDDEN_TRAIN_FILES:
        assert not os.path.exists(os.path.join(dst_dir, bad)), f"{bad} 不应出现在 agent 视图里"
    return checked


def parse_to_arrays(dst_dir):
    """把脱敏后的 CSV 预解析成 .npy，按 data.load() 的规范行序落盘。

    动机：csv 模块解析 143 万行约 11s；50 轮 x 3 seed = 150 次沙箱运行就是 27 分钟纯解析。
    预解析后降到 ~0.1s（np.load + mmap）。

    行序规范（必须与 kit 的 data.load() 逐行一致，row_id 依赖它）：
    先读 log_standard_4_08_to_4_21，再读 4_22_to_5_08，拼接后按 date 过滤，保持原文件顺序。
    这里用「拼接后过滤」的通用写法而不是假设 train 全在 file1，以保证与 data.py 同构。

    official-valid/test 标签不落盘；不是写成 0 或哨兵，以免候选误把占位值当真值。
    """
    vid2author = {}
    with open(os.path.join(dst_dir, 'video_features_basic_pure.csv'), newline='') as fh:
        for r in csv.DictReader(fh):
            vid2author[int(r['video_id'])] = int(r['author_id'])
    static = _static_lookups(dst_dir)

    def blank_to(v, default):
        return default if v == '' else int(v)

    rows = []          # (date, user_id, video_id, author_id, tab, duration_ms, hourmin, time_ms, label)
    targets = []       # 与 rows 平行：TRAIN_ONLY_TARGET 原始值（valid/test 为 WITHHELD）
    for name in LOG_FILES:
        with open(os.path.join(dst_dir, name), newline='') as fh:
            for r in csv.DictReader(fh):
                vid = int(r['video_id'])
                rows.append((
                    int(r['date']), int(r['user_id']), vid,
                    vid2author.get(vid, -1), int(r['tab']),
                    float(r['duration_ms']), int(r['hourmin']), int(r['time_ms']),
                    blank_to(r[C.LABEL], C.WITHHELD),
                ))
                targets.append([blank_to(r[c], C.WITHHELD) for c in C.TRAIN_ONLY_TARGET])

    date_all = np.array([x[0] for x in rows], dtype=np.int32)
    out_dir = os.path.join(dst_dir, 'parsed')
    os.makedirs(out_dir, exist_ok=True)

    spec = [('user_id', 1, np.int32), ('video_id', 2, np.int32), ('author_id', 3, np.int32),
            ('tab', 4, np.int16), ('duration_ms', 5, np.float32), ('hourmin', 6, np.int16),
            ('time_ms', 7, np.int64)]

    # Regeneration must also remove files produced by the older, leaky view format.
    for split in ('valid', 'test'):
        stale = os.path.join(out_dir, f'{split}.label.npy')
        if os.path.exists(stale):
            os.unlink(stale)

    counts = {}
    for split, (lo, hi) in C.SPLITS.items():
        sel = np.flatnonzero((date_all >= lo) & (date_all <= hi))   # flatnonzero 保序
        counts[split] = len(sel)
        np.save(os.path.join(out_dir, f'{split}.date.npy'), date_all[sel])
        for col, idx, dt in spec:
            arr = np.array([rows[i][idx] for i in sel], dtype=dt)
            np.save(os.path.join(out_dir, f'{split}.{col}.npy'), arr)

        user_ids = np.array([rows[i][1] for i in sel], dtype=np.int32)
        video_ids = np.array([rows[i][2] for i in sel], dtype=np.int32)
        for name in ('user_categorical', 'user_numeric'):
            np.save(os.path.join(out_dir, f'{split}.{name}.npy'), static[name][user_ids])
        for name in ('video_categorical', 'video_numeric'):
            np.save(os.path.join(out_dir, f'{split}.{name}.npy'), static[name][video_ids])

        if split == 'train':
            label = np.array([rows[i][8] for i in sel], dtype=np.int8)
            np.save(os.path.join(out_dir, 'train.label.npy'), label)
            # 反馈信号只为 train 落盘 —— valid/test 的根本不写出去
            tgt = np.array([targets[i] for i in sel], dtype=np.int64)
            for j, col in enumerate(C.TRAIN_ONLY_TARGET):
                np.save(os.path.join(out_dir, f'train.target.{col}.npy'), tgt[:, j])

    # 自检：候选视图只有 train 真标签，valid/test 标签文件根本不存在。
    tr = np.load(os.path.join(out_dir, 'train.label.npy'))
    assert set(np.unique(tr).tolist()) <= {0, 1}, f"train 标签异常: {np.unique(tr)}"
    assert not os.path.exists(os.path.join(out_dir, 'valid.label.npy'))
    assert not os.path.exists(os.path.join(out_dir, 'test.label.npy'))
    assert not any(f.startswith(('valid.target.', 'test.target.'))
                   for f in os.listdir(out_dir)), "反馈信号只能为 train 落盘"

    print(f"  预解析 -> {out_dir}/  "
          + " ".join(f"{k}={v:,d}" for k, v in counts.items()))
    print("  ✓ 仅 train.label 落盘；valid/test 标签与反馈均物理扣留")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='../KuaiRand-Pure/data')
    ap.add_argument('--out', default='views/agent')
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)

    for name in COPY_AS_IS:
        src = os.path.join(a.src, name)
        if not os.path.exists(src):
            print(f"  跳过（源不存在）: {name}")
            continue
        shutil.copyfile(src, os.path.join(a.out, name))
        print(f"  复制 {name}")

    for name in LOG_FILES:
        total, blanked = sanitize_log(os.path.join(a.src, name),
                                      os.path.join(a.out, name))
        print(f"  脱敏 {name}: {total:,d} 行，其中 {blanked:,d} 行属 valid/test 段已置空")

    checked = verify(a.out)
    print(f"\n✓ 自检通过：{checked:,d} 个 valid/test 行，"
          f"{len(C.TRAIN_ONLY_TARGET)} 个反馈列全部为空")
    print(f"✓ 隔离文件未挂载：{C.QUARANTINED_FILES + C.FORBIDDEN_TRAIN_FILES}")

    parse_to_arrays(a.out)

    # 视图设为只读，防止 orchestrator 自己手滑写坏
    for root, _, files in os.walk(a.out):
        for name in files:
            os.chmod(os.path.join(root, name), 0o444)
    print(f"✓ {a.out}/ 已置为只读 (0444)")


if __name__ == '__main__':
    main()
