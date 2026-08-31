"""Split manifest —— row_id 与评测口径的唯一权威，G0 的校验对象。

manifest 一旦生成就冻结。每轮迭代首尾各校验一次；任何 hash 变化 = 候选试图篡改
评测基础设施，立即终止并回滚（计划 §4.1 G0）。

被 hash 保护的三类东西：
  1. 评测口径     kit 的 evaluate.py（字节不变，我们从不修改它）
  2. 数据视图     views/agent/ 的每个文件（候选看到的输入）
  3. 真标签来源   原始 KuaiRand-Pure CSV（evaluator 读它算分）

row_order_sha256 锁定「第 n 行是谁」，所以提交对齐不依赖 (user_id, video_id) ——
后者在 test 里 3.06% 重复，本来就不能当主键。
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import columns as C

MANIFEST_NAME = 'manifest.json'
CHUNK = 1 << 20


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def row_order_sha256(parsed_dir, split):
    """对 (user_id, video_id, date) 逐行拼接取 sha256 —— 锁定行序与行身份。"""
    h = hashlib.sha256()
    cols = [np.load(os.path.join(parsed_dir, f'{split}.{c}.npy'))
            for c in ('user_id', 'video_id', 'date')]
    for u, v, d in zip(*cols):
        h.update(f'{u},{v},{d}\n'.encode())
    return h.hexdigest()


def build(view_dir, src_dir, kit_dir, out_path):
    project_root = os.path.dirname(os.path.abspath(out_path))
    parsed = os.path.join(view_dir, 'parsed')
    m = {
        'version': 3,
        'splits': {},
        'view_files': {},
        'source_files': {},
        'kit_files': {},
        'environment_files': {},
        'columns': {
            'inference_feature': C.INFERENCE_FEATURE,
            'train_only_target': C.TRAIN_ONLY_TARGET,
            'label': C.LABEL,
            'withheld_sentinel': C.WITHHELD,
        },
        'not_mounted': C.QUARANTINED_FILES + C.FORBIDDEN_TRAIN_FILES,
    }

    for split, (lo, hi) in C.SPLITS.items():
        n = len(np.load(os.path.join(parsed, f'{split}.user_id.npy')))
        entry = {
            'n_rows': int(n),
            'date_range': [lo, hi],
            'row_order_sha256': row_order_sha256(parsed, split),
        }
        label_path = os.path.join(parsed, f'{split}.label.npy')
        entry['candidate_label_exposed'] = os.path.exists(label_path)
        if split == 'train':
            if not os.path.exists(label_path):
                raise FileNotFoundError('候选视图缺少 train.label.npy')
            arr = np.load(label_path)
            entry['view_label_sha256'] = hashlib.sha256(arr.tobytes()).hexdigest()
        elif os.path.exists(label_path):
            raise ValueError(f'候选视图不得包含 {split}.label.npy')
        m['splits'][split] = entry

    for dirpath, _, files in os.walk(view_dir):
        for name in sorted(files):
            p = os.path.join(dirpath, name)
            m['view_files'][os.path.relpath(p, view_dir)] = file_sha256(p)

    for name in sorted(os.listdir(src_dir)):
        m['source_files'][name] = file_sha256(os.path.join(src_dir, name))

    for name in ('evaluate.py', 'data.py', 'submit.py'):
        p = os.path.join(kit_dir, name)
        if os.path.exists(p):
            m['kit_files'][name] = file_sha256(p)

    for name in ('requirements-candidate.txt', 'env.lock.json'):
        p = os.path.join(project_root, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f'候选环境文件缺失: {name}')
        m['environment_files'][name] = file_sha256(p)

    lock = load(os.path.join(project_root, 'env.lock.json'))
    requirements_hash = file_sha256(
        os.path.join(project_root, 'requirements-candidate.txt'))
    if lock.get('requirements_sha256') != requirements_hash:
        raise ValueError(
            'env.lock.json 与 requirements-candidate.txt 不一致；'
            '请先运行 bash trusted/make_venv.sh')

    with open(out_path, 'w') as fh:
        json.dump(m, fh, indent=2, sort_keys=True)
    return m


def load(path):
    with open(path) as fh:
        return json.load(fh)


def recursive_files(root):
    """返回目录下规范化的相对文件集合；G0 必须同时拒绝缺失与新增文件。"""
    out = set()
    for dirpath, _, files in os.walk(root):
        for name in files:
            out.add(os.path.relpath(os.path.join(dirpath, name), root))
    return out


def verify(manifest_path, view_dir, src_dir, kit_dir, check_sources=True):
    """G0。返回 (ok, 不一致项列表)。"""
    m = load(manifest_path)
    bad = []
    if m.get('version') != 3:
        bad.append(f"manifest 版本过期: {m.get('version')!r}（需要 3）")

    expected_view = set(m['view_files'])
    actual_view = recursive_files(view_dir)
    for rel in sorted(actual_view - expected_view):
        bad.append(f'view 出现未授权文件: {rel}')
    for rel in sorted(expected_view - actual_view):
        bad.append(f'view 缺失: {rel}')

    for rel, want in m['view_files'].items():
        p = os.path.join(view_dir, rel)
        if os.path.exists(p) and file_sha256(p) != want:
            bad.append(f'view 被修改: {rel}')

    for name, want in m['kit_files'].items():
        p = os.path.join(kit_dir, name)
        if not os.path.exists(p):
            bad.append(f'kit 缺失: {name}')
        elif file_sha256(p) != want:
            bad.append(f'kit 被修改: {name}')

    root = os.path.dirname(os.path.abspath(manifest_path))
    expected_environment = {'requirements-candidate.txt', 'env.lock.json'}
    if set(m.get('environment_files', {})) != expected_environment:
        bad.append('manifest 未完整锁定 requirements-candidate.txt 与 env.lock.json')
    for name, want in m.get('environment_files', {}).items():
        p = os.path.join(root, name)
        if not os.path.exists(p):
            bad.append(f'候选环境文件缺失: {name}')
        elif file_sha256(p) != want:
            bad.append(f'候选环境文件被修改: {name}')
    requirements_path = os.path.join(root, 'requirements-candidate.txt')
    lock_path = os.path.join(root, 'env.lock.json')
    if os.path.exists(requirements_path) and os.path.exists(lock_path):
        try:
            lock = load(lock_path)
            if lock.get('requirements_sha256') != file_sha256(requirements_path):
                bad.append('env.lock.json 与 requirements-candidate.txt 不一致')
        except (OSError, ValueError, TypeError):
            bad.append('env.lock.json 不是有效的环境锁文件')

    if check_sources:
        for name, want in m['source_files'].items():
            p = os.path.join(src_dir, name)
            if not os.path.exists(p):
                bad.append(f'源数据缺失: {name}')
            elif file_sha256(p) != want:
                bad.append(f'源数据被修改: {name}')

    parsed = os.path.join(view_dir, 'parsed')
    for split, entry in m['splits'].items():
        got = row_order_sha256(parsed, split)
        if got != entry['row_order_sha256']:
            bad.append(f'{split} 行序被改动')

    return (not bad), bad


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', default=os.path.join(root, 'views/agent'))
    ap.add_argument('--src', default=os.path.join(root, '../KuaiRand-Pure/data'))
    ap.add_argument('--kit', default=os.path.join(root, '../kuairand-starter-kit'))
    ap.add_argument('--out', default=os.path.join(root, MANIFEST_NAME))
    ap.add_argument('--verify', action='store_true', help='校验而不是生成')
    a = ap.parse_args()

    if a.verify:
        ok, bad = verify(a.out, a.view, a.src, a.kit)
        print('✓ G0 通过：manifest 与磁盘一致' if ok else '✗ G0 失败：')
        for b in bad:
            print('   ', b)
        sys.exit(0 if ok else 1)

    m = build(a.view, a.src, a.kit, a.out)
    print(f"已写出 {a.out}")
    for split, e in m['splits'].items():
        print(f"  {split:5s} {e['n_rows']:>9,d} 行  row_order {e['row_order_sha256'][:16]}…")
    print(f"  受保护文件: view {len(m['view_files'])} / 源数据 {len(m['source_files'])}"
          f" / kit {len(m['kit_files'])} / env {len(m['environment_files'])}")


if __name__ == '__main__':
    main()
