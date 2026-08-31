"""确定性门禁 G0–G5（计划 §4.1）。

设计原则：**能靠结构保证的，就不靠检查**。
  - G1（数据访问）不在这里 —— 它由 bwrap 挂载命名空间实现，是「不可能」而非「检查」。
  - G2 在这里，但它的作用是**在烧掉一次运行之前**拦住泄漏方案；
    即使它漏判，评测行的反馈值在磁盘上也不存在，推理时必然崩。

所以 G2 是省预算的，不是最后一道防线。这个次序很重要：任何把 LLM 判定当作
唯一防线的设计（如 MLE-STAR 的 data leakage checker）在本题都不够。
"""
import ast
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import columns as C
import manifest as M

BLOCK_RE = re.compile(r'#\s*<<<(BLOCK|END):([a-z_]+)>>>')

BLOCKS = ['data_view', 'features', 'target', 'model', 'loss', 'train', 'predict']

# 只有这些 block 允许出现反馈信号列名。features/model/predict 里出现 = 把答案当特征。
TARGET_OK_BLOCKS = {'target', 'loss', 'train'}

BASE_IMPORT_ALLOW = {
    'numpy', 'dataview', 'stable_ops', 'evaluate',
    'math', 'json', 'time', 'sys', 'argparse', 'collections', 'itertools',
    'functools', 'dataclasses', 'typing', 'random', 'heapq', 'bisect',
    'copy', 'warnings', 'statistics', 'array', 'struct', 'hashlib',
}
IMPORT_DENY = {
    'os', 'subprocess', 'socket', 'urllib', 'requests', 'http', 'ftplib',
    'ctypes', 'importlib', 'shutil', 'pathlib', 'glob', 'pickle', 'marshal',
    'multiprocessing', 'threading', 'csv', 'sqlite3', 'tempfile', 'io',
    'pip', 'setuptools', 'aiohttp', 'httpx', 'openai', 'urllib3', 'websockets',
}
# 直读数据的通道。候选必须走 dataview。
FORBIDDEN_CALLS = {'open', 'eval', 'exec', 'compile', '__import__', 'globals',
                   'input', 'breakpoint'}
FORBIDDEN_ATTR_CALLS = {
    ('np', 'load'), ('np', 'fromfile'), ('np', 'genfromtxt'), ('np', 'loadtxt'),
    ('numpy', 'load'), ('numpy', 'fromfile'),
}


def locked_import_roots(lock_path=None):
    """Return third-party roots installed in the frozen candidate environment.

    The gate no longer encodes a model-family allowlist.  A library becomes eligible by being
    installed before the run and recorded in env.lock.json; dangerous host/network modules remain
    denied independently of that environment.
    """
    lock_path = lock_path or os.path.join(ROOT, 'env.lock.json')
    try:
        with open(lock_path, encoding='utf-8') as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return set()
    return {
        str(root).split('.', 1)[0]
        for root in payload.get('import_roots', [])
        if isinstance(root, str) and root.isidentifier()
    } - IMPORT_DENY


def import_allow(lock_path=None):
    return BASE_IMPORT_ALLOW | locked_import_roots(lock_path)


IMPORT_ALLOW = import_allow()


class GateResult:
    def __init__(self, gate, ok, violations=None, info=None):
        self.gate, self.ok = gate, ok
        self.violations = violations or []
        self.info = info or {}

    def as_event(self, action=None):
        return {'type': 'POLICY_BLOCK' if not self.ok else 'GATE_PASS',
                'gate': self.gate, 'violations': self.violations,
                'info': self.info, 'action': action}

    def __bool__(self):
        return self.ok

    def __repr__(self):
        s = 'PASS' if self.ok else 'BLOCK'
        return f'<{self.gate} {s}{"" if self.ok else ": " + "; ".join(self.violations[:3])}>'


# --------------------------------------------------------------------------
def parse_blocks(src):
    """把源码按哨兵注释切成 block，返回 {name: (lo_line, hi_line)}（1-based，闭区间）。"""
    spans, open_at = {}, {}
    for i, line in enumerate(src.splitlines(), start=1):
        m = BLOCK_RE.search(line)
        if not m:
            continue
        kind, name = m.group(1), m.group(2)
        if kind == 'BLOCK':
            open_at[name] = i
        elif name in open_at:
            spans[name] = (open_at.pop(name), i)
    return spans


def block_of(spans, lineno):
    for name, (lo, hi) in spans.items():
        if lo <= lineno <= hi:
            return name
    return None


def scaffold_sha256(src):
    """block 之外的「脚手架」（import、main、输出契约）的指纹。

    agent 只能改 block 内部；脚手架冻结意味着它无法偷改输出契约、绕过 dataview、
    或往 main() 里塞一个 open()。有了这条不变量，G2 的 open() 检查只需覆盖 block 内。
    """
    import hashlib
    spans = parse_blocks(src)
    inside = set()
    for lo, hi in spans.values():
        inside.update(range(lo, hi + 1))
    kept = [ln for i, ln in enumerate(src.splitlines(), start=1) if i not in inside]
    return hashlib.sha256('\n'.join(kept).encode()).hexdigest()


def block_sha256s(src):
    """返回每个声明式 block（含哨兵）的内容指纹。"""
    import hashlib
    lines = src.splitlines()
    return {name: hashlib.sha256(
                '\n'.join(lines[lo - 1:hi]).encode()).hexdigest()
            for name, (lo, hi) in parse_blocks(src).items()}


def frozen_scaffold_sha256():
    path = os.path.join(ROOT, 'candidate', 'scaffold.sha256')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return fh.read().strip()


# --------------------------------------------------------------------------
def g0_integrity(manifest_path=None, view=None, src=None, kit=None):
    """评测口径、数据视图、行序的 hash 未被改动。"""
    manifest_path = manifest_path or os.path.join(ROOT, 'manifest.json')
    ok, bad = M.verify(manifest_path,
                       view or os.path.join(ROOT, 'views/agent'),
                       src or os.path.join(ROOT, '..', 'KuaiRand-Pure', 'data'),
                       kit or os.path.join(ROOT, '..', 'kuairand-starter-kit'))
    return GateResult('G0', ok, bad)


def g2_lineage(src_path):
    """数据血缘：反馈信号不得进入推理路径；不得绕过 dataview 直读数据。"""
    with open(src_path) as fh:
        src = fh.read()
    try:
        tree = ast.parse(src, filename=src_path)
    except SyntaxError as e:
        return GateResult('G2', False, [f'语法错误，无法做血缘分析: {e}'])

    spans = parse_blocks(src)
    missing = [b for b in BLOCKS if b not in spans]
    v = []
    if missing:
        v.append(f'缺少 block 哨兵: {missing}')

    for node in ast.walk(tree):
        blk = block_of(spans, getattr(node, 'lineno', -1))

        # 1) import 白/黑名单
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name.split('.')[0] for a in node.names]
                     if isinstance(node, ast.Import)
                     else [(node.module or '').split('.')[0]])
            for n in names:
                if n in IMPORT_DENY:
                    v.append(f'第 {node.lineno} 行 import {n}：不在允许依赖内')
                elif n and n not in IMPORT_ALLOW:
                    v.append(f'第 {node.lineno} 行 import {n}：未知依赖（沙箱内也不存在）')

        # 2) 绕过 dataview 直读数据。只查 block 内 —— block 外的脚手架由 G3 的
        #    scaffold hash 冻结，它里面那个写 meta.json 的 open() 是已知良性的。
        if isinstance(node, ast.Call) and blk is not None:
            f = node.func
            if isinstance(f, ast.Name) and f.id in FORBIDDEN_CALLS:
                v.append(f'第 {node.lineno} 行 block[{blk}] 调用 {f.id}()：'
                         f'数据只能经 dataview 取')
            if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                    and (f.value.id, f.attr) in FORBIDDEN_ATTR_CALLS):
                v.append(f'第 {node.lineno} 行 block[{blk}] 调用 '
                         f'{f.value.id}.{f.attr}()：禁止直读文件')

        # 3) 反馈信号出现在推理路径的 block 里
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in C.TRAIN_ONLY_TARGET and node.value != C.LABEL:
                if blk is not None and blk not in TARGET_OK_BLOCKS:
                    v.append(f'第 {node.lineno} 行 block[{blk}] 出现反馈信号 '
                             f'{node.value!r}：曝光后信号只能当训练目标，不能当特征')
        if isinstance(node, ast.Attribute) and node.attr in C.TRAIN_ONLY_TARGET:
            if blk is not None and blk not in TARGET_OK_BLOCKS:
                v.append(f'第 {node.lineno} 行 block[{blk}] 访问 .{node.attr}：'
                         f'曝光后信号不是推理特征')
        if isinstance(node, ast.Attribute) and node.attr == 'label':
            if blk is not None and blk != 'target':
                v.append(f'第 {node.lineno} 行 block[{blk}] 访问 .label：'
                         '只有 target block 可读取 train.label；valid/test 标签已物理扣留')

    return GateResult('G2', not v, v, {'blocks': sorted(spans)})


def g3_code(src_path, patch_scope=None, expect_scaffold=None,
            parent_path=None, primary_block=None, max_patch_blocks=3):
    """可解析、block 结构完整、脚手架未被改动、patch_scope 合法。

    真正的 smoke run 由 orchestrator 在沙箱里做（那需要执行）。
    """
    v = []
    with open(src_path) as fh:
        src = fh.read()
    try:
        tree = ast.parse(src, filename=src_path)
    except SyntaxError as e:
        return GateResult('G3', False, [f'第 {e.lineno} 行语法错误: {e.msg}'],
                          {'error_class': 'SYNTAX'})

    spans = parse_blocks(src)
    missing = [b for b in BLOCKS if b not in spans]
    if missing:
        v.append(f'缺少 block 哨兵: {missing}')

    # A Python loop nested inside another loop in the loss block is an unsafe pairwise
    # implementation pattern: on user groups it becomes O(B^2), exactly the failure seen
    # in prior LambdaLoss/BPR attempts. Pair sampling must be bounded and vectorized.
    for outer in ast.walk(tree):
        if not isinstance(outer, (ast.For, ast.While)):
            continue
        if block_of(spans, getattr(outer, 'lineno', -1)) != 'loss':
            continue
        nested = [node for statement in outer.body for node in ast.walk(statement)
                  if isinstance(node, (ast.For, ast.While)) and node is not outer]
        if nested:
            v.append(f'第 {outer.lineno} 行 block[loss] 含嵌套 Python 循环：'
                     'pairwise loss 必须使用有界采样和向量化，禁止 O(B^2) 用户内枚举')
            break

    got_scaffold = scaffold_sha256(src)
    if expect_scaffold is None:
        expect_scaffold = frozen_scaffold_sha256()
    if expect_scaffold and got_scaffold != expect_scaffold:
        v.append('脚手架被修改：agent 只能改 block 内部，'
                 'import/main/输出契约是冻结的')

    if patch_scope is not None:
        unknown = [b for b in patch_scope if b not in BLOCKS]
        if unknown:
            v.append(f'patch_scope 含未知 block: {unknown}')
        if len(patch_scope) > max_patch_blocks:
            v.append(f'一轮改了 {len(patch_scope)} 个 block（上限 {max_patch_blocks}）：'
                     'delta 将无法归因')
        if not patch_scope:
            v.append('patch_scope 为空')
        if parent_path is None:
            v.append('缺少 parent_path：无法验证实际修改是否等于 patch_scope')
        elif os.path.exists(parent_path):
            with open(parent_path) as fh:
                parent_src = fh.read()
            parent_blocks = block_sha256s(parent_src)
            child_blocks = block_sha256s(src)
            actual = sorted(name for name in BLOCKS
                            if parent_blocks.get(name) != child_blocks.get(name))
            if set(actual) != set(patch_scope):
                v.append(f'实际修改 block {actual} 与声明 patch_scope '
                         f'{sorted(patch_scope)} 不一致')
            if primary_block is not None and primary_block not in actual:
                v.append(f'primary_block {primary_block!r} 未被实际修改')
        else:
            v.append(f'parent_path 不存在: {parent_path}')
    return GateResult('G3', not v, v,
                      {'blocks': sorted(spans), 'scaffold_sha256': got_scaffold})


def g4_runtime(sandbox_result):
    """运行期：超时 / 资源 / 退出码。错误分类供 §4.2 的恢复策略分派。"""
    r = sandbox_result
    if r.timed_out:
        return GateResult('G4', False, ['执行超时'], {'error_class': 'TIMEOUT'})
    if r.signal == 9:
        return GateResult('G4', False, ['被 SIGKILL（多半是内存超限）'],
                          {'error_class': 'OOM'})
    if r.exit_code != 0:
        tail = (r.stderr_tail or '').strip().splitlines()
        last = tail[-1] if tail else ''
        cls = 'RUNTIME_ERROR'
        for pat, name in [('SyntaxError', 'SYNTAX'), ('ImportError', 'IMPORT'),
                          ('ModuleNotFoundError', 'IMPORT'),
                          ('预测含 NaN/Inf', 'NAN'),
                          ('MemoryError', 'OOM'), ('AttributeError', 'ATTRIBUTE'),
                          ('FileNotFoundError', 'FILE_ACCESS'),
                          ('PermissionError', 'FILE_ACCESS')]:
            if pat in (r.stderr_tail or ''):
                cls = name
                break
        return GateResult('G4', False, [f'退出码 {r.exit_code}: {last[:200]}'],
                          {'error_class': cls})
    return GateResult('G4', True, [], {'wall_s': r.wall_s, 'cpu_s': r.cpu_s,
                                       'max_rss_mb': r.max_rss_mb})


def g5_output(pred_path, split, manifest_path=None):
    """输出：长度对齐 manifest、全为有限实数。row_id 由可信侧生成，候选碰不到。"""
    manifest_path = manifest_path or os.path.join(ROOT, 'manifest.json')
    v = []
    if not os.path.exists(pred_path):
        return GateResult('G5', False, [f'预测文件不存在: {pred_path}'],
                          {'error_class': 'NO_OUTPUT'})
    try:
        pred = np.load(pred_path)
    except Exception as e:
        return GateResult('G5', False, [f'预测无法读取: {e}'],
                          {'error_class': 'BAD_OUTPUT'})

    n_expect = M.load(manifest_path)['splits'][split]['n_rows']
    if pred.ndim != 1:
        v.append(f'预测应为一维，实际 {pred.shape}')
    if pred.shape[0] != n_expect:
        v.append(f'预测 {pred.shape[0]:,d} 行，{split} 应为 {n_expect:,d} 行')
    n_bad = int((~np.isfinite(pred)).sum()) if pred.size else 0
    if n_bad:
        v.append(f'预测含 {n_bad:,d} 个 NaN/Inf')
    if pred.size and float(np.ptp(pred)) == 0.0:
        v.append('预测为常数：用户内排序退化，任何模型都不该输出常数')

    cls = 'FORMAT' if v else None
    return GateResult('G5', not v, v,
                      {'n_rows': int(pred.shape[0]) if pred.ndim == 1 else -1,
                       'error_class': cls})


def run_static_gates(src_path, patch_scope=None, parent_path=None,
                     primary_block=None, expect_scaffold=None,
                     manifest_path=None, view=None, src=None, kit=None,
                     max_patch_blocks=3):
    """执行前的确定性门禁串：G0 -> G3 -> G2。任一失败即短路。"""
    for res in (g0_integrity(manifest_path, view, src, kit),
                g3_code(src_path, patch_scope, expect_scaffold,
                        parent_path, primary_block, max_patch_blocks),
                g2_lineage(src_path)):
        if not res.ok:
            return res
    return GateResult('static', True, [], {'passed': ['G0', 'G3', 'G2']})


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('src', nargs='?', default=os.path.join(ROOT, 'candidate/pipeline.py'))
    a = ap.parse_args()
    for res in (g0_integrity(), g3_code(a.src), g2_lineage(a.src)):
        mark = '✓' if res.ok else '✗'
        print(f'{mark} {res.gate}: {"通过" if res.ok else ""}')
        for x in res.violations:
            print(f'    - {x}')
        if res.info.get('blocks'):
            print(f'    blocks: {res.info["blocks"]}')
