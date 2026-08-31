"""bwrap 沙箱 —— hidden-test 隔离的执行侧（计划 §4.1 G1 / §5.3 F2）。

隔离靠挂载命名空间，不靠源码扫描：真 test 标签、宿主文件系统、网络在沙箱里
**根本不存在**，候选做什么都够不着。已实测：
  - 宿主路径不可达
  - /data 只见到绑定内容
  - socket.create_connection -> OSError（顺带满足「禁止静默联网安装」）
  - 锁定的第三方模型库及其 native runtime 正常

已知限制：本机 bwrap 非 setuid，`--proc /proc` 会报 Operation not permitted，故省略。
连带的设计决定：**候选模型库默认单线程，seed 并行由 orchestrator 在沙箱外做** ——
这同时避免单个候选打满 128 核。

stdout/stderr 由宿主侧打开后把 fd 传进去，写到沙箱**不可达**的路径，
所以候选无法篡改自己的运行日志。
"""
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field, asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 沙箱内的挂载点
MOUNT_VENV, MOUNT_DATA, MOUNT_TASK, MOUNT_WORK = '/venv', '/data', '/task', '/work'

# 系统只读绑定。故意不含 /home、/root、/mnt —— 数据源和真标签都在 /home 下。
SYSTEM_ROBINDS = ['/usr', '/lib', '/lib64', '/bin', '/etc']

# nproc 默认 None：RLIMIT_NPROC 是按 **UID 全局**计数的，不是按进程树。设成小值会
# (a) 因为宿主已有几十个进程而直接让 bwrap 建不出命名空间，(b) 在多 run 并行时互相干扰，
# (c) 真被 fork bomb 打满时连 orchestrator 自己都 fork 不出来 —— 比不设更糟。
# 防 fork bomb 改为依赖：pid namespace（--unshare-all）+ --die-with-parent + 超时 killpg。
DEFAULTS = dict(timeout_s=1800, mem_gb=32, cpu_s=3600, nproc=None, fsize_mb=2048)


@dataclass
class SandboxResult:
    ok: bool
    exit_code: int
    signal: int | None
    timed_out: bool
    wall_s: float
    cpu_s: float
    max_rss_mb: float
    stdout_path: str
    stderr_path: str
    stdout_tail: str = ''
    stderr_tail: str = ''
    limits: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


def _tail(path, n_bytes=8000):
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as fh:
            if size > n_bytes:
                fh.seek(size - n_bytes)
            return fh.read().decode('utf-8', 'replace')
    except OSError:
        return ''


def build_argv(workspace, cmd, venv=None, data=None, task=None, extra_robinds=()):
    """拼出 bwrap 命令行。cmd 是沙箱内的 argv，例如 ['/venv/bin/python','/work/pipeline.py']。"""
    venv = venv or os.path.join(ROOT, 'venv')
    data = data or os.path.join(ROOT, 'views/agent')
    task = task or os.path.join(ROOT, 'task_spec')

    argv = ['bwrap']
    for p in SYSTEM_ROBINDS:
        if os.path.exists(p):
            argv += ['--ro-bind', p, p]
    argv += ['--ro-bind', os.path.abspath(venv), MOUNT_VENV,
             '--ro-bind', os.path.abspath(data), MOUNT_DATA,
             '--ro-bind', os.path.abspath(task), MOUNT_TASK,
             '--bind', os.path.abspath(workspace), MOUNT_WORK]
    for src, dst in extra_robinds:
        argv += ['--ro-bind', os.path.abspath(src), dst]
    argv += ['--dev', '/dev', '--tmpfs', '/tmp',
             '--unshare-all',          # 含断网
             '--die-with-parent',
             '--new-session',          # 防 TIOCSTI 类逃逸
             # 不装 PID-1 reaper，让目标进程直接成为 bwrap 的 wait 对象。
             # 否则 os.wait4 拿到的 rusage 是 reaper 自己的，cpu_s 恒为 0 ——
             # §8 的算力记账会全部失真。候选按设计是单进程，不需要 reaper。
             '--as-pid-1',
             '--chdir', MOUNT_WORK,
             '--setenv', 'HOME', MOUNT_WORK,
             '--setenv', 'TMPDIR', '/tmp',
             '--setenv', 'PYTHONDONTWRITEBYTECODE', '1',
             '--setenv', 'TRACK2_DATA', MOUNT_DATA,
             # Official workers are CPU-only.  Keep the model contract identical even on a
             # developer host that happens to have CUDA/ROCm devices installed.
             '--setenv', 'CUDA_VISIBLE_DEVICES', '',
             '--setenv', 'HIP_VISIBLE_DEVICES', '',
             '--setenv', 'PYTORCH_ENABLE_MPS_FALLBACK', '0',
             # 单线程：并行由 orchestrator 做，避免 128 核被一个候选打满
             '--setenv', 'OMP_NUM_THREADS', '1',
             '--setenv', 'OPENBLAS_NUM_THREADS', '1',
             '--setenv', 'MKL_NUM_THREADS', '1']
    argv += ['--'] + list(cmd)
    return argv


def limit_argv(argv, mem_gb, cpu_s, nproc, fsize_mb):
    """用 util-linux prlimit 设置可继承限制，避免 threaded orchestrator 中的 preexec_fn。"""
    out = ['prlimit', f'--as={mem_gb << 30}', f'--cpu={cpu_s}:{cpu_s + 10}',
           f'--fsize={fsize_mb << 20}', '--core=0']
    if nproc is not None:
        out.append(f'--nproc={nproc}')
    return out + ['--'] + list(argv)


def run(workspace, cmd, log_dir, **kw):
    """在沙箱里跑一条命令。log_dir 必须在沙箱**外**（候选不可达）。

    返回 SandboxResult。资源超限、超时、非零退出都不抛异常 —— 由 G4 分类处理。
    """
    opts = {**DEFAULTS, **kw}
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, 'stdout.txt')
    err_path = os.path.join(log_dir, 'stderr.txt')

    argv = build_argv(workspace, cmd,
                      venv=kw.pop('venv', None), data=kw.pop('data', None),
                      task=kw.pop('task', None),
                      extra_robinds=kw.pop('extra_robinds', ()))
    argv = limit_argv(argv, opts['mem_gb'], opts['cpu_s'],
                      opts['nproc'], opts['fsize_mb'])

    timed_out = False
    t0 = time.time()
    with open(out_path, 'wb') as fo, open(err_path, 'wb') as fe:
        proc = subprocess.Popen(
            argv, stdout=fo, stderr=fe, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + opts['timeout_s']
        while True:
            waited, status, ru = os.wait4(proc.pid, os.WNOHANG)
            if waited == proc.pid:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                time.sleep(0.25)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                _, status, ru = os.wait4(proc.pid, 0)
                break
            time.sleep(0.05)
    wall = time.time() - t0

    sig = status & 0x7F
    code = status >> 8 if sig == 0 else -sig
    return SandboxResult(
        ok=(sig == 0 and code == 0 and not timed_out),
        exit_code=code, signal=(sig or None), timed_out=timed_out,
        wall_s=round(wall, 2),
        cpu_s=round(ru.ru_utime + ru.ru_stime, 2),
        max_rss_mb=round(ru.ru_maxrss / 1024, 1),
        stdout_path=out_path, stderr_path=err_path,
        stdout_tail=_tail(out_path), stderr_tail=_tail(err_path),
        limits={k: opts[k] for k in DEFAULTS},
    )


def selftest():
    """确认隔离性质仍然成立。作为 F2 的常规回归。"""
    import tempfile
    probe = '''
import os, socket, json
r = {}
r["cwd"] = os.getcwd()
r["data_visible"] = sorted(os.listdir("/data"))[:3]
r["host_data_reachable"] = os.path.exists("%s")
r["home_reachable"] = os.path.exists("/home")
try:
    socket.create_connection(("1.1.1.1", 80), timeout=2); r["net"] = "OPEN"
except Exception as e:
    r["net"] = type(e).__name__
import lightgbm, numpy, recbole, scipy, sklearn, torch, torchrec
r["libraries"] = {
    "numpy": numpy.__version__, "scipy": scipy.__version__,
    "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__,
    "torch": torch.__version__, "recbole": recbole.__version__,
    "torchrec": torchrec.__version__,
}
r["torch_cpu_only"] = torch.version.cuda is None and not torch.cuda.is_available()
print(json.dumps(r))
''' % os.path.join(ROOT, '..', 'KuaiRand-Pure', 'data')

    with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as logs:
        with open(os.path.join(ws, 'probe.py'), 'w') as fh:
            fh.write(probe)
        res = run(ws, ['/venv/bin/python', '/work/probe.py'], logs, timeout_s=60)
        print(f'exit={res.exit_code} wall={res.wall_s}s cpu={res.cpu_s}s '
              f'rss={res.max_rss_mb}MB')
        if not res.ok:
            print(res.stderr_tail)
            return False
        r = json.loads(res.stdout_tail)
        checks = [
            ('cwd == /work', r['cwd'] == '/work'),
            ('宿主真标签不可达', r['host_data_reachable'] is False),
            ('/home 不可达', r['home_reachable'] is False),
            ('网络已断', r['net'] != 'OPEN'),
            ('锁定模型库可用', all(r['libraries'].values())),
            ('PyTorch 仅 CPU', r['torch_cpu_only'] is True),
            ('/data 已挂载', len(r['data_visible']) > 0),
        ]
        for name, ok in checks:
            print(f'  {"✓" if ok else "✗"} {name}')
        return all(ok for _, ok in checks)


if __name__ == '__main__':
    import sys
    sys.exit(0 if selftest() else 1)
