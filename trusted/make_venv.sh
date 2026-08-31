#!/usr/bin/env bash
# 构建挂进沙箱的 pinned model venv。
#
# 任意开源模型库都可以在运行前加入 requirements-candidate.txt；构建后精确版本和
# import roots 写进 env.lock.json。正式 run 中该环境只读且断网，所以「任意库」不会
# 退化为候选运行时下载代码或改变依赖。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv"
REQ="$ROOT/requirements-candidate.txt"
LOCK="$ROOT/env.lock.json"
TORCHREC_SPEC="$(sed -n 's/^# no-deps: //p' "$REQ")"

if [[ "$TORCHREC_SPEC" != torchrec==* ]]; then
  echo "requirements-candidate.txt 缺少唯一的 '# no-deps: torchrec==...' 锁定项" >&2
  exit 2
fi

rm -rf "$VENV"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --disable-pip-version-check -r "$REQ"
"$VENV/bin/python" -m pip install --disable-pip-version-check --no-deps "$TORCHREC_SPEC"
"$VENV/bin/python" "$ROOT/trusted/lock_env.py" --requirements "$REQ" --out "$LOCK"
"$VENV/bin/python" - <<'PY'
import lightgbm
import numpy
import recbole
import scipy
import sklearn
import torch
import torchrec
from torchrec.modules.embedding_modules import EmbeddingBagCollection

assert torch.version.cuda is None, f"expected CPU-only torch, got CUDA {torch.version.cuda}"
assert not torch.cuda.is_available(), "candidate environment must not expose CUDA"
print(f"  python model environment ready: numpy={numpy.__version__} "
      f"scipy={scipy.__version__} sklearn={sklearn.__version__} "
      f"lightgbm={lightgbm.__version__} torch={torch.__version__} "
      f"recbole={recbole.__version__} torchrec={torchrec.__version__} "
      f"device=cpu embedding_bag={EmbeddingBagCollection.__name__}")
PY

echo "  venv 大小: $(du -sh "$VENV" | cut -f1)"
echo "✓ $VENV"
