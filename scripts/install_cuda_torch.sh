#!/usr/bin/env bash
# 把 venv 里的 CPU 版 torch 换成 CUDA 版（RTX 40 系用户可选执行）。
#
# 为什么不用 uv 直接拉：
#   实测 download.pytorch.org 单连接 curl 可达 3-4MB/s，
#   但 uv 边下边解压的吞吐只有 ~0.5MB/s（2.5GB 要一个多小时）。
#   国内镜像反而更慢（阿里云 565KB/s）。所以这里用 curl 预下载 wheel，
#   再用 uv pip 本地安装，全程可控且支持断点续传。
#
# ⚠️ 必须用与当前 CPU 版**完全相同的版本号**的 CUDA wheel。
#    本机 CPU 版是 2.14.1，若装成 2.6.0 就是降级，会破坏依赖。
#
# 用法：bash scripts/install_cuda_torch.sh [cu126|cu130]

set -euo pipefail

CUDA_TAG="${1:-cu126}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL_DIR="${PROJECT_ROOT}/.wheels"
mkdir -p "$WHEEL_DIR"

cd "$PROJECT_ROOT"

echo "==> 检查 CUDA 版是否已装"
if uv run python -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)" 2>/dev/null; then
  echo "已是 CUDA 版：$(uv run python -c 'import torch; print(torch.__version__)')"
  exit 0
fi

# 从当前 CPU 版读出版本号，保证装的是同一个版本而不是降级
TORCH_VER="$(uv run python -c 'import torch; print(torch.__version__.split("+")[0])')"
PY_TAG="cp312"
WHEEL_NAME="torch-${TORCH_VER}+${CUDA_TAG}-${PY_TAG}-${PY_TAG}-win_amd64.whl"
URL="https://download.pytorch.org/whl/${CUDA_TAG}/${WHEEL_NAME//+/%2B}"

echo "==> 目标：torch ${TORCH_VER}+${CUDA_TAG}（与当前 CPU 版 ${TORCH_VER} 同版本）"
echo "    ${URL}"

if [ -f "${WHEEL_DIR}/${WHEEL_NAME}" ]; then
  echo "==> wheel 已存在，跳过下载"
else
  # -C - 断点续传：网络中断后重跑本脚本即可接着下
  curl -L -C - --retry 3 --retry-delay 5 \
    -o "${WHEEL_DIR}/${WHEEL_NAME}" "$URL"
fi

echo "==> 校验完整性"
uv run python - <<PY
import sys, zipfile, pathlib
p = pathlib.Path(r"${WHEEL_DIR}/${WHEEL_NAME}")
if not p.exists():
    sys.exit(f"文件不存在：{p}")
try:
    with zipfile.ZipFile(p) as z:
        bad = z.testzip()
        if bad:
            sys.exit(f"压缩包损坏，首个坏文件：{bad}")
        print(f"  OK  {p.stat().st_size / 1e9:.2f} GB，{len(z.namelist())} 个文件")
except zipfile.BadZipFile:
    sys.exit("文件不完整或损坏。删除该文件后重跑本脚本（支持断点续传）")
PY

echo "==> 安装 CUDA 版 torch"
uv pip install --force-reinstall --no-deps "${WHEEL_DIR}/${WHEEL_NAME}"

echo "==> 验证"
uv run python -c "
import torch
print(f'  torch {torch.__version__}')
print(f'  cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  device: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'  capability: sm_{cap[0]}{cap[1]}')
else:
    print('  警告：CUDA 仍不可用，请检查驱动版本')
"

echo
echo "完成。重跑索引即使用 GPU 加速。"
echo
echo "⚠️  重要：之后必须用 'uv run --no-sync'，不能用 'uv run'。"
echo "    原因：uv run 会按 pyproject.toml 重新同步环境，"
echo "          把刚装的 CUDA 版 torch 覆盖回 PyPI 的 CPU 版。"
echo "    正确： uv run --no-sync python -m cs408rag.ingestion.indexer"
echo "    错误： uv run python -m cs408rag.ingestion.indexer   # 会被还原成 CPU"
echo
echo "    （亦可直接用 .venv/Scripts/python.exe，同样不会触发同步）"
echo
echo "注意：.wheels/ 目录下的 wheel 有数 GB，确认装好后可自行删除以释放空间。"