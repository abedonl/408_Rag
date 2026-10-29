#!/usr/bin/env bash
# 一键启动 408-RAG。
#
#   bash scripts/start.sh              起 Streamlit 界面（默认，会打开浏览器）
#   bash scripts/start.sh --api        同时起 FastAPI（:8123，带 /docs 接口文档）
#   bash scripts/start.sh --no-browser 不自动开浏览器
#   bash scripts/start.sh --foreground 前台运行，Ctrl+C 一并停掉
#   bash scripts/start.sh --bootstrap  数据缺失时先跑完整数据管道（耗时约 10 分钟）
#
# 为什么所有命令都用 `uv run --no-sync`：
#   `uv run` 会按 pyproject.toml 重新同步环境，把手工装的 CUDA 版 torch
#   覆盖回 PyPI 的 CPU 版，BGE-M3 会从 913 条/秒掉到 3.5 条/秒。
#   --no-sync 跳过同步，保住 GPU 加速。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

UI_PORT="${UI_PORT:-8501}"
API_PORT="${API_PORT:-8123}"

WITH_API=0
OPEN_BROWSER=1
FOREGROUND=0
BOOTSTRAP=0

for arg in "$@"; do
  case "$arg" in
    --api) WITH_API=1 ;;
    --no-browser) OPEN_BROWSER=0 ;;
    --foreground) FOREGROUND=1 ;;
    --bootstrap) BOOTSTRAP=1 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数：$arg（--help 查看用法）" >&2; exit 1 ;;
  esac
done

PY="$PROJECT_ROOT/.venv/Scripts/python.exe"
[ -f "$PY" ] || PY="$PROJECT_ROOT/.venv/bin/python"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '    \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. 前置检查
# ---------------------------------------------------------------------------
say "检查运行环境"

command -v uv >/dev/null 2>&1 || die "找不到 uv。安装：https://docs.astral.sh/uv/"
ok "uv $(uv --version | awk '{print $2}')"

[ -f "$PY" ] || die "虚拟环境不存在（$PY）。先执行：uv venv --python 3.12 && uv sync"
ok "Python 环境就绪"

[ -f .env ] || { warn ".env 不存在，已从 .env.example 复制（需要自己填 API key）"; cp .env.example .env; }
ok ".env 存在"

# ---------------------------------------------------------------------------
# 2. Docker 与数据库
# ---------------------------------------------------------------------------
say "检查 Docker"
if docker info >/dev/null 2>&1; then
  ok "Docker 守护进程运行中"
else
  warn "Docker 未运行，尝试启动 Docker Desktop…"
  DD="/c/Program Files/Docker/Docker Desktop.exe"
  [ -f "$DD" ] || DD="/d/Program Files/Docker/Docker Desktop.exe"
  if [ -f "$DD" ]; then
    powershell.exe -NoProfile -Command "Start-Process -FilePath '$(cygpath -w "$DD" 2>/dev/null || echo "$DD")'" >/dev/null 2>&1 || true
    printf '    等待 Docker 就绪'
    for _ in $(seq 1 60); do
      docker info >/dev/null 2>&1 && { printf '\n'; ok "Docker 已就绪"; break; }
      printf '.'; sleep 3
    done
    docker info >/dev/null 2>&1 || die "Docker 启动超时。请手动启动 Docker Desktop 后重试"
  else
    die "找不到 Docker Desktop，请手动启动后重试"
  fi
fi

say "启动数据库容器（Qdrant + Neo4j）"
# 镜像不在本地时给个提示（Docker Hub 在本机不可达，需走镜像站）
if ! docker image inspect qdrant/qdrant:v1.20.0 >/dev/null 2>&1; then
  warn "本地没有 qdrant 镜像，正在从镜像站拉取（首次约 1-2 分钟）"
  bash scripts/pull_images.sh || die "镜像拉取失败"
fi

docker compose up -d >/dev/null
ok "容器已启动，等待健康检查…"

wait_healthy() {
  local name="$1" url="$2" tries="${3:-60}"
  for _ in $(seq 1 "$tries"); do
    if curl -sf -m 3 "$url" >/dev/null 2>&1; then ok "$name 就绪"; return 0; fi
    sleep 2
  done
  die "$name 健康检查超时"
}
wait_healthy "Qdrant" "http://localhost:6333/"
wait_healthy "Neo4j"  "http://localhost:7474/"

# ---------------------------------------------------------------------------
# 3. 数据完整性
# ---------------------------------------------------------------------------
say "检查数据"

need_bootstrap=0
if [ ! -f data/raw/quiz/questions.json ]; then warn "缺少真题库 data/raw/quiz/questions.json"; need_bootstrap=1; fi
if [ ! -d data/raw/wangdao/Markdown ]; then warn "缺少王道语料 data/raw/wangdao/Markdown"; need_bootstrap=1; fi
if [ ! -f data/processed/image_index.json ]; then warn "缺少图片索引"; need_bootstrap=1; fi
if [ ! -f data/eval/eval_set.json ]; then warn "缺少评测集"; need_bootstrap=1; fi

chunks=$("$PY" - <<'PY' 2>/dev/null || echo 0
from qdrant_client import QdrantClient
try:
    c = QdrantClient(url="http://localhost:6333")
    print(c.count("cs408_chunks", exact=True).count)
except Exception:
    print(0)
PY
)
if [ "${chunks:-0}" -lt 1000 ]; then warn "向量索引为空或过小（$chunks 条）"; need_bootstrap=1; else ok "向量索引 $chunks 条"; fi

if [ "$need_bootstrap" -eq 1 ]; then
  if [ "$BOOTSTRAP" -eq 1 ]; then
    say "运行完整数据管道（约 10 分钟）"
    for step in \
      "cs408rag.ingestion.download --source all" \
      "cs408rag.ingestion.quiz_loader" \
      "cs408rag.ingestion.xmind_loader" \
      "cs408rag.ingestion.tag_mapper" \
      "cs408rag.ingestion.image_index" \
      "cs408rag.ingestion.indexer" \
      "cs408rag.retrieval.graph_store" \
      "cs408rag.eval.build_eval_set"; do
      echo "    → python -m $step"
      uv run --no-sync python -m $step >/dev/null 2>&1 || die "步骤失败：$step"
    done
    ok "数据管道完成"
  else
    die "数据不完整。加 --bootstrap 自动构建，或按 README「快速开始」手动执行各步骤"
  fi
fi

# ---------------------------------------------------------------------------
# 4. 启动服务
# ---------------------------------------------------------------------------
mkdir -p logs
say "启动服务"

start_bg() {
  local name="$1" port="$2"; shift 2
  if netstat -ano 2>/dev/null | grep -q ":$port .*LISTENING"; then
    warn "$name 端口 $port 已被占用，跳过启动"
    return 0
  fi
  nohup "$@" > "logs/${name}.log" 2>&1 &
  echo $! > "logs/${name}.pid"
  ok "$name 启动中（日志 logs/${name}.log）"
}

if [ "$WITH_API" -eq 1 ]; then
  start_bg api "$API_PORT" uv run --no-sync uvicorn cs408rag.api.main:app --host 127.0.0.1 --port "$API_PORT"
fi

if [ "$FOREGROUND" -eq 1 ]; then
  say "前台运行 Streamlit（Ctrl+C 退出）"
  [ "$OPEN_BROWSER" -eq 1 ] && { sleep 3; powershell.exe -NoProfile -Command "Start-Process 'http://localhost:$UI_PORT'" >/dev/null 2>&1 || true; }
  exec uv run --no-sync streamlit run app/streamlit_app.py --server.port "$UI_PORT" --server.headless true --browser.gatherUsageStats false
fi

start_bg streamlit "$UI_PORT" uv run --no-sync streamlit run app/streamlit_app.py --server.port "$UI_PORT" --server.headless true --browser.gatherUsageStats false

# ---------------------------------------------------------------------------
# 5. 健康检查
# ---------------------------------------------------------------------------
say "等待服务就绪"
wait_healthy "Streamlit" "http://localhost:$UI_PORT/" 60
if [ "$WITH_API" -eq 1 ]; then
  for _ in $(seq 1 60); do
    curl -sf -m 3 "http://localhost:$API_PORT/v1/health" >/dev/null 2>&1 && { ok "FastAPI 就绪"; break; }
    sleep 2
  done
fi

# ---------------------------------------------------------------------------
# 6. 汇总
# ---------------------------------------------------------------------------
cat <<EOF

$(printf '\033[1;32m启动完成\033[0m')

  界面      http://localhost:$UI_PORT
EOF
[ "$WITH_API" -eq 1 ] && echo "  接口文档  http://localhost:$API_PORT/docs"
cat <<EOF
  Qdrant    http://localhost:6333/dashboard
  Neo4j     http://localhost:7474  （neo4j / cs408rag_dev_pw）

  日志      logs/
  停止      bash scripts/stop.sh
EOF

if [ "$OPEN_BROWSER" -eq 1 ]; then
  sleep 2
  powershell.exe -NoProfile -Command "Start-Process 'http://localhost:$UI_PORT'" >/dev/null 2>&1 || true
fi

# LLM key 提示
if grep -qE '^OPENAI_API_KEY=sk-your' .env 2>/dev/null; then
  printf '\n  \033[33m提示\033[0m .env 里的 OPENAI_API_KEY 还是占位符，问答页只会显示检索结果。\n'
fi
echo