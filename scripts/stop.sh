#!/usr/bin/env bash
# 停止 408-RAG 应用（默认保留数据库容器与数据）。
#
#   bash scripts/stop.sh            只停 Streamlit / FastAPI
#   bash scripts/stop.sh --all      连 Qdrant / Neo4j 容器一起停（数据卷保留）
#   bash scripts/stop.sh --purge    连数据卷一起删（下次需重建索引，慎用）

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

STOP_ALL=0
PURGE=0
for arg in "$@"; do
  case "$arg" in
    --all) STOP_ALL=1 ;;
    --purge) STOP_ALL=1; PURGE=1 ;;
    -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数：$arg" >&2; exit 1 ;;
  esac
done

say() { printf '\033[1;36m==> %s\033[0m\n' "$*"; }
ok()  { printf '    \033[32m✓\033[0m %s\n' "$*"; }
warn(){ printf '    \033[33m!\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 应用进程：优先按 pid 文件停，pid 失效则按端口找
# ---------------------------------------------------------------------------
say "停止应用"
for name in streamlit api; do
  stopped=0
  pidfile="logs/${name}.pid"
  if [ -f "$pidfile" ]; then
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      stopped=1
    fi
    rm -f "$pidfile"
  fi

  # pid 文件不管用（比如手动启动过），退回按端口清扫
  port=0
  case "$name" in
    streamlit) port=8501 ;;
    api) port=8123 ;;
  esac
  if [ "$stopped" -eq 0 ] && [ "$port" -ne 0 ]; then
    for p in $(netstat -ano 2>/dev/null | grep ":$port " | grep LISTENING | awk '{print $5}' | sort -u); do
      taskkill //F //PID "$p" >/dev/null 2>&1 || kill -9 "$p" 2>/dev/null || true
      stopped=1
    done
  fi

  # 进程退出需要一点时间，等端口真正释放再下结论。
  # 否则会看到「已停止」但端口仍显示 LISTENING（TIME_WAIT / 正在退出）。
  if [ "$stopped" -eq 1 ] && [ "$port" -ne 0 ]; then
    for _ in $(seq 1 10); do
      netstat -ano 2>/dev/null | grep -q ":$port .*LISTENING" || break
      sleep 1
    done
  fi

  if [ "$stopped" -eq 1 ]; then
    if [ "$port" -ne 0 ] && netstat -ano 2>/dev/null | grep -q ":$port .*LISTENING"; then
      warn "${name} 已发送终止信号，但端口 $port 仍被占用"
    else
      ok "${name} 已停止"
    fi
  else
    warn "${name} 未在运行"
  fi
done

# ---------------------------------------------------------------------------
# 容器
# ---------------------------------------------------------------------------
if [ "$STOP_ALL" -eq 1 ]; then
  say "停止数据库容器"
  if [ "$PURGE" -eq 1 ]; then
    warn "--purge：将删除数据卷，下次启动需重建索引"
    docker compose down -v
    ok "容器与数据卷已删除"
  else
    docker compose down
    ok "容器已停止（数据卷保留）"
  fi
else
  say "数据库容器保持运行（要一并停止加 --all）"
fi

echo