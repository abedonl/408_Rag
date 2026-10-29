#!/usr/bin/env bash
# 从国内镜像站拉取镜像并打上标准 tag。
#
# 背景：本机实测 Docker Hub (registry-1.docker.io) 不可达，
#       `docker compose up` 会直接失败在拉取阶段。
#
# 用法：bash scripts/pull_images.sh
#
# 脚本是幂等的：镜像已存在则跳过。

set -euo pipefail

# 按可达性排序，实测结果：
#   docker.m.daocloud.io  -> 401（正常，/v2/ 未认证即返回 401）  ✅ 首选
#   docker.1panel.live    -> 200                                ✅ 备用
#   hub.rat.dev           -> 302                                ✅ 备用
#   dockerproxy.cn / docker.nju.edu.cn / mirror.ccs.tencentyun.com -> 不可达
MIRRORS=(
  "docker.m.daocloud.io"
  "docker.1panel.live"
  "hub.rat.dev"
)

# 镜像列表："官方 tag|镜像站内的路径"
IMAGES=(
  "qdrant/qdrant:v1.20.0|qdrant/qdrant:v1.20.0"
  "neo4j:5.26-community|library/neo4j:5.26-community"
)

has_image() {
  docker image inspect "$1" >/dev/null 2>&1
}

pull_from_any() {
  local target="$1" remote_path="$2"
  for m in "${MIRRORS[@]}"; do
    local ref="${m}/${remote_path}"
    echo "  尝试 ${ref}"
    if docker pull "$ref"; then
      docker tag "$ref" "$target"
      echo "  -> 已打标为 ${target}"
      return 0
    fi
    echo "  失败，换下一个镜像源"
  done
  return 1
}

main() {
  echo "==> 检查 Docker 守护进程"
  if ! docker info >/dev/null 2>&1; then
    echo "错误：Docker 守护进程未运行。请先启动 Docker Desktop。" >&2
    exit 1
  fi

  local failed=0
  for entry in "${IMAGES[@]}"; do
    local target="${entry%%|*}"
    local remote_path="${entry##*|}"
    if has_image "$target"; then
      echo "==> ${target} 已存在，跳过"
      continue
    fi
    echo "==> 拉取 ${target}"
    if ! pull_from_any "$target" "$remote_path"; then
      echo "错误：所有镜像源均失败，无法获取 ${target}" >&2
      failed=1
    fi
  done

  echo
  if [ "$failed" -eq 0 ]; then
    echo "完成。现在可以执行：docker compose up -d"
  else
    echo "存在失败项，请检查网络或手动配置镜像源。" >&2
    exit 1
  fi
}

main "$@"