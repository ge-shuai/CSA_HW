#!/usr/bin/env bash
# 全流水线一键复现（可选）。默认会跑 A–H 全部单元，耗时可能数小时（E1/E4 最贵）。
# 用法：
#   bash run_all.sh                 # 全跑
#   PY=/d/anaconda3/python.exe bash run_all.sh
#   bash run_all.sh --only A,B,E5   # 只跑匹配的单元（按目录名/脚本名子串）
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"
export PYTHONHASHSEED=0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

ONLY=""
if [ "${1:-}" = "--only" ]; then
  ONLY="${2:-}"
fi

run_one() {
  local s="$1"
  if [ -n "$ONLY" ]; then
    local hit=0
    IFS=',' read -ra KEYS <<< "$ONLY"
    for k in "${KEYS[@]}"; do
      [ -z "$k" ] && continue
      case "$s" in *"$k"*) hit=1 ;; esac
    done
    [ "$hit" = "1" ] || return 0
  fi
  echo "-- $s"
  "$PY" "$s"
}

echo "== 0. 锚点（不过就停）=="
"$PY" 00_build_data.py

echo "== 1. 分析单元 A–G（顺序无关，但都依赖 data/）=="
for d in out/A* out/B* out/C* out/D* out/E* out/F* out/G*; do
  [ -d "$d" ] || continue
  for s in "$d"/run_*.py; do
    [ -e "$s" ] || continue
    run_one "$s"
  done
done

echo "== 2. 图表 H（只读 A–G 的 CSV）=="
for d in out/H*; do
  [ -d "$d" ] || continue
  for s in "$d"/run_*.py; do
    [ -e "$s" ] || continue
    run_one "$s"
  done
done

echo "== 3. 独立校验 =="
"$PY" ../verify/check_claims.py

echo "== DONE =="
