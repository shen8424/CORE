#!/bin/bash
# 通用推理：运行时顺序融合 LoRA 到临时目录 → swift infer vLLM → 自动清理临时模型
# Usage: bash code/Inference/run_inference.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
echo "[INFO] 运行通用 vLLM 推理（临时融合模型，结束自动清理）..."
python3 "$SCRIPT_DIR/infer.py" --config "$SCRIPT_DIR/infer.yaml"
