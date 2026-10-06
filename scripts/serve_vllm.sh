#!/usr/bin/env bash

set -euo pipefail

vllm serve Qwen/Qwen3-8B-AWQ \
  --served-model-name qwen3-8b \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key local-dev-key \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85