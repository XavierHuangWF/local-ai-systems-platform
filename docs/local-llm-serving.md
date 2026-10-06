# Local LLM Serving with vLLM

## Environment

- Windows 11
- WSL2 Ubuntu
- Python 3.12
- PyTorch 2.13.0+cu132
- CUDA Toolkit 13.2
- vLLM
- NVIDIA GeForce RTX 3080 16 GB
- Model: Qwen/Qwen3-8B-AWQ

## Start the vLLM Server

Activate the WSL Python environment:

```bash
source ~/.venv/bin/activate
```

Start the local inference server:

```bash
vllm serve Qwen/Qwen3-8B-AWQ \
  --served-model-name qwen3-8b \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key local-dev-key \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85
```

## Server Configuration

### `--served-model-name qwen3-8b`

Defines the model name exposed through the API.

### `--host 127.0.0.1`

Restricts the inference service to the local machine.

### `--port 8000`

Runs the HTTP API on port 8000.

### `--api-key local-dev-key`

Requires an API key when accessing the model endpoint.

### `--max-model-len 8192`

Limits the maximum sequence length to 8192 tokens.

Longer context lengths require more KV-cache memory.

### `--gpu-memory-utilization 0.85`

Allows vLLM to use approximately 85% of available GPU memory while
leaving additional VRAM for CUDA and runtime allocations.

## Verification Commands

After the vLLM server is running, its health can be checked with:

```bash
curl http://127.0.0.1:8000/health
```

The served models can be listed with:

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer local-dev-key"
```

The configured model is exposed as:

```text
qwen3-8b
```

## Initial Inference Test

The local model was successfully accessed through the
OpenAI-compatible API.

One Python client test produced:

```text
Prompt tokens: 35
Completion tokens: 172
Total tokens: 207
Latency: 4.84 seconds
```

The inference path is:

```text
Python Client
    ↓
OpenAI-compatible HTTP API
    ↓
vLLM
    ↓
Qwen3-8B-AWQ
    ↓
PyTorch / CUDA
    ↓
RTX 3080
```

## Model and GPU Memory

The selected model is:

```text
Qwen/Qwen3-8B-AWQ
```

AWQ quantization reduces the GPU memory required for model weights.

During vLLM initialization, the model used approximately 5.7 GiB of
GPU memory, leaving additional VRAM for:

- KV cache
- CUDA graphs
- runtime buffers
- model execution

## KV Cache

The KV cache stores the attention key and value tensors for previously
processed tokens.

During autoregressive decoding, these cached tensors are reused instead
of recomputing the complete previous sequence for every generated token.

Conceptually:

```text
Longer context
    ↓
Larger KV cache
    ↓
More VRAM usage
    ↓
Lower possible concurrency
```

## Prefill and Decode

### Prefill

The model processes the input prompt and creates the attention state
needed for subsequent generation.

### Decode

The model generates output tokens autoregressively, one token at a time,
while reusing the KV cache.

## Observability

vLLM exposes Prometheus-compatible metrics at:

```text
http://127.0.0.1:8000/metrics
```

Metrics inspected included:

- time to first token
- generated token count
- KV-cache utilization
- successful requests
- request errors

The initial inference tests completed successfully without request errors.

## Architecture

```text
Application / Client
        ↓
OpenAI-compatible API
        ↓
vLLM Inference Server
        ↓
Qwen3-8B-AWQ
        ↓
PyTorch / CUDA
        ↓
NVIDIA RTX 3080
```

This inference service will later act as the model backend for the
agent harness, tools, MCP integration, guardrails, and evaluation
components.