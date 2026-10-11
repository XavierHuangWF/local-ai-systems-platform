# Local Large Language Model Serving with vLLM

## Objective

Run a quantized Large Language Model (LLM) on a local Graphics Processing Unit (GPU) and expose it through an OpenAI-compatible Application Programming Interface (API). The inference server is separate from the Python application so the application can send requests without loading model weights itself.

This document records the local inference configuration and the results observed during its initial verification. The commands below can be used to repeat those checks; they do not imply that the server is currently running.

## Environment

| Component | Verified configuration |
| --- | --- |
| Host operating system | Windows 11 |
| Linux environment | Windows Subsystem for Linux 2 (WSL2), Ubuntu |
| GPU | NVIDIA GeForce RTX 3080, 16 GB video memory (VRAM) |
| Python | 3.12 |
| PyTorch | `2.13.0+cu132` |
| Compute Unified Device Architecture (CUDA) toolkit | 13.2 |
| Model-serving software | vLLM |
| Model checkpoint | `Qwen/Qwen3-8B-AWQ` |
| API model identifier | `qwen3-8b` |

**Activation-aware Weight Quantization (AWQ)** reduces the memory used by the model's weights compared with an unquantized checkpoint. The initial vLLM startup log reported approximately **5.71 GiB of model weights** loaded onto the GPU. Additional GPU memory is needed for inference buffers, compiled execution artifacts, and the attention key/value (KV) cache.

## Architecture

```text
Interactive client / Agent Harness
              |
              v
      src/client/vllm_client.py
              |
              v
    OpenAI-compatible HTTP API
              |
              v
       vLLM inference server
              |
              v
         Qwen3-8B-AWQ
              |
              v
        PyTorch / CUDA
              |
              v
       NVIDIA RTX 3080
```

**Hypertext Transfer Protocol (HTTP)** is used between the application and the vLLM API server. The Agent Harness and its model client have since been implemented and verified separately; see [`agent-harness.md`](agent-harness.md).

## Start the Inference Server

Start an Ubuntu terminal in WSL2 and activate the existing Python virtual environment:

```bash
source ~/.venv/bin/activate
```

From the project root, use the checked-in startup script:

```bash
bash scripts/serve_vllm.sh
```

That script runs the equivalent configuration:

```bash
vllm serve Qwen/Qwen3-8B-AWQ \
  --served-model-name qwen3-8b \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key local-dev-key \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85
```

The process remains running in that terminal. The previous server process stops when the machine is restarted; the installed Python packages, CUDA toolkit, and cached model files do not ordinarily need to be reinstalled.

### Server configuration rationale

| Option | Purpose |
| --- | --- |
| `--served-model-name qwen3-8b` | Provides the model identifier used by application requests. |
| `--host 127.0.0.1` | Binds the server to the Linux loopback interface rather than all network interfaces. |
| `--port 8000` | Selects the server's HTTP port. |
| `--api-key local-dev-key` | Configures an API key for the development server. |
| `--max-model-len 8192` | Sets the maximum supported sequence length, including input and generated output. |
| `--gpu-memory-utilization 0.85` | Sets vLLM's target GPU-memory utilization for its allocations. |

The hard-coded API key is a **local-development placeholder**, not a production secret-management approach. The current configuration does not implement transport-layer encryption, access policy enforcement, or remote-service hardening.

## Verification Commands

Run the following in another terminal **after** vLLM reports that application startup is complete.

### Health check

```bash
curl http://127.0.0.1:8000/health
```

A successful response indicates that the health endpoint is reachable; it need not display a response body.

### List available models

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer local-dev-key"
```

The initial verification exposed the served model identifier `qwen3-8b`, backed by `Qwen/Qwen3-8B-AWQ`.

### Inference verification

The model was previously exercised through both an OpenAI-compatible chat-completions request and a Python client. The reusable application client is stored at:

```text
src/client/vllm_client.py
```

A subsequent real-model integration check is stored as:

```bash
python -m scripts.verify_agent_integration
```

That integration script requires the vLLM server to be running. Its results and scope are recorded in [`agent-harness.md`](agent-harness.md).

## Recorded Initial Inference Results

One earlier Python client request returned these usage and timing measurements:

```text
Prompt tokens:      35
Completion tokens: 172
Total tokens:      207
End-to-end latency: 4.84 seconds
```

Dividing the 172 completion tokens by the 4.84-second total gives approximately **35.5 completion tokens per second**, but this is an end-to-end ratio, **not** an isolated decoding-throughput benchmark. It also includes initial prompt processing, scheduling, and client overhead.

These figures describe the initial recorded test, not the performance of every request or the currently running state.

## Inference Mechanics

### Prefill

During **prefill**, the model processes the input sequence, computes attention states, and builds the initial KV cache.

### Decode

During **decode**, the model generates output tokens autoregressively. The KV cache allows previously computed attention key and value tensors to be reused instead of repeatedly processing all earlier tokens from scratch.

### KV-cache and context capacity

```text
Longer formatted conversations
              |
              v
More attention key/value states
              |
              v
More GPU memory per request
              |
              v
Potentially lower concurrency
```

The maximum model sequence length configured for this server is **8,192 tokens**. The Agent Harness separately reserves **512 tokens** for generated output and enforces a **7,680-token input budget**. Those application-level limits are documented in [`agent-harness.md`](agent-harness.md); they are not additional server flags in `serve_vllm.sh`.

## Observability

When the server is running, vLLM exposes Prometheus-compatible metrics at:

```text
http://127.0.0.1:8000/metrics
```

The original checks inspected metrics relating to generated-token counts, time to first token, KV-cache use, completed requests, and request errors. The initial requests completed without recorded request errors.

This verifies that the metrics endpoint was available during the original session; it does not mean that a persistent Prometheus monitoring service has been deployed.

## Scope and Limitations

- The server is a locally operated development service. Availability depends on its process remaining active.
- The first latency result is a single-request observation, not a statistically representative benchmark.
- Model output correctness requires separate evaluation; successful API calls only establish that inference requests complete.
- The current inference document covers text-based model serving, not hardened multi-user access or production operations.
- Skills are currently loaded and validated by a separate component but are not yet integrated into the Agent Harness. Tool integration, Model Context Protocol (MCP), additional security guardrails, and broader evaluation remain planned work.

For setup failures involving the Linux compiler and CUDA compiler (`nvcc`), see [`troubleshooting.md`](troubleshooting.md).
