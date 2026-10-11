# Troubleshooting: Local vLLM Inference

## Scope

This document records two startup failures encountered while configuring local vLLM inference for `Qwen/Qwen3-8B-AWQ` on an NVIDIA GeForce RTX 3080 (16 GB) under Windows Subsystem for Linux 2 (WSL2) Ubuntu.

The failures occurred during runtime compilation. They were environment and toolchain issues, not evidence that the model checkpoint was invalid.

## 1. Missing Linux C/C++ compiler

### Observed error

During the initial vLLM startup, PyTorch's TorchInductor compilation failed:

```text
torch._inductor.exc.InductorError:
RuntimeError: Failed to find C compiler.
```

### Diagnosis

Model weights had loaded onto the Graphics Processing Unit (GPU), but the startup process could not complete because the Ubuntu environment lacked a host C/C++ compilation toolchain.

PyTorch's optimized execution path can compile generated code during initialization. Successful model loading does not guarantee that all required runtime compilation dependencies are installed.

### Resolution

Installed Ubuntu's build tools:

```bash
sudo apt update
sudo apt install -y build-essential
```

The compilers can be checked with:

```bash
gcc --version
g++ --version
```

### Observed outcome

The TorchInductor compilation stage completed successfully after the build toolchain was installed. Startup then progressed to a separate CUDA compiler issue described below.

## 2. Missing CUDA compiler (`nvcc`)

### Observed error

After the host compiler issue was resolved, vLLM progressed further but failed during FlashInfer initialization:

```text
RuntimeError:
Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

### Diagnosis

PyTorch could access the NVIDIA GPU through the available driver and CUDA (Compute Unified Device Architecture) runtime support. However, FlashInfer attempted to compile a CUDA sampling kernel at startup using just-in-time (JIT) compilation.

That compilation path also required the CUDA development toolkit, including `nvcc`. GPU access alone was insufficient.

The active Python environment reported:

```text
PyTorch: 2.13.0+cu132
CUDA build: 13.2
```

### Resolution

Installed CUDA Toolkit 13.2 **inside WSL2 Ubuntu** to provide the required CUDA compiler, consistent with the PyTorch CUDA build.

Verified the compiler with:

```bash
nvcc --version
```

The verification reported CUDA compilation tools release 13.2.

### Observed outcome

After the toolkit was installed and the environment configured, vLLM completed initialization and started its OpenAI-compatible Application Programming Interface (API) server.

## 3. Engineering explanation: execution versus compilation

A GPU-enabled Python environment may successfully execute existing CUDA kernels while remaining unable to compile additional kernels at runtime.

```text
Execute existing GPU kernels:
    NVIDIA driver + CUDA runtime
                 |
                 v
             GPU execution

Compile CUDA kernels at runtime:
    CUDA Toolkit + nvcc
                 |
                 v
          Generated GPU code
                 |
                 v
             GPU execution
```

The two failures involved different compiler dependencies:

| Startup stage | Missing dependency | Corrective action |
|---|---|---|
| PyTorch TorchInductor compilation | Linux host C/C++ toolchain | Install `build-essential` |
| FlashInfer CUDA JIT compilation | CUDA Toolkit / `nvcc` | Install CUDA Toolkit 13.2 in WSL2 |

These issues should be diagnosed separately. Installing a CUDA toolkit does not substitute for a Linux host compiler, and installing `gcc` does not provide `nvcc`.

## 4. Verification boundary

The observed successful vLLM startup demonstrates that these two missing-compiler issues were resolved in the recorded environment. It does not, by itself, establish model answer accuracy, throughput under load, or reliability across future dependency upgrades.

For application-level inference verification and measured request results, see [`local-llm-serving.md`](local-llm-serving.md). For Agent Harness behavior and integration tests, see [`agent-harness.md`](agent-harness.md).
