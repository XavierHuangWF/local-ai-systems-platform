# Troubleshooting

## Missing C Compiler

### Error

During the initial vLLM startup:

```text
torch._inductor.exc.InductorError:
RuntimeError: Failed to find C compiler.
```

### Cause

The model successfully loaded onto the GPU, but PyTorch TorchInductor
needed a Linux C/C++ compiler while compiling optimized execution code.

The WSL environment did not initially contain the required compiler
toolchain.

### Fix

Installed the Ubuntu build tools:

```bash
sudo apt update
sudo apt install -y build-essential
```

Verified the compilers:

```bash
gcc --version
g++ --version
```

After installation, the PyTorch compilation stage completed
successfully.

---

## Missing CUDA Compiler (`nvcc`)

### Error

After resolving the C compiler issue, vLLM progressed further but
failed during FlashInfer initialization:

```text
RuntimeError:
Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

### Cause

CUDA execution was already working through PyTorch and the NVIDIA
driver.

However, FlashInfer attempted to JIT-compile a CUDA sampling kernel,
which required the CUDA development toolkit and the `nvcc` compiler.

The active PyTorch environment reported:

```text
PyTorch: 2.13.0+cu132
CUDA build: 13.2
```

### Fix

Installed CUDA Toolkit 13.2 inside WSL to match the PyTorch CUDA build.

The CUDA compiler was then verified with:

```bash
nvcc --version
```

After installing the toolkit, vLLM completed initialization and the
OpenAI-compatible API server started successfully.

## CUDA Runtime vs CUDA Toolkit

A working CUDA runtime does not necessarily mean that CUDA source code
can be compiled.

```text
NVIDIA Driver
    ↓
CUDA Runtime
    ↓
Execute existing CUDA kernels
```

JIT compilation additionally requires:

```text
CUDA Toolkit
    ↓
nvcc
    ↓
Compile CUDA kernels
```

This explains why PyTorch could already access the RTX 3080 while
FlashInfer still failed until `nvcc` was installed.