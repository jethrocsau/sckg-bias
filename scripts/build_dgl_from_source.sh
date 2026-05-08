#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path-to-dgl-source> [cuda|cpu]" >&2
  echo "Example: $0 ~/src/dgl cuda" >&2
  exit 1
fi

DGL_SRC="$1"
BUILD_MODE="${2:-cuda}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

if [[ ! -d "$DGL_SRC" ]]; then
  echo "DGL source directory not found: $DGL_SRC" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is not available in the current environment" >&2
  exit 1
fi

for cmd in git cmake; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required build command missing: $cmd" >&2
    exit 1
  fi
done

if [[ "$BUILD_MODE" != "cpu" && "$BUILD_MODE" != "cuda" ]]; then
  echo "Invalid build mode: $BUILD_MODE" >&2
  echo "Expected 'cuda' or 'cpu'." >&2
  exit 1
fi

if [[ "$BUILD_MODE" == "cuda" ]] && ! command -v nvcc >/dev/null 2>&1; then
  echo "CUDA build requested, but nvcc was not found in PATH." >&2
  echo "Activate a CUDA-enabled environment or set CUDA_HOME/CUDACXX correctly." >&2
  exit 1
fi

resolve_cuda_root() {
  local nvcc_path="${1:-}"
  if [[ -z "$nvcc_path" ]]; then
    return 1
  fi
  dirname "$(dirname "$(readlink -f "$nvcc_path")")"
}

find_cuda_include_dir() {
  local root="$1"
  if [[ -f "$root/include/cuda_runtime.h" ]]; then
    printf '%s\n' "$root/include"
    return 0
  fi

  local candidate
  for candidate in "$root"/targets/*/include; do
    if [[ -f "$candidate/cuda_runtime.h" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

find_cuda_lib_dir() {
  local root="$1"
  if [[ -f "$root/lib64/libcudart.so" ]]; then
    printf '%s\n' "$root/lib64"
    return 0
  fi
  if [[ -f "$root/lib/libcudart.so" ]]; then
    printf '%s\n' "$root/lib"
    return 0
  fi

  local candidate
  for candidate in "$root"/targets/*/lib64 "$root"/targets/*/lib; do
    if [[ -f "$candidate/libcudart.so" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

CUDA_CMAKE_ARGS=""
if [[ "$BUILD_MODE" == "cuda" ]]; then
  CUDA_NVCC_PATH="${CUDACXX:-$(command -v nvcc)}"
  CUDA_ROOT_CANDIDATE="${CUDA_HOME:-$(resolve_cuda_root "$CUDA_NVCC_PATH")}" 
  CUDA_INCLUDE_DIR=""
  CUDA_LIB_DIR=""

  if [[ -n "$CUDA_ROOT_CANDIDATE" ]]; then
    CUDA_INCLUDE_DIR="$(find_cuda_include_dir "$CUDA_ROOT_CANDIDATE" || true)"
    CUDA_LIB_DIR="$(find_cuda_lib_dir "$CUDA_ROOT_CANDIDATE" || true)"
  fi

  if [[ -z "$CUDA_INCLUDE_DIR" || -z "$CUDA_LIB_DIR" ]] && [[ -d /usr/local/cuda ]]; then
    FALLBACK_INCLUDE_DIR="$(find_cuda_include_dir /usr/local/cuda || true)"
    FALLBACK_LIB_DIR="$(find_cuda_lib_dir /usr/local/cuda || true)"
    if [[ -n "$FALLBACK_INCLUDE_DIR" && -n "$FALLBACK_LIB_DIR" ]]; then
      if [[ -n "$CUDA_ROOT_CANDIDATE" ]]; then
        echo "[build] Active CUDA toolkit at $CUDA_ROOT_CANDIDATE is incomplete for CMake detection; falling back to /usr/local/cuda"
      fi
      CUDA_ROOT_CANDIDATE="/usr/local/cuda"
      CUDA_NVCC_PATH="/usr/local/cuda/bin/nvcc"
      CUDA_INCLUDE_DIR="$FALLBACK_INCLUDE_DIR"
      CUDA_LIB_DIR="$FALLBACK_LIB_DIR"
    fi
  fi

  if [[ -z "$CUDA_ROOT_CANDIDATE" || -z "$CUDA_INCLUDE_DIR" || -z "$CUDA_LIB_DIR" ]]; then
    echo "Unable to resolve a complete CUDA toolkit with both headers and libcudart.so." >&2
    echo "Set CUDA_HOME/CUDACXX to a full CUDA install, or use cpu build mode." >&2
    exit 1
  fi

  export CUDA_HOME="$CUDA_ROOT_CANDIDATE"
  export CUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT_CANDIDATE"
  export CUDACXX="$CUDA_NVCC_PATH"
  export PATH="$CUDA_HOME/bin:$PATH"

  CUDA_CMAKE_ARGS+=" -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME"
  CUDA_CMAKE_ARGS+=" -DCUDA_NVCC_EXECUTABLE=$CUDACXX"
  CUDA_CMAKE_ARGS+=" -DCUDA_TOOLKIT_INCLUDE=$CUDA_INCLUDE_DIR"
  CUDA_CMAKE_ARGS+=" -DCUDA_INCLUDE_DIRS=$CUDA_INCLUDE_DIR"
  CUDA_CMAKE_ARGS+=" -DCUDA_CUDART_LIBRARY=$CUDA_LIB_DIR/libcudart.so"

  if [[ -f "$CUDA_LIB_DIR/libcublas.so" ]]; then
    CUDA_CMAKE_ARGS+=" -DCUDA_cublas_LIBRARY=$CUDA_LIB_DIR/libcublas.so"
  fi
  if [[ -f "$CUDA_LIB_DIR/libcusparse.so" ]]; then
    CUDA_CMAKE_ARGS+=" -DCUDA_cusparse_LIBRARY=$CUDA_LIB_DIR/libcusparse.so"
  fi
  if [[ -f "$CUDA_HOME/lib64/stubs/libcuda.so" ]]; then
    CUDA_CMAKE_ARGS+=" -DCUDA_CUDA_LIBRARY=$CUDA_HOME/lib64/stubs/libcuda.so"
  elif [[ -f /usr/lib/aarch64-linux-gnu/libcuda.so ]]; then
    CUDA_CMAKE_ARGS+=" -DCUDA_CUDA_LIBRARY=/usr/lib/aarch64-linux-gnu/libcuda.so"
  fi
fi

export DGLBACKEND=pytorch
export USE_OPENMP=ON
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
export DGL_HOME="$DGL_SRC"

pushd "$DGL_SRC" >/dev/null

git submodule update --init --recursive

python -m pip install --upgrade pip setuptools wheel

if [[ "$BUILD_MODE" == "cpu" ]]; then
  echo "[build] Building DGL (CPU) with $CMAKE_BUILD_PARALLEL_LEVEL parallel jobs"
  bash script/build_dgl.sh -c
else
  echo "[build] Building DGL (CUDA) with $CMAKE_BUILD_PARALLEL_LEVEL parallel jobs"
  if [[ -n "${CUDA_HOME:-}" ]]; then
    echo "[build] CUDA_HOME=$CUDA_HOME"
  fi
  if [[ -n "${CUDACXX:-}" ]]; then
    echo "[build] CUDACXX=$CUDACXX"
  fi
  if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    echo "[build] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
  fi
  bash script/build_dgl.sh -g -e "$CUDA_CMAKE_ARGS"
fi

pushd python >/dev/null
python -m pip install -v .
popd >/dev/null

python - <<'PY'
import dgl
import torch
print('dgl', dgl.__version__)
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda_device_count', torch.cuda.device_count())
PY

popd >/dev/null
