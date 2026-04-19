#!/usr/bin/env bash
# One-shot setup for the bai project.
#
# Does:
#   1. Fetch all git submodules (BALROG, MARAProtocol, Autumn.cpp).
#   2. Install Python deps via `uv sync` (maraprotocol + balrog editable).
#   3. Build the Autumn.cpp Python interpreter module.
#   4. Generate MARAProtocol protobuf Python stubs.
#   5. Download the AutumnBench example dataset.
#
# Prereqs on the host: git, curl, jq, cmake, make, a C++ compiler, and uv.

set -euo pipefail

BAI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BAI_ROOT}"

echo "==> [1/5] Initializing git submodules"
git submodule update --init --recursive

echo "==> [2/5] Installing Python deps (uv sync)"
uv sync

echo "==> [3/5] Building Autumn.cpp Python interpreter module"
PYBIND11_CMAKE_DIR="$(uv run python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
mkdir -p Autumn.cpp/build
(
  cd Autumn.cpp/build
  cmake .. -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}"
  make -j"$(nproc 2>/dev/null || echo 4)"
)
# Drop any shipped platform-specific .so and install the freshly built one
find MARAProtocol/python_examples/autumnbench -maxdepth 1 -name 'interpreter_module.*.so' -delete
cp Autumn.cpp/build/interpreter_module*.so MARAProtocol/python_examples/autumnbench/

echo "==> [4/5] Generating MARAProtocol protobuf stubs"
(
  cd MARAProtocol
  mkdir -p generated/mara
  touch generated/__init__.py generated/mara/__init__.py
  for proto in mara_environment mara_environment_service mara_registry \
               mara_agent mara_evaluation mara_evaluation_controller; do
    uv run --project "${BAI_ROOT}" python -m grpc_tools.protoc \
      --proto_path=./protocols \
      --python_out=./generated/mara \
      --grpc_python_out=./generated/mara \
      "./protocols/${proto}.proto"
  done
  # Rewrite generated imports to the nested `generated.mara.*` path
  find ./generated/mara -name '*.py' -type f -exec sed -i \
    -e 's/^from mara_/from generated.mara.mara_/g' \
    -e 's/^import mara_/import generated.mara.mara_/g' {} +
)

echo "==> [5/5] Downloading AutumnBench example dataset"
# The download script writes to a relative path (python_examples/autumnbench/
# example_benchmark), so run it from inside MARAProtocol or it creates a
# top-level ./python_examples that shadows the editable package.
(cd MARAProtocol && bash scripts/download_dataset.sh)

echo
echo "Setup complete. Try:"
echo "  uv run stepwise_eb_learn.py envs.names=autumn tasks.autumn_tasks=[ice]"
