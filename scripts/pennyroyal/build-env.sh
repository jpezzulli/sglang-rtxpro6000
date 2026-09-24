#!/usr/bin/env bash
# Sourceable build environment for the repeated export blocks in BUILD.md.
# Same variables, same defaults, no behaviour change: `source` it instead of
# pasting the block. This helper installs nothing and runs nothing; it only
# exports compiler and job-count variables with the existing defaults so the
# first source build and the recipes agree.
#
#   source scripts/pennyroyal/build-env.sh
#   export PENNY_BUILD_JOBS=8   # raise the job counts before sourcing to change them
: "${CUDA_HOME:=/usr/local/cuda}"
: "${CUDACXX:=$CUDA_HOME/bin/nvcc}"
: "${CC:=/usr/bin/gcc-15}"
: "${CXX:=/usr/bin/g++-15}"
: "${CUDAHOSTCXX:=$CXX}"
: "${TORCH_CUDA_ARCH_LIST:=12.0}"
: "${PENNY_BUILD_JOBS:=4}"
: "${MAX_JOBS:=$PENNY_BUILD_JOBS}"
: "${CMAKE_BUILD_PARALLEL_LEVEL:=$PENNY_BUILD_JOBS}"
: "${CARGO_BUILD_JOBS:=$PENNY_BUILD_JOBS}"
: "${FLASHINFER_NINJA_JOBS:=$PENNY_BUILD_JOBS}"
: "${FLASHINFER_NVCC_THREADS:=1}"
: "${TORCHINDUCTOR_COMPILE_THREADS:=$PENNY_BUILD_JOBS}"
export CUDA_HOME CUDACXX CC CXX CUDAHOSTCXX TORCH_CUDA_ARCH_LIST
export PENNY_BUILD_JOBS MAX_JOBS CMAKE_BUILD_PARALLEL_LEVEL CARGO_BUILD_JOBS
export FLASHINFER_NINJA_JOBS FLASHINFER_NVCC_THREADS TORCHINDUCTOR_COMPILE_THREADS
