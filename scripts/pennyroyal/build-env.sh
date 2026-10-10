#!/usr/bin/env bash
# Sourceable build environment for the repeated export blocks in BUILD.md.
# Same variables, same defaults, no behaviour change: `source` it instead of
# pasting the block. This helper installs nothing and runs nothing; it only
# exports compiler and job-count variables with the existing defaults so the
# first source build and the recipes agree.
#
#   source scripts/pennyroyal/build-env.sh
#   export PENNY_BUILD_JOBS=8   # raise the job counts before sourcing to change them
#
# TORCH_CUDA_ARCH_LIST covers the generic builds of these instructions for the
# three supported RTX families (SM86 Ampere, SM89 Ada, SM120 Blackwell), with
# PTX kept so JIT stays available on newer chips. The accepted SM120 fused-MoE
# module is not affected: the FlashInfer packaging step pins its own build to
# 12.0f through FLASHINFER_CUDA_ARCH_LIST.
: "${CUDA_HOME:=/usr/local/cuda}"
: "${CUDACXX:=$CUDA_HOME/bin/nvcc}"
: "${CC:=/usr/bin/gcc-15}"
: "${CXX:=/usr/bin/g++-15}"
: "${CUDAHOSTCXX:=$CXX}"
: "${TORCH_CUDA_ARCH_LIST:=8.6 8.9 12.0+PTX}"
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
