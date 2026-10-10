# syntax=docker/dockerfile:1
# Fedora preserves the qualified GCC 15 paths; the host can run any compatible
# Linux distribution with Docker and the NVIDIA Container Toolkit.
FROM registry.fedoraproject.org/fedora:44 AS base
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
COPY docker/pennyroyal/cuda-fedora44.repo /etc/yum.repos.d/cuda-fedora44.repo
RUN dnf install -y --setopt=install_weak_deps=False --nodocs \
      ca-certificates curl-minimal git gcc15 gcc15-c++ make ninja-build \
      libaio-devel liburing-devel numactl libibverbs \
      libsndfile ffmpeg-free \
      cuda-nvcc-13-3 cuda-libraries-devel-13-3 cuda-driver-devel-13-3 \
      shadow-utils \
    && dnf clean all \
    && useradd --uid 1000 --create-home penny \
    && git config --system --add safe.directory /opt/pennyroyal
ENV CUDA_HOME=/usr/local/cuda-13.3 \
    CUDACXX=/usr/local/cuda-13.3/bin/nvcc \
    CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15 \
    NVCC_CCBIN=/usr/bin/g++-15 \
    TORCH_CUDA_ARCH_LIST="8.6 8.9 12.0+PTX" \
    PATH=/opt/pennyroyal/.venv/bin:/usr/local/cuda-13.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64 \
    UV_PYTHON_INSTALL_DIR=/opt/python UV_NO_CACHE=1 UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PENNY_BUILD_JOBS=4 MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 \
    CARGO_BUILD_JOBS=4 FLASHINFER_NINJA_JOBS=4 FLASHINFER_NVCC_THREADS=1 \
    TORCHINDUCTOR_COMPILE_THREADS=4 \
    CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_INCREMENTAL=0

FROM base AS build
ARG SOURCE_REVISION
ARG SOURCE_REPOSITORY=https://github.com/jpezzulli/sglang-rtxpro6000.git
# The base above carries only what the serving image runs on: the compilers,
# CUDA headers and Ninja that runtime JIT needs, and the NIXL/multimedia
# runtime libraries. These build-only tools are installed after the stage
# split, so they never propagate into the final image: CMake, Meson and
# pkg-config for the NIXL/PLE source builds, Rust and Cargo for the Rust
# wheels and reader, uv to create the venv below, the protobuf compiler,
# openssl development headers and findutils.
RUN dnf install -y --setopt=install_weak_deps=False --nodocs \
      cmake meson pkgconf-pkg-config openssl-devel protobuf-compiler \
      rust cargo uv findutils \
    && dnf clean all
RUN uv python install 3.12.13 \
    && uv venv --python 3.12.13 /opt/pennyroyal/.venv \
    && uv pip install --python /opt/pennyroyal/.venv/bin/python \
      pip 'setuptools>=80.9' setuptools-rust setuptools-scm wheel build \
      meson-python pybind11 patchelf tomlkit pyyaml types-PyYAML
# Clone exactly the reviewed commit. Keeping shallow Git metadata lets the
# existing namespace helper identify the source without inventing an identity.
RUN [[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] \
    && git -C /opt/pennyroyal init \
    && git -C /opt/pennyroyal remote add origin "$SOURCE_REPOSITORY" \
    && git -C /opt/pennyroyal fetch --depth=1 origin "$SOURCE_REVISION" \
    && git -C /opt/pennyroyal checkout --detach FETCH_HEAD
WORKDIR /opt/pennyroyal
RUN export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="0.5.19+g${SOURCE_REVISION:0:12}" \
    && CARGO_TARGET_DIR=/tmp/penny-cargo uv pip install \
      --python .venv/bin/python --prerelease=allow --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      --extra-index-url https://docs.sglang.ai/whl/cu130/ \
      --constraint docker/pennyroyal/constraints.txt --no-build-isolation-package sglang -e python \
    && rm -rf /tmp/penny-cargo
# Match the native installation's prebuilt SM120 kernels. Without these packages
# FlashInfer compiles its large fused-MoE extension on every fresh cache root.
# Since 0.7.0 flashinfer-jit-cache is a shim; --no-deps must pull the SM120-family
# provider wheel explicitly (and both stay aligned with pyproject's flashinfer pin).
RUN uv pip install --python .venv/bin/python --no-deps \
      --index-url https://flashinfer.ai/whl/cu130 \
      'flashinfer-jit-cache==0.7.0.post1+cu130' \
      'flashinfer-jit-cache-sm120f==0.7.0.post1+cu130'
# Package the accepted FlashInfer SM120 source into the image's own FlashInfer
# package and compile the fused-MoE module from those patched headers, exactly
# as the native sequences do. The step hands nvcc the CUDAHOSTCXX above (g++-15)
# as FlashInfer's -ccbin, which reads that setting from CC and would otherwise
# override NVCC_CCBIN with the C driver; the image now builds the C++ and the CUDA
# host side with the same GCC 15. The stock provider wheels stay for the other
# kernels; FlashInfer prefers the package-local module below. Building here is
# what lets a downloaded image run the Next profiles with no rebuild, no
# configurator and no first-launch compilation; the JIT objects stay in the
# build stage. The step also guards one stock include of <sanitizer/...> that the
# gcc15 package above does not ship, which is why the compile needs no sanitizer
# development headers here either. When the optional flashinfer-cubin wheel is
# installed here, the same step removes its trtllm-gen cubins that the pinned
# runners dispatch only on SM100/103/107: no RTX GPU can load them, and pruning
# at this layer, before the COPY below, is what keeps their bytes out of the
# published image instead of merely deleting them in a later one.
# The hosted build action exited 137 during this compile with four jobs.
# Serialize this memory-heavy extension on the small runner. Local builders
# can raise the build-only argument; the serving/JIT defaults stay unchanged.
ARG FLASHINFER_BUILD_JOBS=1
RUN MAX_JOBS="$FLASHINFER_BUILD_JOBS" FLASHINFER_NINJA_JOBS="$FLASHINFER_BUILD_JOBS" \
    FLASHINFER_WORKSPACE_BASE=/tmp/penny-fi-sm120 \
    .venv/bin/python scripts/pennyroyal/flashinfer/install.py \
    && .venv/bin/python scripts/pennyroyal/flashinfer/install.py --check >/dev/null \
    && rm -rf /tmp/penny-fi-sm120
# NIXL POSIX and optional NVMe PLE reader are built from pinned/vendored source.
COPY scripts/pennyroyal/flashinfer/check_dependencies.py /tmp/penny-check-dependencies.py
RUN bash docker/pennyroyal/build-nixl.sh \
    && CARGO_TARGET_DIR=/tmp/penny-ple-cargo PYTHON=/opt/pennyroyal/.venv/bin/python \
      bash tools/ple_nvme/install.sh \
    && rm -rf /tmp/penny-ple-cargo \
    && .venv/bin/python /tmp/penny-check-dependencies.py \
    && rm /tmp/penny-check-dependencies.py \
    && .venv/bin/python -m pip freeze > /opt/pennyroyal-container-packages.txt

FROM base AS runtime
ARG SOURCE_REVISION
LABEL org.opencontainers.image.title="Pennyroyal SGLang for RTX PRO 6000" \
      org.opencontainers.image.source="https://github.com/jpezzulli/sglang-rtxpro6000" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0"
COPY --from=build /opt/pennyroyal /opt/pennyroyal
# The venv is a symlink farm onto the uv-managed interpreter; it lives in the
# build stage because only uv there writes it, so the interpreter itself is the
# one runtime artifact copied out of that stage.
COPY --from=build /opt/python /opt/python
COPY --from=build /opt/pennyroyal-container-packages.txt /opt/pennyroyal-container-packages.txt
RUN install -m 0755 /opt/pennyroyal/docker/pennyroyal/entrypoint.sh /usr/local/bin/pennyroyal \
    && mkdir -p /cache /nixl \
    && chown penny:penny /cache /nixl
ENV CACHE_BASE=/cache NIXL_STORAGE_BASE=/nixl \
    HF_HOME=/cache/huggingface XDG_CACHE_HOME=/cache \
    SGLANG_MM_PREPROCESS_DEVICE=cpu \
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false
WORKDIR /opt/pennyroyal
USER penny
RUN PENNY_EXPECTED_SOURCE_REVISION="$SOURCE_REVISION" pennyroyal --check \
    && pennyroyal --help
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=20m --retries=3 \
  CMD ["/opt/pennyroyal/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3).close()"]
ENTRYPOINT ["/usr/local/bin/pennyroyal"]
CMD ["next"]
