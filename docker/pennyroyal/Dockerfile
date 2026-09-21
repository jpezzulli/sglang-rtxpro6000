# syntax=docker/dockerfile:1
# Fedora preserves the qualified GCC 15 paths; the host can run any compatible
# Linux distribution with Docker and the NVIDIA Container Toolkit.
FROM registry.fedoraproject.org/fedora:44 AS base
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
COPY docker/pennyroyal/cuda-fedora44.repo /etc/yum.repos.d/cuda-fedora44.repo
RUN dnf install -y --setopt=install_weak_deps=False --nodocs \
      ca-certificates curl-minimal git gcc15 gcc15-c++ make cmake ninja-build \
      meson pkgconf-pkg-config libaio-devel liburing-devel numactl libibverbs \
      libsndfile ffmpeg-free \
      openssl-devel protobuf-compiler rust cargo uv findutils shadow-utils \
      cuda-nvcc-13-3 cuda-libraries-devel-13-3 cuda-driver-devel-13-3 \
    && dnf clean all \
    && useradd --uid 1000 --create-home penny \
    && git config --system --add safe.directory /opt/pennyroyal
ENV CUDA_HOME=/usr/local/cuda-13.3 \
    CUDACXX=/usr/local/cuda-13.3/bin/nvcc \
    CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15 \
    NVCC_CCBIN=/usr/bin/g++-15 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    PATH=/opt/pennyroyal/.venv/bin:/usr/local/cuda-13.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64 \
    UV_PYTHON_INSTALL_DIR=/opt/python UV_NO_CACHE=1 UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PENNY_BUILD_JOBS=4 MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 \
    CARGO_BUILD_JOBS=4 FLASHINFER_NINJA_JOBS=4 FLASHINFER_NVCC_THREADS=1 \
    TORCHINDUCTOR_COMPILE_THREADS=4 \
    CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_INCREMENTAL=0
RUN uv python install 3.12.13 \
    && uv venv --python 3.12.13 /opt/pennyroyal/.venv \
    && uv pip install --python /opt/pennyroyal/.venv/bin/python \
      pip 'setuptools>=80.9' setuptools-rust setuptools-scm wheel build \
      meson-python pybind11 patchelf tomlkit pyyaml types-PyYAML

FROM base AS build
ARG SOURCE_REVISION
ARG SOURCE_REPOSITORY=https://github.com/jpezzulli/sglang-rtxpro6000.git
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
# Match the native installation's prebuilt SM120 kernels. Without this package
# FlashInfer compiles its large fused-MoE extension on every fresh cache root.
RUN uv pip install --python .venv/bin/python --no-deps \
      --index-url https://flashinfer.ai/whl/cu130 \
      'flashinfer-jit-cache==0.6.17+cu130'
# NIXL POSIX and optional NVMe PLE reader are built from pinned/vendored source.
RUN bash docker/pennyroyal/build-nixl.sh \
    && CARGO_TARGET_DIR=/tmp/penny-ple-cargo PYTHON=/opt/pennyroyal/.venv/bin/python \
      bash tools/ple_nvme/install.sh \
    && rm -rf /tmp/penny-ple-cargo \
    && .venv/bin/python -m pip check \
    && .venv/bin/python -m pip freeze > /opt/pennyroyal-container-packages.txt

FROM base AS runtime
ARG SOURCE_REVISION
LABEL org.opencontainers.image.title="Pennyroyal SGLang for RTX PRO 6000" \
      org.opencontainers.image.source="https://github.com/jpezzulli/sglang-rtxpro6000" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0"
COPY --from=build /opt/pennyroyal /opt/pennyroyal
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
