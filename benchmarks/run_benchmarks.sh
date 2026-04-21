#!/bin/bash

# Configuration
CONTAINER_IMAGE="johannahaffner/pycutest:latest"
MOUNT_PATH="/workspace"
LOCAL_PATH="$(cd "$(dirname "$0")/.." && pwd)"

# Build env flags — skip EAGER_CONSTANT_FOLDING if --no-flags is passed
ENV_FLAGS="-e EAGER_CONSTANT_FOLDING=TRUE"
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--no-flags" ]; then
        ENV_FLAGS=""
    else
        ARGS+=("$arg")
    fi
done

# Run benchmarks in container
docker run --rm \
  $ENV_FLAGS \
  -v ${LOCAL_PATH}:${MOUNT_PATH} \
  -w ${MOUNT_PATH} \
  ${CONTAINER_IMAGE} \
  bash -c "pip install -e . && pytest --benchmark-only \"\$@\"" -- "${ARGS[@]}"
