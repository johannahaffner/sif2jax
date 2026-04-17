#!/bin/bash

# Configuration
CONTAINER_IMAGE="johannahaffner/pycutest:latest"
MOUNT_PATH="/workspace"
LOCAL_PATH="$(cd "$(dirname "$0")" && pwd)"

# EAGER_CONSTANT_FOLDING improves AD performance for problems with
# constant sub-expressions. Requires JAX >= 0.9.2 (see docker/Dockerfile).
docker run --rm \
  -e EAGER_CONSTANT_FOLDING=TRUE \
  -v ${LOCAL_PATH}:${MOUNT_PATH} \
  -w ${MOUNT_PATH} \
  ${CONTAINER_IMAGE} \
  bash -c "pip install -e . && pytest \"\$@\"" -- "$@"
