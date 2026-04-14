#!/bin/bash

# Configuration
CONTAINER_IMAGE="johannahaffner/pycutest:latest"
MOUNT_PATH="/workspace"  # Path inside container where your code will be mounted
LOCAL_PATH="$(cd "$(dirname "$0")/.." && pwd)"

# Run benchmarks in container
docker run --rm \
  -e EAGER_CONSTANT_FOLDING=TRUE \
  -e JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=TRUE \
  -v ${LOCAL_PATH}:${MOUNT_PATH} \
  -w ${MOUNT_PATH} \
  ${CONTAINER_IMAGE} \
  bash -c "pip install -e . && pytest benchmarks/ --benchmark-only $*"