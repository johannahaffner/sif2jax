#!/bin/bash

# Configuration
CONTAINER_IMAGE="johannahaffner/pycutest:latest"
MOUNT_PATH="/workspace"  # Path inside container where your code will be mounted
LOCAL_PATH="/Users/jhaffner/Desktop/projects/benchmarks/eigen"

# Run tests in container
docker run --rm \
  -v ${LOCAL_PATH}:${MOUNT_PATH} \
  -w ${MOUNT_PATH} \
  ${CONTAINER_IMAGE} \
  bash -c "pip install -e . && pytest \"\$@\"" -- "$@"
