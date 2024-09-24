#!/bin/bash

set -exv

IMAGE="quay.io/app-sre/tangerine-backend"
IMAGE_TAG=$(git rev-parse --short=7 HEAD)

if [[ -z "$QUAY_USER" || -z "$QUAY_TOKEN" ]]; then
    echo "QUAY_USER and QUAY_TOKEN must be set"
    exit 1
fi

# Create tmp dir to store data in during job run (do NOT store in $WORKSPACE)
export TMP_JOB_DIR=$(mktemp -d -p "$HOME" -t "build-${JOB_NAME}-${BUILD_NUMBER}-XXXXXX")
echo "job tmp dir location: $TMP_JOB_DIR"

function job_cleanup() {
    echo "cleaning up job tmp dir: $TMP_JOB_DIR"
    rm -fr $TMP_JOB_DIR
}

trap job_cleanup EXIT ERR SIGINT SIGTERM

# Set up podman cfg
AUTH_CONF_DIR="${TMP_JOB_DIR}/.podman"
mkdir -p $AUTH_CONF_DIR
export REGISTRY_AUTH_FILE="$AUTH_CONF_DIR/auth.json"


podman login -u="$QUAY_USER" -p="$QUAY_TOKEN" quay.io

# Build main image
podman build --pull=true -f Dockerfile -t "${IMAGE}:${IMAGE_TAG}" .

# push main image
podman push "${IMAGE}:${IMAGE_TAG}"
podman tag "${IMAGE}:${IMAGE_TAG}" "${IMAGE}:latest"
podman push "${IMAGE}:latest"
