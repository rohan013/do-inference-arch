#!/usr/bin/env bash
# Build and push the request router image to DO Container Registry.

set -euo pipefail

source "$(dirname "$0")/lib/common.sh"
load_config
ensure_doctl_auth
require_cmd docker

if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run: would ensure registry ${DO_REGISTRY_NAME} exists"
elif ! doctl registry get "${DO_REGISTRY_NAME}" >/dev/null 2>&1; then
  log "creating container registry ${DO_REGISTRY_NAME}..."
  run doctl registry create "${DO_REGISTRY_NAME}"
fi

log "logging in to DO registry..."
if [[ "${DRY_RUN}" != "1" ]]; then
  doctl registry login
fi

log "building router image for linux/amd64..."
# Mac builders: DOCKER_BUILDKIT=0 avoids occasional cross-platform BuildKit issues.
run env DOCKER_BUILDKIT=0 docker build --platform linux/amd64 -t "${ROUTER_IMAGE}" "${REPO_ROOT}/router"

log "pushing ${ROUTER_IMAGE}..."
run docker push "${ROUTER_IMAGE}"

log "router image published"
