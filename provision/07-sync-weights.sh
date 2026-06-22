#!/usr/bin/env bash
# Upload model weights to DO Spaces (S3-compatible).

set -euo pipefail

source "$(dirname "$0")/lib/common.sh"
load_config
ensure_doctl_auth

require_cmd aws

: "${SPACES_ACCESS_KEY_ID:?Set SPACES_ACCESS_KEY_ID (Cloud Panel → API → Spaces Keys)}"
: "${SPACES_SECRET_ACCESS_KEY:?Set SPACES_SECRET_ACCESS_KEY}"

LOCAL_WEIGHTS_DIR="${1:-./llama-3.1-8b}"
ENDPOINT="https://${SPACES_REGION}.digitaloceanspaces.com"
REMOTE_URI="s3://${SPACES_BUCKET}/${MODEL_PREFIX}/"

export AWS_ACCESS_KEY_ID="${SPACES_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${SPACES_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION="${SPACES_REGION}"

log "creating bucket ${SPACES_BUCKET} (if missing)..."
run aws s3 mb "s3://${SPACES_BUCKET}" --endpoint-url "${ENDPOINT}" 2>/dev/null || true

[[ -d "${LOCAL_WEIGHTS_DIR}" ]] || die "local weights directory not found: ${LOCAL_WEIGHTS_DIR}"

log "syncing ${LOCAL_WEIGHTS_DIR} → ${REMOTE_URI}"
run aws s3 sync "${LOCAL_WEIGHTS_DIR}/" "${REMOTE_URI}" \
  --endpoint-url "${ENDPOINT}"

log "weights uploaded — update vLLM --model to local cache path for faster cold start"
log "  s3://${SPACES_BUCKET}/${MODEL_PREFIX}/"
