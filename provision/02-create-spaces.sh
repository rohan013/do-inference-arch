#!/usr/bin/env bash
# Create DO Spaces bucket for model weight storage (S3-compatible).

set -euo pipefail

source "$(dirname "$0")/lib/common.sh"
load_config
ensure_doctl_auth

log "Spaces bucket target: ${SPACES_BUCKET} (${SPACES_REGION})"
log "Create via Cloud Panel → Spaces, or:"
log "  aws s3 mb s3://${SPACES_BUCKET} --endpoint-url https://${SPACES_REGION}.digitaloceanspaces.com"

cat <<EOF
Upload quantized weights (example):

  s3cmd sync ./weights/ s3://${SPACES_BUCKET}/${MODEL_PREFIX}/ \\
    --host=${SPACES_REGION}.digitaloceanspaces.com \\
    --host-bucket="%(bucket)s.${SPACES_REGION}.digitaloceanspaces.com"

Or with aws-cli:

  aws s3 sync ./weights/ s3://${SPACES_BUCKET}/${MODEL_PREFIX}/ \\
    --endpoint-url https://${SPACES_REGION}.digitaloceanspaces.com

Update k8s ConfigMaps MODEL_WEIGHTS_URI to:
  s3://${SPACES_BUCKET}/${MODEL_PREFIX}/
EOF
