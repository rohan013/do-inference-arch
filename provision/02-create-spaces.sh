#!/usr/bin/env bash
# Create DO Spaces bucket for model weight storage (S3-compatible).

set -euo pipefail

source "$(dirname "$0")/lib/common.sh"
load_config
ensure_doctl_auth

if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run: would create spaces bucket ${SPACES_BUCKET} if missing"
elif doctl spaces list --format Name --no-header 2>/dev/null | grep -qx "${SPACES_BUCKET}"; then
  log "spaces bucket ${SPACES_BUCKET} already exists — skipping create"
else
  log "creating spaces bucket ${SPACES_BUCKET} in ${SPACES_REGION}..."
  run doctl spaces create "${SPACES_BUCKET}" --region "${SPACES_REGION}"
fi

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
