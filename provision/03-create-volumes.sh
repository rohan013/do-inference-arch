#!/usr/bin/env bash
# Create Block Storage volumes for FP8 checkpoint cache on GPU nodes.

set -euo pipefail

source "$(dirname "$0")/lib/common.sh"
load_config
ensure_doctl_auth

create_volume_if_missing() {
  local name="$1"
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry-run: would create volume ${name} if missing"
    return 0
  fi
  if doctl compute volume list --format Name --no-header | grep -qx "${name}"; then
    log "volume ${name} already exists — skipping"
    return 0
  fi
  log "creating volume ${name} (${VOLUME_SIZE_GB}GiB) in ${CLUSTER_REGION}..."
  run doctl compute volume create "${name}" \
    --region "${CLUSTER_REGION}" \
    --size "${VOLUME_SIZE_GB}GiB" \
    --desc "FP8 checkpoint cache for inference GPU nodes"
}

create_volume_if_missing "${VOLUME_NAME_PREFIX}-prefill-0"
create_volume_if_missing "${VOLUME_NAME_PREFIX}-decode-0"

cat <<EOF
Attach volumes to GPU Droplets backing each node pool, then mount at:
  /mnt/nvme/checkpoints

vLLM deployments already mount hostPath at that location.
Use a DaemonSet or cloud-init on pool join to format/mount Block Storage.
EOF
