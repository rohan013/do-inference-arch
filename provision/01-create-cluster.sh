#!/usr/bin/env bash
# Create DOKS cluster with system + GPU node pools for prefill/decode.

set -euo pipefail

source "$(dirname "$0")/lib/common.sh"
load_config
ensure_doctl_auth
require_cmd kubectl

if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run: would create cluster ${CLUSTER_NAME} if missing"
elif doctl kubernetes cluster get "${CLUSTER_NAME}" >/dev/null 2>&1; then
  log "cluster ${CLUSTER_NAME} already exists — skipping create"
else
  log "creating cluster ${CLUSTER_NAME} in ${CLUSTER_REGION}..."
  run doctl kubernetes cluster create "${CLUSTER_NAME}" \
    --region "${CLUSTER_REGION}" \
    --version "${CLUSTER_VERSION}" \
    --tag inference \
    --node-pool "name=system;size=${SYSTEM_NODE_SIZE};count=${SYSTEM_NODE_COUNT};auto-scale=false;tags=system"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run: would add GPU node pools ${PREFILL_POOL_NAME}, ${DECODE_POOL_NAME}"
elif ! doctl kubernetes cluster node-pool list "${CLUSTER_NAME}" --format Name --no-header | grep -qx "${PREFILL_POOL_NAME}"; then
  log "adding prefill GPU pool ${PREFILL_POOL_NAME} (${PREFILL_NODE_SIZE})..."
  run doctl kubernetes cluster node-pool create "${CLUSTER_NAME}" \
    --name "${PREFILL_POOL_NAME}" \
    --size "${PREFILL_NODE_SIZE}" \
    --count "${PREFILL_NODE_COUNT}" \
    --auto-scale=false \
    --tag prefill \
    --label "inference.do/pool=prefill"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  :
elif ! doctl kubernetes cluster node-pool list "${CLUSTER_NAME}" --format Name --no-header | grep -qx "${DECODE_POOL_NAME}"; then
  log "adding decode GPU pool ${DECODE_POOL_NAME} (${DECODE_NODE_SIZE})..."
  run doctl kubernetes cluster node-pool create "${CLUSTER_NAME}" \
    --name "${DECODE_POOL_NAME}" \
    --size "${DECODE_NODE_SIZE}" \
    --count "${DECODE_NODE_COUNT}" \
    --auto-scale=false \
    --tag decode \
    --label "inference.do/pool=decode"
fi

wait_for_cluster
label_gpu_pools
log "cluster ${CLUSTER_NAME} ready"
