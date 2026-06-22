#!/usr/bin/env bash
# Create or update the HuggingFace token secret in the inference namespace.

set -euo pipefail

source "$(dirname "$0")/lib/common.sh"
load_config
ensure_doctl_auth
require_cmd kubectl

: "${HF_TOKEN:?Set HF_TOKEN (accept Llama 3.1 license at huggingface.co first)}"

wait_for_cluster

log "applying hf-token secret to namespace inference..."
if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run: would create secret hf-token"
  exit 0
fi

kubectl create namespace inference --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic hf-token \
  --namespace inference \
  --from-literal=token="${HF_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f -

log "hf-token secret ready"
