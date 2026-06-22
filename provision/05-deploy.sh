#!/usr/bin/env bash
# Deploy Kubernetes manifests and patch router image.

set -euo pipefail

source "$(dirname "$0")/lib/common.sh"
load_config
ensure_doctl_auth
require_cmd kubectl

wait_for_cluster

log "integrating DO Container Registry with cluster..."
if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run: would run doctl kubernetes cluster registry add"
else
  doctl kubernetes cluster registry add "${CLUSTER_NAME}" 2>/dev/null || true
fi

KUSTOMIZE_PATH="${KUSTOMIZE_PATH:-${REPO_ROOT}/k8s/base}"
log "applying k8s manifests from ${KUSTOMIZE_PATH}..."
run kubectl apply -f "${REPO_ROOT}/k8s/base/namespace.yaml"
if [[ "${DRY_RUN}" != "1" ]]; then
  kubectl wait --for=jsonpath='{.status.phase}'=Active namespace/inference --timeout=60s
  doctl registry kubernetes-manifest --namespace inference | kubectl apply -f -
fi
if [[ -f "${KUSTOMIZE_PATH}/kustomization.yaml" ]]; then
  run kubectl apply -k "${KUSTOMIZE_PATH}"
else
  run kubectl apply -R -f "${KUSTOMIZE_PATH}"
fi

if [[ -n "${ROUTER_IMAGE:-}" ]]; then
  log "patching request-router image to ${ROUTER_IMAGE}..."
  run kubectl set image deployment/request-router \
    router="${ROUTER_IMAGE}" \
    -n inference
fi

log "waiting for core deployments..."
if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run: would wait for rollouts"
  exit 0
fi

kubectl rollout status deployment/request-router -n inference --timeout=300s || true
kubectl rollout status deployment/redis -n inference --timeout=300s || true
kubectl rollout status deployment/prometheus -n inference --timeout=300s || true
kubectl rollout status deployment/grafana -n inference --timeout=300s || true

kubectl get pods,svc -n inference
log "deploy complete (GPU pods remain Pending until GPU pools are available)"
