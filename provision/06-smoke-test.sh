#!/usr/bin/env bash
# Smoke-test the inference API via the DO Load Balancer.

set -euo pipefail

source "$(dirname "$0")/lib/common.sh"
load_config
ensure_doctl_auth
require_cmd kubectl curl

if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run: would curl ingress-lb /v1/chat/completions"
  exit 0
fi

run doctl kubernetes cluster kubeconfig save "${CLUSTER_NAME}" --expiry-seconds 600

log "waiting for LoadBalancer IP on ingress-lb..."
LB_IP=""
for _ in $(seq 1 60); do
  LB_IP="$(kubectl get svc ingress-lb -n inference -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)"
  [[ -n "${LB_IP}" ]] && break
  sleep 10
done

[[ -n "${LB_IP}" ]] || die "ingress-lb has no external IP yet"

log "load balancer IP: ${LB_IP}"

HEALTH_CODE="$(curl -s -o /dev/null -w '%{http_code}' "http://${LB_IP}/healthz")"
[[ "${HEALTH_CODE}" == "200" ]] || die "/healthz returned ${HEALTH_CODE}"

log "POST /v1/chat/completions..."
CHAT_CODE="$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://${LB_IP}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${DEFAULT_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}")"
if [[ "${CHAT_CODE}" == "200" ]]; then
  log "chat completions returned 200"
elif [[ "${REQUIRE_CHAT_200:-0}" == "1" ]]; then
  die "chat completions returned ${CHAT_CODE} (expected 200 — check vLLM pods and HF_TOKEN)"
else
  log "WARN: chat completions returned ${CHAT_CODE} (expected until vLLM GPU pods are ready)"
fi

log "smoke test passed — router reachable at http://${LB_IP}/healthz"
