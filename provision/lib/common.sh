#!/usr/bin/env bash
# Shared helpers for DigitalOcean provisioning scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.env"

DRY_RUN="${DRY_RUN:-0}"

log() {
  printf '[provision] %s\n' "$*"
}

die() {
  printf '[provision] ERROR: %s\n' "$*" >&2
  exit 1
}

run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] %q' "$1"
    shift
    for arg in "$@"; do
      printf ' %q' "${arg}"
    done
    printf '\n'
    return 0
  fi
  "$@"
}

require_cmd() {
  local cmd
  for cmd in "$@"; do
    command -v "${cmd}" >/dev/null 2>&1 || die "missing required command: ${cmd}"
  done
}

load_config() {
  if [[ ! -f "${CONFIG_FILE}" && -f "${SCRIPT_DIR}/config.env.example" ]]; then
    log "config.env not found — using config.env.example (copy and edit for production runs)"
    CONFIG_FILE="${SCRIPT_DIR}/config.env.example"
  fi
  [[ -f "${CONFIG_FILE}" ]] || die "config not found: ${CONFIG_FILE} (copy config.env.example)"
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
  : "${CLUSTER_NAME:?CLUSTER_NAME required}"
  : "${CLUSTER_REGION:?CLUSTER_REGION required}"
}

ensure_doctl_auth() {
  require_cmd doctl
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry-run: skipping doctl auth check"
    return 0
  fi
  if [[ -n "${DO_TOKEN:-}" ]]; then
    export DIGITALOCEAN_ACCESS_TOKEN="${DO_TOKEN}"
  fi
  doctl account get >/dev/null 2>&1 || die "doctl not authenticated — run 'doctl auth init' or set DO_TOKEN"
}

wait_for_cluster() {
  log "waiting for cluster ${CLUSTER_NAME} to become ready..."
  run doctl kubernetes cluster kubeconfig save "${CLUSTER_NAME}" --expiry-seconds 600
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  kubectl wait --for=condition=Ready nodes --all --timeout=900s
}

label_gpu_pools() {
  log "labeling GPU node pools (inference.do/pool)..."
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry-run: would label nodes in pools ${PREFILL_POOL_NAME} and ${DECODE_POOL_NAME}"
    return 0
  fi

  kubectl label nodes -l doks.digitalocean.com/node-pool="${PREFILL_POOL_NAME}" \
    inference.do/pool=prefill --overwrite
  kubectl label nodes -l doks.digitalocean.com/node-pool="${DECODE_POOL_NAME}" \
    inference.do/pool=decode --overwrite
}
