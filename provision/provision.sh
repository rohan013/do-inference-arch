#!/usr/bin/env bash
# End-to-end DigitalOcean provisioning for the inference stack (Phase 4).

set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run] [step]

Steps (default: all):
  01-create-cluster
  02-create-spaces
  03-create-volumes
  04-publish-router
  05-deploy
  06-smoke-test
  07-sync-weights
  08-apply-hf-secret

Environment:
  DRY_RUN=1          Print commands without executing
  provision/config.env   Required configuration (copy from config.env.example)

Example:
  cp provision/config.env.example provision/config.env
  ./provision/provision.sh --dry-run
  ./provision/provision.sh
EOF
}

STEP=""
for arg in "$@"; do
  case "${arg}" in
    --dry-run) export DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) STEP="${arg}" ;;
  esac
done

DIR="$(cd "$(dirname "$0")" && pwd)"

run_step() {
  local script="$1"
  log "=== ${script} ==="
  bash "${DIR}/${script}"
}

source "${DIR}/lib/common.sh"

if [[ -n "${STEP}" ]]; then
  run_step "${STEP}"
  exit 0
fi

run_step 01-create-cluster.sh
run_step 02-create-spaces.sh
run_step 03-create-volumes.sh
run_step 04-publish-router.sh
run_step 05-deploy.sh
run_step 06-smoke-test.sh

log "Phase 4 provisioning complete"
