# Implementation Plan — DO Agentic Inference Cloud

This file tracks the implementation roadmap so collaborators and future agents
can pick up work without needing prior context.

## Current Status

Phase 1 complete: architecture specification, core Python modules, and system diagram.
Phase 2 complete: Kubernetes manifests (namespace, router, vLLM pools, Redis, monitoring, ingress).
Phase 3 complete: request router (FastAPI proxy, classifier, Dockerfile).
Phase 4 complete: DO provisioning scripts (cluster, Spaces, volumes, deploy, smoke test).
All planned phases complete.

---

## What Exists Now

```
do-inference-arch/
├── inference-arch-spec.md        ✅ 2-3 page production architecture document
├── PLAN.md                       ✅ This file
├── code/
│   ├── quant_config.py           ✅ FP8/INT8/BF16 per-layer quantization config
│   ├── agentic_scheduler.py      ✅ Iteration-level scheduler with tool preemption
│   └── telemetry.py              ✅ Prometheus metrics + RequestTrace dataclass
├── diagrams/
│   └── system-architecture.mmd   ✅ Mermaid source for full system diagram
├── k8s/                          ✅ Kubernetes manifests (Phase 2)
│   ├── base/
│   │   ├── namespace.yaml
│   │   ├── request-router/
│   │   ├── vllm/
│   │   └── ...
│   └── overlays/dev/             ✅ Single-GPU dev fallback (TP=1)
├── router/                       ✅ Request router (Phase 3)
│   ├── main.py
│   ├── classifier.py
│   ├── Dockerfile
│   └── requirements.txt
└── provision/                    ✅ DO provisioning (Phase 4)
    ├── config.env.example
    ├── provision.sh
    └── 01-create-cluster.sh … 06-smoke-test.sh
```

---

## Deploy (Option A — all on DOKS)

Everything runs in one DOKS cluster. Public API: `ingress-lb` → `request-router`.

### One-time bootstrap (local)

```bash
# Verify GPU slugs for your region
doctl kubernetes options sizes | grep -i gpu

cp provision/config.env.example provision/config.env
# Edit CLUSTER_REGION, PREFILL_NODE_SIZE, DECODE_NODE_SIZE, DO_REGISTRY_NAME
doctl auth init

./provision/provision.sh --dry-run   # preview
./provision/provision.sh             # create cluster + first deploy
```

### Ongoing deploys (GitHub Actions)

Push to `main` runs [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml):
builds router image → pushes to DOCR → `kubectl apply -R -f k8s/` → smoke tests `/healthz`.

**Required GitHub secret:** `DIGITALOCEAN_ACCESS_TOKEN` (read/write DO API token).

### After bootstrap — GPU readiness

1. Verify GPU slugs: `doctl kubernetes options sizes | grep -i gpu`
2. Production (H200 + MI300X): `./provision/01-create-cluster.sh` with `USE_DEV_GPU=0`
3. Dev fallback (single GPU): set `USE_DEV_GPU=1`, create `gpu-dev` pool, deploy overlay:
   ```bash
   KUSTOMIZE_PATH=k8s/overlays/dev ./provision/05-deploy.sh
   ```
4. HuggingFace token (required for Llama 3.1):
   ```bash
   HF_TOKEN=dhf_... ./provision/08-apply-hf-secret.sh
   ```
5. Upload model weights to DO Spaces: `./provision/07-sync-weights.sh ./llama-3.1-8b`
6. Confirm vLLM pods schedule: `kubectl get pods -n inference`
7. End-to-end test: `REQUIRE_CHAT_200=1 ./provision/06-smoke-test.sh`

---

## What Needs to Be Built Next

All phases complete.

### Phase 2 — Kubernetes Manifests ✅

See `k8s/` directory. Deploy with:

```bash
kubectl apply -k k8s/base
# or dev overlay:
kubectl apply -k k8s/overlays/dev
kubectl get pods -n inference
```

### Phase 3 — Request Router Code ✅

```bash
cd router
pip install -r requirements.txt
python -m unittest test_classifier.py
docker build -t do-inference-router .
```

### Phase 4 — GPU Droplet Provisioning ✅

```bash
cp provision/config.env.example provision/config.env
./provision/provision.sh
```

Steps automated:
1. Create DOKS cluster + system node pool
2. Add H200 prefill and MI300X decode GPU node pools (labeled `inference.do/pool`)
3. Create DO Spaces bucket for model weights
4. Create Block Storage volumes for FP8 checkpoint cache
5. Build/push router image to DO Container Registry
6. Deploy `k8s/` manifests and smoke-test `/v1/chat/completions`

---

## Hardware Assignment Rationale

| Role | GPU | Why |
|------|-----|-----|
| Prefill | H200 / B200 | Compute-bound (high TFLOPS). H200: 3,958 FP8 TFLOPS, 4.8 TB/s |
| Decode | MI300X / MI325X | Memory-bandwidth-bound. MI300X: 192GB HBM, 5.3 TB/s |
| Draft model | Any 1x GPU | 7B INT8 ≈ 7GB, fits in any GPU |

## Key Design Decisions

1. **EP over PP for MoE**: non-uniform expert routing makes PP bubbles unpredictable
2. **Disaggregate only at prompt_len > 2K**: 270ms KV transfer cost only justified above this threshold
3. **Triton kernels**: single source compiles to NVIDIA PTX and AMD GCN — no forked codebases
4. **FP8 never on lm_head**: logit resolution determines reasoning quality; always BF16
5. **Ring attention intra-node only**: 25Gbps VPC breaks even above ~512K tokens only

---

## How to Run Tests (Phase 2+)

```bash
# Validate Python modules
cd do-inference-arch
python -c "from code.quant_config import MIXTRAL_QUANT; print(MIXTRAL_QUANT)"
python -c "from code.agentic_scheduler import AgenticBatchScheduler; s = AgenticBatchScheduler(); print(s.stats())"

# Render the Mermaid diagram
# Open diagrams/system-architecture.mmd at https://mermaid.live

# Deploy to DOKS (Phase 4)
kubectl apply -R -f k8s/
kubectl get pods -n inference
curl -X POST http://$(kubectl get svc ingress-lb -n inference -o jsonpath='{.status.loadBalancer.ingress[0].ip}')/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.1-8b","messages":[{"role":"user","content":"Hello"}]}'
```
