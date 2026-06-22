# Implementation Plan — DO Agentic Inference Cloud

This file tracks the implementation roadmap so collaborators and future agents
can pick up work without needing prior context.

## Current Status

Phase 1 complete: architecture specification, core Python modules, and system diagram.
Phase 2 complete: Kubernetes manifests (namespace, router, vLLM pools, Redis, monitoring, ingress).
Phase 3 complete: request router (FastAPI proxy, classifier, Dockerfile).
Phase 4 (GPU droplet provisioning) is next.

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
└── k8s/                          ✅ Kubernetes manifests (Phase 2)
    ├── namespace.yaml
    ├── request-router/
    ├── vllm/
    ├── redis/
    ├── monitoring/
    └── ingress/
└── router/                         ✅ Request router (Phase 3)
    ├── main.py
    ├── classifier.py
    ├── Dockerfile
    └── requirements.txt
```

---

## What Needs to Be Built Next

### Phase 2 — Kubernetes Manifests ✅

See `k8s/` directory. Deploy with:

```bash
kubectl apply -R -f k8s/
kubectl get pods -n inference
```

### Phase 3 — Request Router Code ✅

```bash
cd router
pip install -r requirements.txt
python -m unittest test_classifier.py
docker build -t do-inference-router .
```

### Phase 4 — GPU Droplet Provisioning

Steps (using doctl CLI — already installed in interview environment):
1. `doctl kubernetes cluster create inference-cluster --node-pool "..."`
2. Add GPU node pool: H200 slug for prefill, MI300X slug for decode
3. Create DO Spaces bucket, upload quantized model weights
4. Create Block Storage Volumes, attach to GPU Droplets
5. Deploy manifests: `kubectl apply -f k8s/`
6. Test endpoint: `curl https://<DO-LB-IP>/v1/chat/completions`

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
