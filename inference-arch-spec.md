# Kernel Optimization for AI Inference
### Production Architecture Specification — DigitalOcean Agentic Inference Cloud

---

## Abstract

DigitalOcean's Agentic Inference Cloud must simultaneously serve three fundamentally different model classes: large Mixture-of-Experts models (200B+ parameters), dense long-context reasoning models, and small fast quantized draft models. Each class has different hardware bottlenecks, parallelism requirements, and precision tolerances — a single configuration cannot optimize for all three.

This document proposes a layered inference stack where **precision, parallelism, and caching decisions are co-designed**, not independently tuned. Key design outcomes:

- AMD MI300X/MI325X nodes handle decode (192–256GB HBM, memory-bandwidth-bound)
- NVIDIA H200/B200 nodes handle prefill (3,958–9,000 FP8 TFLOPS, compute-bound)
- FP8 quantization applied selectively — safe for MoE FFN layers, never for final projection
- Disaggregated prefill/decode only for prompts >2K tokens (KV transfer cost: ~270ms at 25Gbps)
- Target KPIs: TTFT P99 <800ms, TPOT <30ms/token, cold start <30s for 200B model

---

## Hardware Reference

| GPU | HBM | Bandwidth | FP8 TFLOPS | BF16 TFLOPS | Interconnect |
|-----|-----|-----------|------------|-------------|--------------|
| NVIDIA H200 SXM | 141 GB HBM3e | 4.8 TB/s | 3,958 | 1,979 | NVLink 900 GB/s |
| NVIDIA B200 | 192 GB HBM3e | 8.0 TB/s | ~9,000 | ~4,500 | NVLink 1,800 GB/s |
| AMD MI300X | 192 GB HBM3 | 5.3 TB/s | 2,614 | 1,307 | xGMI 896 GB/s |
| AMD MI325X | 256 GB HBM3e | 6.0 TB/s | ~3,000 | ~1,500 | xGMI 896 GB/s |
| AMD MI350X | 288 GB HBM3e | 8.0 TB/s | ~5,000 | ~2,500 | xGMI 896 GB/s |

**Assignment rationale:** Prefill is compute-bound (large matrix multiplications over the full prompt). H200/B200 win here with higher TFLOPS. Decode is memory-bandwidth-bound (load model weights once per generated token). MI300X/MI325X win here with more HBM and higher bandwidth, enabling more concurrent KV pages in memory.

---

## Section 1 — Kernel & Precision Engineering

### 1.1 Kernel Development Strategy

The fleet spans both NVIDIA and AMD hardware. Maintaining two separate CUDA and HIP codebases doubles maintenance cost and diverges over time. The strategy is **Triton as the primary kernel authoring language** — it compiles to PTX (NVIDIA) and GCN/CDNA (AMD) from a single source, achieving ~90% of hand-tuned CUDA performance while running on both fleets without forking.

Exceptions where Triton is insufficient:

- **H200 attention hot path**: Use CUTLASS-based FlashAttention-3, which exploits WGMMA tensor core instructions and warp specialization (producer/consumer split) available only on Hopper architecture. These instructions are not expressible in Triton.
- **MoE fused dispatch kernel**: The expert routing path (gate logits → topk → scatter → expert GEMM → gather) crosses three separate HBM round-trips in a naive implementation. A fused Triton kernel eliminates all intermediate allocations, keeping routing indices and gate logits in SRAM across the full dispatch cycle.
- **Chunked QKV projection**: For sequences exceeding 32K tokens, the Q/K/V projection GEMMs become memory-bandwidth-bound. Tiling the sequence dimension into 4K-token chunks with the KV tile held in SRAM across chunks converts this to a compute-bound operation.

### 1.2 Precision Strategy

Numbers in GPUs can be stored at different levels of detail. More detail = more memory, slower throughput. Less detail = faster, but risks degrading model accuracy. The right choice depends on which part of the model and what the model is doing.

| Precision | Use Case | Safe When | Avoid When |
|-----------|----------|-----------|------------|
| **FP8 E4M3** | MoE FFN expert layers, prefill GEMMs | Post-ReLU activations (bounded range) | `lm_head`, `embed_tokens` (logit resolution matters) |
| **FP8 E5M2** | Attention score accumulation | Short context (<16K tokens) | Long context — activation outliers cause overflow |
| **INT8 SmoothQuant** | Dense reasoning model weights + KV cache | After SmoothQuant α=0.85 migration; MMLU regression <0.5% | Chain-of-thought tasks without quantization-aware training |
| **BF16** | `embed_tokens`, `lm_head`, first/last transformer layers | Always — reference precision floor | N/A |

**The non-negotiable rule:** The final layer that picks the next word (`lm_head`) must always run in BF16. Small differences in logit values determine whether a reasoning model continues a chain-of-thought or collapses to a shorter answer. Before promoting any new quantization configuration to production, run a regression on a 1,000-sample MMLU/MATH-500 calibration set. A >1% accuracy drop triggers an automatic rollback.

### 1.3 Memory Hierarchy & Tiling

```
H200 Memory Hierarchy — Attention Kernel Placement
────────────────────────────────────────────────────

  L2 Cache (50MB shared across SMs)
  ┌──────────────────────────────┐
  │  KV hot tiles (current chunk)│  ← stays here across the inner loop
  │  Expert routing logits       │
  └──────────────────────────────┘
              │ spill
  HBM (141GB @ 4.8 TB/s)
  ┌──────────────────────────────┐
  │  Full KV cache (paged, 16-tok│  ← PagedAttention blocks
  │  Model weights (FP8 tiles)   │  ← streamed in, not resident
  │  Activations                 │
  └──────────────────────────────┘
              │ overflow
  CPU RAM / Local NVMe
  ┌──────────────────────────────┐
  │  Swapped KV (tool-blocked)   │  ← agentic requests waiting on tools
  │  FP8 model checkpoint        │  ← cold start cache (~17s load vs ~33s from Spaces)
  └──────────────────────────────┘
```

### 1.4 Long-Context Attention

**Ring Attention** (sequence parallelism across GPUs): Split the sequence dimension across all GPUs in a ring. Each GPU holds `seq_len / N` tokens. K/V chunks rotate around the ring via NVLink while each GPU accumulates its local attention contribution using online softmax. This is only viable **intra-node** — NVLink provides 900 GB/s bidirectional bandwidth. At 25 Gbps inter-node VPC, the communication overhead exceeds compute savings for all practical sequence lengths below ~512K tokens.

**Sliding Window Attention** (Mistral/Mixtral native): Only attend to the nearest W tokens (e.g., W=4,096). KV cache memory is capped at `W × d_head × n_heads × sizeof(dtype)` regardless of total session length — critical for multi-turn agentic conversations that accumulate context over many turns.

**Chunked Prefill** for TTFT control: Split long prompts into 2–4K token chunks processed sequentially, interleaved with ongoing decode steps. Prefill is compute-bound and can absorb delay. Decode is latency-bound and cannot stall. Chunking gives the scheduler interrupt points to serve pending decode requests during a long prefill.

---

## Section 2 — Distributed Inference & Execution Orchestration

### 2.1 Parallelism Strategy by Model Class

```
┌─────────────────────────────────────────────────────────────┐
│ MoE 200B+ (e.g., Mixtral-style, 8 active experts of 64)     │
│   Primary:   Expert Parallelism EP=8 (one expert set/device) │
│   Secondary: Tensor Parallelism TP=2 within each expert GEMM │
│   Config:    16 GPUs minimum; 8-GPU slug → EP=4, TP=2        │
│   AVOID:     Pipeline Parallelism — explained below          │
├─────────────────────────────────────────────────────────────┤
│ Dense Long-Context (400B+, e.g., Llama-class)               │
│   Primary:   Tensor Parallelism TP=8 (intra-node, NVLink)   │
│   Secondary: Pipeline Parallelism PP=2 (inter-node only)    │
│   PP bubble: ~11% at micro_batch=8 — acceptable             │
│   Config:    16 GPUs (2 nodes of TP=8)                      │
├─────────────────────────────────────────────────────────────┤
│ Small Draft Model (<7B)                                     │
│   Primary:   Data Parallelism DP=8 (full replicas per GPU)  │
│   Config:    1x GPU Droplet slugs — 7B INT8 ≈ 7GB fits      │
└─────────────────────────────────────────────────────────────┘
```

**Why Expert Parallelism beats Pipeline Parallelism for MoE:** Expert routing is non-uniform — in any given batch, some experts receive 3× more tokens than others. Pipeline stages stall waiting for the overloaded expert. EP routes tokens directly to the GPU holding the expert; the scatter/gather communication cost is bounded by NVLink bandwidth and is predictable regardless of routing distribution. PP introduces unpredictable bubbles that scale with the imbalance.

### 2.2 Disaggregated Prefill / Decode

Prefill (reading the prompt) and decode (generating the response) have opposite hardware needs. Mixing them on the same GPU forces a trade-off that satisfies neither.

```
                    ┌─────────────────────────────────────┐
                    │         DO Load Balancer             │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │         Request Router               │
                    │   classifies by prompt_len           │
                    └──────┬───────────────────────┬───────┘
                           │                       │
              prompt > 2K  │                       │  prompt ≤ 2K
                           ▼                       ▼
             ┌─────────────────────┐   ┌─────────────────────┐
             │   Prefill Pool      │   │   Decode Pool        │
             │  H200/B200 nodes    │   │  MI300X/MI325X nodes │
             │  (compute-bound)    │   │  (bandwidth-bound)   │
             │  Chunked prefill    │   │  Continuous batch    │
             └──────────┬──────────┘   └──────────▲──────────┘
                        │   KV transfer            │
                        └──────────────────────────┘
                         ~270ms at 25Gbps for 4K ctx
```

The KV transfer cost (~270ms for a 4K-context 200B model at 25Gbps) is acceptable only when it prevents the alternative: a single 32K-token prefill blocking the decode queue for ~800ms. For short prompts (≤2K tokens), the transfer cost exceeds the benefit — route directly to the decode pool.

### 2.3 Distributed KV Cache & Prefix Caching

Rather than allocating KV memory contiguously per request (wasted space for variable-length sequences), **PagedAttention** divides KV cache into fixed 16-token pages. Pages are reference-counted and the page table is maintained at the cluster level — shared across all decode nodes, not per-instance.

**Prefix caching** stores the computed KV pages for repeated prefixes (e.g., a system prompt sent by every API request). The cache is keyed by `SHA256(token_ids[:n])`. On a hit, the decode node fetches pre-computed pages from the cache node over NVLink or VPC rather than recomputing — effectively zeroing out the prefill cost for repeated context.

Eviction policy: frequency-weighted LRU. System prompt pages are pinned indefinitely. User session pages have a 10-minute TTL. The top 1% of pages by access frequency are replicated to all decode nodes asynchronously — the first requester pays the transfer cost; all subsequent requests hit local HBM.

### 2.4 Continuous Batching for Agentic Workflows

Agentic requests behave differently from single-turn chat — they pause mid-generation to call tools (search, code execution, API calls), then resume. A naive batch scheduler holds the GPU slot idle during tool execution, collapsing utilization to ~40%.

The fix is **iteration-level scheduling with preemption**: at every decode step, re-evaluate which requests are runnable.

```python
class AgenticBatchScheduler:
    def schedule(self) -> list[Request]:
        # Resume any requests whose tool calls just completed
        for req in self.swapped:
            if req.tool_result_ready():
                req.swap_kv_to_gpu()   # async H2D transfer
                self.running.append(req)
                self.swapped.remove(req)

        # Preempt requests that just issued a tool call
        for req in list(self.running):
            if req.emitted_tool_call():
                req.swap_kv_to_cpu()   # async D2H transfer, frees GPU slot
                self.swapped.append(req)
                self.running.remove(req)

        # Fill freed slots from the waiting queue
        token_budget = self.max_batch_tokens - sum(r.kv_len for r in self.running)
        while self.waiting and token_budget > 0:
            req = self.waiting[0]
            if req.prompt_len <= token_budget:
                self.running.append(self.waiting.popleft())
                token_budget -= req.prompt_len
            else:
                break
        return self.running
```

`max_batch_tokens=32,768` is derived from H200's 4.8 TB/s bandwidth: loading 32K BF16 KV tokens takes ~1ms, staying within a single decode step budget.

---

## Section 3 — Infrastructure Resiliency & Observability

### 3.1 Cold Start Mitigation

Loading a 200B model (400GB in BF16) from object storage over a network takes ~33 seconds at 12 GB/s — unacceptable for a serverless environment. Three-tier mitigation:

**Tier 1 — Local NVMe cache:** Store FP8-quantized checkpoints (200B model = ~200GB in FP8) on Block Storage Volumes attached to each GPU Droplet. At ~12 GB/s NVMe read, load time drops to ~17s.

**Tier 2 — Compute-overlapped streaming:** Begin the forward pass on layer 0 while layers 1–N continue loading from NVMe. Since transformer layers are sequential, the first layer can execute while the rest load. Effective cold start: ~8–10s.

**Tier 3 — Memory pool pre-allocation:** At node startup, allocate a pinned CUDA memory pool equal to `model_size × 1.1`. On model swap (e.g., A/B deploy), overwrite weights in-place — no reallocation or kernel launch overhead. A DOKS CronJob pre-warms nodes 5 minutes before predicted traffic peaks using historical traffic patterns.

**Fallback:** BF16 reference weights stored in DO Spaces (S3-compatible). On NVMe cache miss, stream from Spaces (~33s). Cache miss rate should be near zero in steady state.

### 3.2 DigitalOcean Infrastructure Mapping

| Component | DigitalOcean Product |
|-----------|----------------------|
| Prefill compute | H200/B200 GPU Droplets (8x slug) |
| Decode compute | MI300X/MI325X GPU Droplets (8x slug) |
| Draft model | 1x GPU Droplet (replicated via DP) |
| Container orchestration | DOKS with GPU node pools |
| Model weight storage | DO Spaces (S3-compatible) |
| Fast checkpoint cache | Block Storage Volumes (NVMe) |
| Inter-node networking | DO VPC (25 Gbps private) |
| Prefix cache metadata | DO Managed Redis |
| API ingress | DO Load Balancer (L7) |
| GPU hardware metrics | DCGM Exporter (NVIDIA) + ROCm SMI (AMD) |
| Metrics aggregation | Self-hosted Prometheus + Grafana on standard Droplet |

### 3.3 Telemetry & KPI Framework

Every request carries a structured trace capturing five timestamps: received → queue exit → prefill start → first token → last token. KPIs are derived at request completion and emitted as structured JSON to a Unix socket (overhead <1µs), aggregated by a Fluent Bit sidecar, and shipped to Prometheus.

| KPI | Definition | P50 Target | P99 Alert | What a violation means |
|-----|-----------|-----------|-----------|------------------------|
| **TTFT** | first_token_time − received_time | <200ms | <800ms | Queue backup or prefill bottleneck |
| **TPOT** | decode_duration / (output_tokens − 1) | <30ms | <80ms | Decode pool undersized or batch too large |
| **ITL** | P99 inter-token gap within a request | <35ms | <100ms | Batch preemption spikes; correlate with swap_count |
| **Tokens/$/hr** | output_tokens / (node_cost × hours) | >5,000 | <2,000 | GPU utilization too low; check MFU |
| **Prefill MFU** | achieved_TFLOPS / peak_TFLOPS | >55% | <30% | Kernel not using tensor cores; check precision config |
| **KV Cache Hit Rate** | prefix_tokens_reused / prompt_tokens | >70% | <40% | Prefix cache misconfigured; check TTL or DHT |
| **Expert Load CV** | std(expert_tokens) / mean | <0.3 | >0.6 | MoE routing collapse; one expert absorbing all traffic |

GPU-level metrics (SM utilization, HBM bandwidth, NVLink throughput) are collected via DCGM (NVIDIA) and ROCm SMI (AMD) at 1-second granularity and surfaced in a Grafana dashboard alongside the application-level KPIs above.

---

## Appendix: Key Trade-off Decisions

| Decision | Choice | Rejected Alternative | Reason |
|----------|--------|---------------------|--------|
| MoE parallelism | Expert Parallelism (EP) | Pipeline Parallelism (PP) | PP bubbles are non-deterministic with hot experts |
| Decode hardware | AMD MI300X/MI325X | NVIDIA H200 | More HBM (192–256GB vs 141GB) = more concurrent KV pages |
| Disaggregation threshold | prompt_len >2K | Always disaggregate | 270ms KV transfer only justified when preventing >800ms stalls |
| Ring attention scope | Intra-node only | Cross-node | 25Gbps VPC makes inter-node ring viable only above ~512K tokens |
| Kernel language | Triton | Pure CUDA/HIP | Single codebase for both NVIDIA and AMD fleets |
| lm_head precision | Always BF16 | FP8 | Logit resolution determines reasoning chain quality |
