"""
Iteration-level batch scheduler for agentic inference workloads.

Standard continuous batching (vLLM-style) stalls GPU slots during tool execution.
This scheduler preempts tool-blocked requests by swapping their KV cache to CPU,
freeing GPU slots for other requests immediately.

At every decode step the scheduler re-evaluates the batch:
  1. Resume requests whose tool results have arrived (CPU→GPU KV swap)
  2. Preempt requests that just issued a tool call (GPU→CPU KV swap)
  3. Fill freed slots from the waiting queue (chunked prefill if needed)
"""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class Request:
    request_id: str
    prompt_len: int
    max_new_tokens: int
    kv_len: int = 0               # tokens currently in KV cache (grows during decode)
    _tool_call_pending: bool = False
    _tool_result_ready: bool = False

    def emitted_tool_call(self) -> bool:
        return self._tool_call_pending

    def tool_result_ready(self) -> bool:
        return self._tool_result_ready

    def swap_kv_to_cpu(self) -> None:
        # In production: cudaMemcpyAsync D2H, double-buffered to avoid stall.
        # KV pages are returned to the PagedAttention page pool on the GPU.
        self._tool_call_pending = False

    def swap_kv_to_gpu(self) -> None:
        # In production: cudaMemcpyAsync H2D, pages re-allocated from pool.
        self._tool_result_ready = False


class AgenticBatchScheduler:
    """
    max_batch_tokens: derived from GPU HBM bandwidth budget.
    H200 at 4.8 TB/s can load 32K BF16 KV tokens in ~1ms — within one decode step.

    max_seqs: practical limit on number of concurrent sequences before
    attention kernel overhead dominates (typically 256–512 for 70B models).
    """

    def __init__(self, max_batch_tokens: int = 32_768, max_seqs: int = 256):
        self.max_batch_tokens = max_batch_tokens
        self.max_seqs = max_seqs
        self.running: list[Request] = []
        self.waiting: deque[Request] = deque()
        # Requests blocked on tool execution — KV is on CPU
        self.swapped: list[Request] = []

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def schedule(self) -> list[Request]:
        # Phase 1: resume swapped requests whose tool results are ready
        newly_resumed = []
        still_swapped = []
        for req in self.swapped:
            if req.tool_result_ready():
                req.swap_kv_to_gpu()
                newly_resumed.append(req)
            else:
                still_swapped.append(req)
        self.swapped = still_swapped
        self.running.extend(newly_resumed)

        # Phase 2: preempt requests that emitted a tool call this step
        still_running = []
        for req in self.running:
            if req.emitted_tool_call():
                req.swap_kv_to_cpu()
                self.swapped.append(req)
            else:
                still_running.append(req)
        self.running = still_running

        # Phase 3: fill remaining capacity from the waiting queue
        current_tokens = sum(r.kv_len for r in self.running)
        token_budget = self.max_batch_tokens - current_tokens

        while self.waiting and token_budget > 0 and len(self.running) < self.max_seqs:
            req = self.waiting[0]
            if req.prompt_len <= token_budget:
                self.running.append(self.waiting.popleft())
                token_budget -= req.prompt_len
                req.kv_len = req.prompt_len
            else:
                # Head-of-line: use chunked prefill — process first chunk_size tokens,
                # leave remainder in waiting queue for next step.
                # This prevents a large prompt from permanently blocking the queue.
                chunk_size = min(token_budget, 4096)
                if chunk_size > 0:
                    req.prompt_len -= chunk_size
                    partial = Request(
                        request_id=req.request_id + "_chunk",
                        prompt_len=chunk_size,
                        max_new_tokens=0,
                        kv_len=chunk_size,
                    )
                    self.running.append(partial)
                    token_budget -= chunk_size
                break

        return self.running

    def stats(self) -> dict:
        return {
            "running": len(self.running),
            "waiting": len(self.waiting),
            "swapped": len(self.swapped),
            "batch_tokens": sum(r.kv_len for r in self.running),
        }
