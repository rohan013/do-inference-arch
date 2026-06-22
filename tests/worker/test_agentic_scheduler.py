"""Tests for the agentic batch scheduler."""

from worker.agentic_scheduler import AgenticBatchScheduler, Request


def _request(request_id: str, prompt_len: int, **kwargs) -> Request:
    return Request(request_id=request_id, prompt_len=prompt_len, max_new_tokens=128, **kwargs)


class TestAgenticBatchScheduler:
    def test_add_request_increases_waiting(self):
        scheduler = AgenticBatchScheduler()
        scheduler.add_request(_request("r1", 100))
        assert scheduler.stats()["waiting"] == 1

    def test_schedule_fills_running_from_waiting(self):
        scheduler = AgenticBatchScheduler(max_batch_tokens=1000, max_seqs=4)
        scheduler.add_request(_request("r1", 100))
        scheduler.add_request(_request("r2", 200))
        running = scheduler.schedule()
        assert len(running) == 2
        stats = scheduler.stats()
        assert stats["running"] == 2
        assert stats["waiting"] == 0
        assert stats["batch_tokens"] == 300

    def test_schedule_respects_max_seqs(self):
        scheduler = AgenticBatchScheduler(max_batch_tokens=10_000, max_seqs=1)
        scheduler.add_request(_request("r1", 50))
        scheduler.add_request(_request("r2", 50))
        running = scheduler.schedule()
        assert len(running) == 1
        assert scheduler.stats()["waiting"] == 1

    def test_schedule_respects_token_budget(self):
        scheduler = AgenticBatchScheduler(max_batch_tokens=100, max_seqs=10)
        scheduler.add_request(_request("r1", 100))
        scheduler.add_request(_request("r2", 50))
        running = scheduler.schedule()
        assert len(running) == 1
        assert scheduler.stats()["waiting"] == 1

    def test_preempt_tool_call_moves_to_swapped(self):
        scheduler = AgenticBatchScheduler()
        req = _request("r1", 50, kv_len=50)
        req._tool_call_pending = True
        scheduler.running = [req]
        scheduler.schedule()
        assert scheduler.stats()["running"] == 0
        assert scheduler.stats()["swapped"] == 1
        assert req._tool_call_pending is False

    def test_resume_tool_result_moves_back_to_running(self):
        scheduler = AgenticBatchScheduler()
        req = _request("r1", 50, kv_len=50)
        req._tool_result_ready = True
        scheduler.swapped = [req]
        running = scheduler.schedule()
        assert req in running
        assert scheduler.stats()["swapped"] == 0
        assert req._tool_result_ready is False

    def test_swapped_request_stays_when_tool_not_ready(self):
        scheduler = AgenticBatchScheduler()
        req = _request("r1", 50, kv_len=50)
        scheduler.swapped = [req]
        running = scheduler.schedule()
        assert running == []
        assert scheduler.stats()["swapped"] == 1

    def test_chunked_prefill_for_oversized_head_of_line(self):
        scheduler = AgenticBatchScheduler(max_batch_tokens=4096, max_seqs=10)
        scheduler.add_request(_request("large", 10_000))
        running = scheduler.schedule()
        assert len(running) == 1
        assert running[0].request_id == "large_chunk"
        assert running[0].kv_len == 4096
        assert scheduler.waiting[0].prompt_len == 10_000 - 4096

    def test_request_tool_state_helpers(self):
        req = _request("r1", 10)
        req._tool_call_pending = True
        req._tool_result_ready = True
        assert req.emitted_tool_call() is True
        assert req.tool_result_ready() is True
        req.swap_kv_to_cpu()
        req.swap_kv_to_gpu()
        assert req.emitted_tool_call() is False
        assert req.tool_result_ready() is False
