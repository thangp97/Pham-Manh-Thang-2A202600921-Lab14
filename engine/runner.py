"""
Khối D của Giai đoạn 2: Benchmark Runner (tích hợp Agent + Retrieval Eval + Multi-Judge).

Mỗi case: gọi Agent -> đo latency/token -> tính Retrieval (Hit/MRR) -> chấm Multi-Judge.
Chạy song song theo batch để nhanh (yêu cầu Performance < 2 phút cho 50 case).
"""
import asyncio
import time
from typing import Dict, List

from engine.cost import estimate_cost, estimate_cost_by_model


class BenchmarkRunner:
    def __init__(self, agent, retrieval_eval, judge):
        self.agent = agent
        self.retrieval_eval = retrieval_eval
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()

        # 1. Gọi Agent (retrieval + generation)
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time

        # 2. Retrieval Eval (Hit Rate / MRR) — so retrieved_ids vs expected_retrieval_ids
        expected_ids = test_case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])
        retrieval_result = self.retrieval_eval.evaluate_case(expected_ids, retrieved_ids)

        # 3. Multi-Judge chấm câu trả lời
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"],
        )

        # 4. Cost & token
        agent_model = response.get("metadata", {}).get("model", "")
        agent_tokens = response.get("metadata", {}).get("tokens_used", 0)
        judge_tokens = judge_result.get("tokens_used", 0)
        cost = estimate_cost(agent_model, agent_tokens) + estimate_cost_by_model(
            judge_result.get("tokens_by_model", {})
        )

        return {
            "id": test_case.get("id"),
            "type": test_case.get("metadata", {}).get("type"),
            "question": test_case["question"],
            "agent_response": response["answer"],
            "latency": latency,
            "tokens": {"agent": agent_tokens, "judge": judge_tokens,
                       "total": agent_tokens + judge_tokens},
            "cost_usd": cost,
            "retrieval": {
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
                **retrieval_result,
            },
            "judge": judge_result,
            "status": "fail" if judge_result["final_score"] < 3 else "pass",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 10) -> List[Dict]:
        """Chạy song song theo batch để tránh rate limit nhưng vẫn nhanh."""
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results
