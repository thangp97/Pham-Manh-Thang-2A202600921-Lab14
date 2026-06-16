"""
Điểm vào chính của Evaluation Factory (Khối D - tích hợp + Giai đoạn 3 Regression).

Quy trình:
1. Chạy benchmark cho Agent V1 (base) và V2 (optimized) trên golden_set.jsonl.
2. Gom metrics: avg_score, Hit Rate, MRR, Agreement Rate, Cohen's Kappa, cost, latency.
3. So sánh Regression V1 vs V2 + Release Gate đa tiêu chí (chất lượng/chi phí/hiệu năng).
4. Ghi reports/summary.json và reports/benchmark_results.json.
"""
import asyncio
import json
import os
import statistics
import sys
import time

from agent.main_agent import MainAgent
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_dataset(path: str = "data/golden_set.jsonl"):
    if not os.path.exists(path):
        print(f"❌ Thiếu {path}. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    if not dataset:
        print(f"❌ File {path} rỗng. Hãy tạo ít nhất 1 test case.")
        return None
    return dataset


def aggregate_metrics(results):
    """Tổng hợp metrics từ kết quả per-case."""
    total = len(results)
    # Retrieval: chỉ tính trên case có đáp án (loại out-of-scope)
    scored = [r for r in results if not r["retrieval"]["out_of_scope"]]
    hit_rate = sum(r["retrieval"]["hit"] for r in scored) / len(scored) if scored else 0.0
    mrr = sum(r["retrieval"]["rr"] for r in scored) / len(scored) if scored else 0.0

    # Multi-Judge reliability
    judge_results = [r["judge"] for r in results]
    reliability = LLMJudge.compute_reliability(judge_results)

    return {
        "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
        "pass_rate": sum(1 for r in results if r["status"] == "pass") / total,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "agreement_rate": reliability["agreement_rate"],
        "cohen_kappa": reliability["cohen_kappa"],
        "total_tokens": sum(r["tokens"]["total"] for r in results),
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 6),
        "avg_latency": round(statistics.mean(r["latency"] for r in results), 3),
    }


async def run_benchmark(label: str, agent_version: str, dataset):
    """Chạy benchmark cho 1 phiên bản Agent, trả về (results, summary)."""
    print(f"🚀 Khởi động Benchmark cho {label} ({len(dataset)} cases)...")
    t0 = time.perf_counter()

    agent = MainAgent(version=agent_version)
    runner = BenchmarkRunner(agent, RetrievalEvaluator(top_k=3), LLMJudge())
    results = await runner.run_all(dataset)

    elapsed = time.perf_counter() - t0
    metrics = aggregate_metrics(results)
    summary = {
        "metadata": {
            "version": label,
            "total": len(results),
            "elapsed_sec": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": metrics,
    }
    print(f"   ✅ {label}: avg_score={metrics['avg_score']:.2f} "
          f"hit_rate={metrics['hit_rate']*100:.1f}% "
          f"agreement={metrics['agreement_rate']*100:.1f}% "
          f"({elapsed:.1f}s, ${metrics['total_cost_usd']:.4f})")
    return results, summary


def release_gate(v1, v2):
    """Quyết định APPROVE/ROLLBACK dựa trên Chất lượng + Chi phí + Hiệu năng."""
    m1, m2 = v1["metrics"], v2["metrics"]
    delta_score = m2["avg_score"] - m1["avg_score"]
    cost_ratio = m2["total_cost_usd"] / m1["total_cost_usd"] if m1["total_cost_usd"] else 1.0
    latency_ratio = m2["avg_latency"] / m1["avg_latency"] if m1["avg_latency"] else 1.0

    reasons = []
    if delta_score < 0:
        reasons.append(f"Chất lượng giảm (Δscore={delta_score:+.2f})")
    if cost_ratio > 1.10:
        reasons.append(f"Chi phí tăng >10% (x{cost_ratio:.2f})")
    if latency_ratio > 1.10:
        reasons.append(f"Latency tăng >10% (x{latency_ratio:.2f})")

    decision = "APPROVE" if not reasons else "ROLLBACK"
    return {
        "decision": decision,
        "delta_score": round(delta_score, 3),
        "cost_ratio": round(cost_ratio, 3),
        "latency_ratio": round(latency_ratio, 3),
        "reasons": reasons,
        "v1_score": round(m1["avg_score"], 3),
        "v2_score": round(m2["avg_score"], 3),
    }


async def main():
    dataset = load_dataset()
    if dataset is None:
        return

    v1_results, v1_summary = await run_benchmark("Agent_V1_Base", "v1", dataset)
    v2_results, v2_summary = await run_benchmark("Agent_V2_Optimized", "v2", dataset)

    gate = release_gate(v1_summary, v2_summary)
    v2_summary["regression"] = gate

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    print(f"V1 Score: {gate['v1_score']}  |  V2 Score: {gate['v2_score']}  "
          f"|  Δ: {gate['delta_score']:+.2f}")
    print(f"Cost ratio: x{gate['cost_ratio']}  |  Latency ratio: x{gate['latency_ratio']}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)
    # Lưu thêm V1 để phân tích
    with open("reports/v1_summary.json", "w", encoding="utf-8") as f:
        json.dump(v1_summary, f, ensure_ascii=False, indent=2)

    if gate["decision"] == "APPROVE":
        print("\n✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print(f"\n❌ QUYẾT ĐỊNH: TỪ CHỐI (ROLLBACK) — {'; '.join(gate['reasons'])}")
    print("📁 Đã ghi reports/summary.json & reports/benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
