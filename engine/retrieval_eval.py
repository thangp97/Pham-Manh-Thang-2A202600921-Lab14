"""
Khối B của Giai đoạn 2: Retrieval Evaluation thật.

Đo riêng chất lượng bước Retrieval của Agent bằng cách so:
    expected_retrieval_ids (đáp án đúng, từ golden_set)
    vs retrieved_ids       (tài liệu Agent thực sự lấy, từ MainAgent)

Hai chỉ số:
- Hit Rate: trong top-k có ít nhất 1 tài liệu đúng không (1/0).
- MRR     : 1 / vị trí tài liệu đúng đầu tiên (vị trí càng cao điểm càng lớn).

Nhóm case out-of-scope (expected_ids = []) được tách riêng: không có tài liệu
đúng để "trúng", nên ta đo "abstention" (Agent có biết KHÔNG lấy bừa không)
thay vì gộp vào Hit Rate/MRR (gộp sẽ làm sai lệch chỉ số).
"""
import asyncio
from typing import Dict, List


class RetrievalEvaluator:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def calculate_hit_rate(
        self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = None
    ) -> float:
        """1.0 nếu ít nhất 1 expected_id nằm trong top_k của retrieved_ids."""
        k = top_k or self.top_k
        top_retrieved = retrieved_ids[:k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(
        self, expected_ids: List[str], retrieved_ids: List[str]
    ) -> float:
        """1 / (vị trí 1-indexed của expected_id đầu tiên). Không thấy -> 0."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate_case(
        self, expected_ids: List[str], retrieved_ids: List[str]
    ) -> Dict:
        """Đánh giá 1 case. Trả về hit/rr, hoặc nhãn out-of-scope nếu không có đáp án."""
        if not expected_ids:
            # Out-of-scope: đúng = không lấy tài liệu nào (abstain).
            return {
                "out_of_scope": True,
                "abstained": len(retrieved_ids) == 0,
                "hit": None,
                "rr": None,
            }
        return {
            "out_of_scope": False,
            "abstained": None,
            "hit": self.calculate_hit_rate(expected_ids, retrieved_ids),
            "rr": self.calculate_mrr(expected_ids, retrieved_ids),
        }

    def aggregate(self, per_case: List[Dict]) -> Dict:
        """Tổng hợp metric từ danh sách kết quả per-case (đã có 'eval')."""
        scored = [c for c in per_case if not c["eval"]["out_of_scope"]]
        oos = [c for c in per_case if c["eval"]["out_of_scope"]]

        n = len(scored)
        hit_rate = sum(c["eval"]["hit"] for c in scored) / n if n else 0.0
        mrr = sum(c["eval"]["rr"] for c in scored) / n if n else 0.0

        oos_n = len(oos)
        abstention = sum(1 for c in oos if c["eval"]["abstained"]) / oos_n if oos_n else None

        return {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "num_evaluated": n,
            "num_out_of_scope": oos_n,
            "oos_abstention_rate": abstention,
            "per_case": per_case,
        }

    async def _get_retrieved_ids(self, agent, question: str) -> List[str]:
        """Lấy retrieved_ids. Ưu tiên agent.retrieve() (TF-IDF, miễn phí, không gọi LLM)."""
        if hasattr(agent, "retrieve"):
            return [r["id"] for r in agent.retrieve(question)]
        resp = await agent.query(question)  # fallback cho agent không tách retrieve
        return resp.get("retrieved_ids", [])

    async def evaluate_batch(
        self, agent, dataset: List[Dict], batch_size: int = 10
    ) -> Dict:
        """
        Chạy retrieval của Agent qua toàn bộ dataset và tính Hit Rate / MRR thật.
        """
        per_case: List[Dict] = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            retrieved_lists = await asyncio.gather(
                *(self._get_retrieved_ids(agent, case["question"]) for case in batch)
            )
            for case, retrieved in zip(batch, retrieved_lists):
                expected = case.get("expected_retrieval_ids", [])
                per_case.append(
                    {
                        "id": case.get("id"),
                        "type": case.get("metadata", {}).get("type"),
                        "expected_ids": expected,
                        "retrieved_ids": retrieved,
                        "eval": self.evaluate_case(expected, retrieved),
                    }
                )
        return self.aggregate(per_case)


if __name__ == "__main__":
    import json
    import os
    import sys

    sys.path.insert(0, os.path.abspath("."))
    from agent.main_agent import MainAgent

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    async def demo():
        with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
            dataset = [json.loads(l) for l in f if l.strip()]

        for version in ["v1", "v2"]:
            agent = MainAgent(version=version)
            ev = RetrievalEvaluator(top_k=3)
            res = await ev.evaluate_batch(agent, dataset)
            print(f"\n=== Agent {version} ===")
            print(f"Hit Rate : {res['hit_rate']*100:.1f}%  (trên {res['num_evaluated']} case có đáp án)")
            print(f"MRR      : {res['mrr']:.3f}")
            print(f"Out-of-scope: {res['num_out_of_scope']} case, "
                  f"tỉ lệ abstain (không lấy bừa): "
                  f"{(res['oos_abstention_rate'] or 0)*100:.1f}%")

    asyncio.run(demo())
