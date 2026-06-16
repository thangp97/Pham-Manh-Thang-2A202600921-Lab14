"""
Khối A của Giai đoạn 2: Agent RAG thật (Retrieval + Generation).

Retriever dùng TF-IDF (scikit-learn) trên corpus data/corpus/chunks.jsonl:
- KHÔNG cần embedding API (OpenRouter chỉ proxy chat/completions, không có endpoint
  embeddings ổn định) -> TF-IDF là retrieval thật, deterministic, miễn phí.
- query() trả về `retrieved_ids` THẬT để Giai đoạn 2 (retrieval_eval) tính Hit Rate/MRR.

Generation gọi LLM thật qua OpenRouter, đáp án bám context đã truy hồi.

Hỗ trợ 2 phiên bản để Giai đoạn 3 so Regression:
- "v1": top_k=2, KHÔNG có ngưỡng out-of-scope, prompt lỏng  -> dễ hallucinate.
- "v2": top_k=3, CÓ ngưỡng out-of-scope (biết nói "không có thông tin"),
        prompt chặt "chỉ trả lời dựa trên context".
"""
import asyncio
import json
import os
import sys
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

USE_OPENROUTER = bool(os.getenv("OPENROUTER_API_KEY"))
GEN_MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4o-mini" if USE_OPENROUTER else "gpt-4o-mini")
CORPUS_PATH = "data/corpus/chunks.jsonl"

# Cấu hình khác nhau giữa 2 phiên bản Agent (phục vụ Regression V1 vs V2)
VERSION_CONFIG = {
    "v1": {"top_k": 2, "oos_threshold": 0.0, "strict": False},
    "v2": {"top_k": 3, "oos_threshold": 0.05, "strict": True},
}


def make_client() -> AsyncOpenAI:
    if USE_OPENROUTER:
        return AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    return AsyncOpenAI()


class MainAgent:
    """
    Agent RAG: TF-IDF retrieval + LLM generation.

    Args:
        version: "v1" (base, yếu) hoặc "v2" (optimized, tốt hơn).
    """

    def __init__(self, version: str = "v2", corpus_path: str = CORPUS_PATH):
        if version not in VERSION_CONFIG:
            raise ValueError(f"version phải thuộc {list(VERSION_CONFIG)}")
        self.version = version
        self.name = f"SupportAgent-{version}"
        self.cfg = VERSION_CONFIG[version]
        self.client = make_client()

        # Nạp corpus + dựng chỉ mục TF-IDF
        self.chunks = self._load_corpus(corpus_path)
        self.ids = [c["id"] for c in self.chunks]
        texts = [f"{c['section']}. {c['text']}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(texts)

    @staticmethod
    def _load_corpus(path: str) -> List[Dict]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Thiếu {path}. Hãy chạy 'python data/build_corpus.py' trước."
            )
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def retrieve(self, question: str) -> List[Dict]:
        """Trả về danh sách chunk top_k kèm điểm tương đồng (đã lọc ngưỡng OOS)."""
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = np.argsort(sims)[::-1][: self.cfg["top_k"]]
        results = []
        for idx in top_idx:
            score = float(sims[idx])
            # v2: nếu điểm quá thấp coi như không tìm thấy (out-of-scope)
            if score < self.cfg["oos_threshold"]:
                continue
            results.append({**self.chunks[idx], "score": score})
        return results

    def _build_prompt(self, question: str, contexts: List[str]) -> List[Dict]:
        context_block = "\n\n".join(f"- {c}" for c in contexts) or "(Không có tài liệu liên quan)"
        if self.cfg["strict"]:
            system = (
                "Bạn là trợ lý hỗ trợ khách hàng. CHỈ trả lời dựa trên TÀI LIỆU được cung "
                "cấp. Nếu tài liệu không chứa thông tin, hãy nói rõ 'Tôi không có thông tin "
                "về vấn đề này trong tài liệu hiện có' thay vì bịa. Từ chối lịch sự các yêu "
                "cầu ngoài phạm vi hỗ trợ. Trả lời ngắn gọn bằng tiếng Việt."
            )
        else:
            system = (
                "Bạn là trợ lý hỗ trợ khách hàng. Hãy trả lời câu hỏi của người dùng "
                "dựa trên tài liệu tham khảo. Trả lời bằng tiếng Việt."
            )
        user = f"TÀI LIỆU:\n{context_block}\n\nCÂU HỎI: {question}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    async def query(self, question: str) -> Dict:
        """Quy trình RAG: retrieve -> generate. Trả về retrieved_ids thật + token usage."""
        retrieved = self.retrieve(question)
        retrieved_ids = [r["id"] for r in retrieved]
        contexts = [r["text"] for r in retrieved]

        resp = await self.client.chat.completions.create(
            model=GEN_MODEL,
            temperature=0.2,
            messages=self._build_prompt(question, contexts),
        )
        answer = resp.choices[0].message.content.strip()
        tokens_used = resp.usage.total_tokens if resp.usage else 0

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": GEN_MODEL,
                "version": self.version,
                "tokens_used": tokens_used,
                "sources": [r["section"] for r in retrieved],
            },
        }


if __name__ == "__main__":
    async def test():
        agent = MainAgent(version="v2")
        for q in ["Làm thế nào để đổi mật khẩu?", "Thời tiết Hà Nội hôm nay thế nào?"]:
            resp = await agent.query(q)
            print(f"\nQ: {q}")
            print("retrieved_ids:", resp["retrieved_ids"])
            print("answer:", resp["answer"][:200])
            print("tokens:", resp["metadata"]["tokens_used"])

    asyncio.run(test())
