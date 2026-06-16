# Reflection cá nhân — Phạm Mạnh Thắng (2A202600921)

## 1. Đóng góp kỹ thuật (Engineering Contribution)

Tôi xây dựng toàn bộ pipeline Evaluation Factory (bài cá nhân), gồm:

| Module | File | Nội dung |
|--------|------|----------|
| SDG + Corpus | `data/build_corpus.py`, `data/synthetic_gen.py` | Sinh KB 25 chunk + 56 golden cases grounded (có `expected_retrieval_ids`) |
| Agent RAG | `agent/main_agent.py` | TF-IDF retrieval + LLM generation, 2 phiên bản V1/V2 |
| Retrieval Eval | `engine/retrieval_eval.py` | Hit Rate, MRR, OOS abstention |
| Multi-Judge | `engine/llm_judge.py` | GPT-4o + Gemini 2.5 Flash, Agreement, Cohen's Kappa, tie-breaker, position bias |
| Runner + Cost | `engine/runner.py`, `engine/cost.py` | Async batch, đo latency/token/cost |
| Tích hợp | `main.py` | Gom metrics, Regression V1/V2, Release Gate đa tiêu chí |

**Kết quả đo thật:** 56 cases · avg 4.12/5 · Hit Rate 95.8% · MRR 0.917 · Agreement 78.6% · Kappa 0.665 · $0.12 · 64s (< 2 phút).

## 2. Chiều sâu kỹ thuật (Technical Depth)

### MRR (Mean Reciprocal Rank)
MRR = trung bình của `1/vị_trí` tài liệu đúng đầu tiên. Khác với Hit Rate (chỉ hỏi "có trúng trong top-k không"), MRR **thưởng cho việc xếp tài liệu đúng lên cao**. Hệ thống của tôi đạt MRR 0.917 ≈ tài liệu đúng gần như luôn ở vị trí 1 — quan trọng vì LLM chú ý context đầu tiên nhiều nhất.

### Cohen's Kappa
Agreement Rate thô (78.6%) có thể bị thổi phồng do 2 judge tình cờ trùng nhau khi cùng thiên về điểm cao. Kappa = `(Po − Pe)/(1 − Pe)` trừ đi phần đồng thuận **ngẫu nhiên** (Pe). Tôi dùng `weights="quadratic"` vì điểm 1–5 là thang **thứ tự** — lệch "4 vs 5" phải bị phạt nhẹ hơn "1 vs 5", điều mà kappa thường (coi nhãn rời rạc ngang nhau) không làm được. Kết quả κ = 0.665 ("substantial") cho thấy hệ đa-judge **đáng tin thật**, không phải ăn may. (Đáng chú ý: trên mẫu nhỏ 5 case, κ = 0.0 do thiếu phương sai — minh chứng kappa cần đủ dữ liệu.)

### Position Bias
LLM judge có thể thiên vị câu trả lời đặt ở vị trí đầu. Tôi kiểm tra bằng cách đưa cặp (A,B) rồi hoán đổi (B,A); nếu kết luận **đổi theo vị trí** → judge bị bias → gắn cờ. Đây là lý do không nên tin một judge đơn lẻ.

### Trade-off Chi phí ↔ Chất lượng
GPT-4o mạnh nhưng đắt gấp ~16× Gemini 2.5 Flash. Tôi kết hợp GPT-4o (judge chính) + Gemini (đối chứng rẻ) + tie-breaker `gpt-4o-mini` chỉ khi xung đột. Đề xuất giảm 30% chi phí: **cascade** — dùng Gemini cho mọi case, chỉ escalate GPT-4o khi nghi ngờ + cache theo hash.

## 3. Giải quyết vấn đề (Problem Solving)

- **OpenRouter không hỗ trợ embeddings** → chuyển retriever sang **TF-IDF (scikit-learn)**: retrieval thật, miễn phí, deterministic; vẫn dùng OpenRouter cho generation + judge.
- **Gemini trả JSON kèm markdown fence** → viết `_parse_json()` chịu lỗi (bóc fence, trích `{...}`).
- **Lỗi UnicodeEncodeError trên Windows (cp1252)** với emoji/tiếng Việt → ép `sys.stdout.reconfigure(encoding="utf-8")`.
- **Phát hiện sâu sắc qua eval:** Hit Rate 95.8% cao nhưng vẫn có fail — nhờ **tách Retrieval Eval riêng** mới biết nhiều lỗi nằm ở **Prompting** (over-refusal q036) chứ không phải Retrieval. Đây là giá trị cốt lõi của một Evaluation Factory: chỉ ra **lỗi nằm ở tầng nào**.

## 4. Bài học rút ra
Đo lường tách tầng (Retrieval vs Generation) + đa-judge có định lượng độ tin cậy (Kappa) là điều kiện cần để cải tiến Agent một cách có bằng chứng, thay vì "cảm tính".
