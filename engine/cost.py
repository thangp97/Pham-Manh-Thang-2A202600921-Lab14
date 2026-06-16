"""
Tiện ích ước tính chi phí (USD) theo số token, phục vụ báo cáo Performance/Cost.

Dùng giá blended (gộp input+output) theo USD / 1 triệu token. Đây là ước tính
xấp xỉ cho mục đích báo cáo của lab (chỉ theo dõi total_tokens mỗi lần gọi).
"""
from typing import Dict

# USD / 1,000,000 token (blended, ước tính)
PRICING_PER_1M: Dict[str, float] = {
    "openai/gpt-4o": 5.0,
    "openai/gpt-4o-mini": 0.30,
    "google/gemini-2.5-flash": 0.30,
    # tên không có prefix (khi dùng OpenAI thuần)
    "gpt-4o": 5.0,
    "gpt-4o-mini": 0.30,
}
DEFAULT_PRICE_PER_1M = 1.0


def estimate_cost(model: str, tokens: int) -> float:
    """Ước tính chi phí USD cho 1 lần gọi model với `tokens` token."""
    price = PRICING_PER_1M.get(model, DEFAULT_PRICE_PER_1M)
    return tokens / 1_000_000 * price


def estimate_cost_by_model(tokens_by_model: Dict[str, int]) -> float:
    """Tổng chi phí khi token chia theo nhiều model (vd nhiều judge)."""
    return sum(estimate_cost(m, t) for m, t in tokens_by_model.items())
