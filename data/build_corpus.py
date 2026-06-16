"""
Bước A của Giai đoạn 1: Sinh Corpus (Knowledge Base).

Vì nhóm chưa có tài liệu nguồn sẵn, ta dùng LLM để sinh ra một "sổ tay hỗ trợ"
giả định, chia thành các chunk có ID ổn định. Các ID này chính là Ground Truth
để Giai đoạn 2 tính Hit Rate / MRR cho Retrieval.

Output: data/corpus/chunks.jsonl  (mỗi dòng 1 chunk)
    {"id": "doc_01", "section": "...", "text": "..."}

Chạy:  python data/build_corpus.py
"""
import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Ép UTF-8 để in được emoji/tiếng Việt trên console Windows (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# Cấu hình. Hỗ trợ cả OpenRouter (1 key, nhiều model) lẫn OpenAI thuần.
USE_OPENROUTER = bool(os.getenv("OPENROUTER_API_KEY"))
MODEL = os.getenv("SDG_MODEL", "openai/gpt-4o-mini" if USE_OPENROUTER else "gpt-4o-mini")
NUM_CHUNKS = 25
OUTPUT_PATH = "data/corpus/chunks.jsonl"


def make_client() -> AsyncOpenAI:
    """Tạo client trỏ tới OpenRouter nếu có OPENROUTER_API_KEY, ngược lại dùng OpenAI."""
    if USE_OPENROUTER:
        return AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    return AsyncOpenAI()  # dùng OPENAI_API_KEY mặc định

# Các chủ đề của sổ tay hỗ trợ. Cố ý có vài chủ đề GẦN GIỐNG NHAU
# (đổi mật khẩu vs khôi phục mật khẩu, hoàn tiền vs hủy đơn) để tạo "nhiễu"
# thật cho retrieval -> MRR/Hit Rate mới có ý nghĩa.
TOPICS = [
    "Tạo tài khoản mới", "Đăng nhập và xác thực 2 lớp (2FA)",
    "Đổi mật khẩu", "Khôi phục mật khẩu đã quên",
    "Cập nhật thông tin cá nhân", "Xóa tài khoản vĩnh viễn",
    "Các gói cước và bảng giá", "Phương thức thanh toán được hỗ trợ",
    "Nâng cấp / hạ cấp gói cước", "Chính sách hoàn tiền",
    "Hủy đơn hàng", "Chính sách bảo hành sản phẩm",
    "Theo dõi trạng thái đơn hàng", "Phí và thời gian vận chuyển",
    "Đổi / trả hàng", "Bảo mật dữ liệu và quyền riêng tư",
    "Cài đặt thông báo", "Tích hợp API cho nhà phát triển",
    "Giới hạn sử dụng (rate limit)", "Liên hệ bộ phận hỗ trợ",
    "Giờ làm việc của tổng đài", "Chính sách bảo trì hệ thống",
    "Khắc phục lỗi không tải được trang", "Sử dụng trên thiết bị di động",
    "Điều khoản dịch vụ",
]

SYSTEM_PROMPT = (
    "Bạn là chuyên viên biên soạn tài liệu (technical writer) cho bộ phận "
    "Chăm sóc khách hàng của một công ty phần mềm SaaS. Văn phong chuyên nghiệp, "
    "rõ ràng, đúng sự thật, dùng tiếng Việt."
)

USER_PROMPT_TEMPLATE = (
    "Viết MỘT đoạn nội dung cho mục sổ tay hỗ trợ có tiêu đề: \"{topic}\".\n"
    "Yêu cầu:\n"
    "- Độ dài 80-150 từ, là một đoạn văn hoàn chỉnh, tự chứa thông tin.\n"
    "- Nêu các bước/quy định cụ thể (có thể có con số: thời gian, phí, ngày...).\n"
    "- KHÔNG lặp lại tiêu đề ở đầu đoạn, chỉ viết phần nội dung.\n"
    "Trả về JSON đúng dạng: {{\"text\": \"...nội dung đoạn...\"}}"
)


async def generate_chunk(client: AsyncOpenAI, topic: str) -> str:
    """Sinh nội dung text cho 1 chunk từ một chủ đề."""
    resp = await client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(topic=topic)},
        ],
    )
    content = resp.choices[0].message.content
    text = json.loads(content)["text"].strip()
    usage = resp.usage.total_tokens if resp.usage else 0
    return text, usage


async def main() -> None:
    if not (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("❌ Thiếu OPENROUTER_API_KEY (hoặc OPENAI_API_KEY) trong .env")
        return

    client = make_client()
    topics = TOPICS[:NUM_CHUNKS]
    print(f"📚 Đang sinh {len(topics)} chunk corpus bằng model '{MODEL}'...")

    # Sinh song song để nhanh
    results = await asyncio.gather(
        *(generate_chunk(client, t) for t in topics), return_exceptions=True
    )

    os.makedirs("data/corpus", exist_ok=True)
    chunks = []
    total_tokens = 0
    for i, (topic, result) in enumerate(zip(topics, results), start=1):
        if isinstance(result, Exception):
            print(f"⚠️ Bỏ qua '{topic}': {result}")
            continue
        text, tokens = result
        total_tokens += tokens
        chunks.append({"id": f"doc_{i:02d}", "section": topic, "text": text})

    # Validate: ID unique, không chunk rỗng
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "ID chunk bị trùng!"
    assert all(c["text"] for c in chunks), "Có chunk rỗng!"

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"✅ Đã tạo {len(chunks)} chunk -> {OUTPUT_PATH}")
    print(f"💰 Tổng token dùng để sinh corpus: {total_tokens:,}")


if __name__ == "__main__":
    asyncio.run(main())
