from __future__ import annotations
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_policy_markdown(markdown_text: str) -> list[dict]:
    """
    Phân tách tài liệu chính sách liên tầng (Hybrid Chunking):
    Bước 1: Tách theo cấu trúc đầu mục tiêu đề (H2, H3).
    Bước 2: Sử dụng Recursive Character Splitter băm nhỏ các mục quá dài
            nhưng vẫn giữ nguyên vẹn nhãn Metadata và Citation gốc.
    """
    structural_chunks = []
    
    current_h2 = "Mục tổng quan"
    current_h3 = ""
    current_content_lines = []

    lines = markdown_text.split("\n")

    def save_structural_chunk():
        content_str = "\n".join(current_content_lines).strip()
        if content_str:
            if current_h3:
                citation_str = f"policy_mock_vi.md > {current_h2} > {current_h3}"
            else:
                citation_str = f"policy_mock_vi.md > {current_h2}"

            structural_chunks.append({
                "section_h2": current_h2,
                "section_h3": current_h3,
                "content": content_str,
                "citation": citation_str
            })

    # --- Bước 1: Quét cấu trúc thô ---
    for line in lines:
        if line.startswith("## "):
            save_structural_chunk()
            current_h2 = line.replace("## ", "").strip()
            current_h3 = ""
            current_content_lines = []
        elif line.startswith("### "):
            save_structural_chunk()
            current_h3 = line.replace("### ", "").strip()
            current_content_lines = []
        else:
            if current_content_lines or line.strip():
                current_content_lines.append(line)
    save_structural_chunk()

    # --- Bước 2: Rã nhỏ các khối text lớn một cách tinh tế ---
    final_chunks = []
    # Khởi tạo bộ băm nhỏ ký tự tự nhiên theo dấu ngắt dòng câu
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Giới hạn kích thước ký tự lý tưởng cho 1 chunk
        chunk_overlap=50,     # Khoảng gối đầu bảo toàn thông tin liên tục
        separators=["\n\n", "\n", "- ", "* ", ". ", " ", ""]
    )

    for parent_chunk in structural_chunks:
        # Thực hiện rã nhỏ nội dung văn bản thô của chunk mẹ
        sub_splits = sub_splitter.split_text(parent_chunk["content"])
        
        for idx, sub_text in enumerate(sub_splits):
            # Tái tạo lại chuỗi rendered_text mang đầy đủ thông tin ngữ cảnh đầu mục
            header_prefix = f"Chương: {parent_chunk['section_h2']}\n"
            if parent_chunk['section_h3']:
                header_prefix += f"Mục: {parent_chunk['section_h3']}\n"
                
            rendered_text = f"{header_prefix}Nội dung đoạn: {sub_text}"
            
            final_chunks.append({
                "section_h2": parent_chunk["section_h2"],
                "section_h3": parent_chunk["section_h3"],
                "content": sub_text,
                "citation": parent_chunk["citation"],
                "rendered_text": rendered_text,
                "sub_index": idx
            })

    return final_chunks


if __name__ == "__main__":
    print("=" * 70)
    print("🎯 Testing Task 2: Advanced Hybrid Markdown Parser")
    print("=" * 70)
    
    base_dir = Path(__file__).parent.parent.parent
    policy_path = base_dir / "data" / "policy_mock_vi.md"
    
    if policy_path.exists():
        raw_text = policy_path.read_text(encoding="utf-8")
        parsed_chunks = parse_policy_markdown(raw_text)
        
        # Số lượng chunks bây giờ sẽ tăng lên khoảng từ 50-70 chunks vì các mục dài đã được rã nhỏ
        print(f"📊 Phân tách liên tầng thành công! Tổng số Chunks thu được: {len(parsed_chunks)}")
        print("-" * 70)
        
        # In thử một đoạn chunk nhỏ đã được rã để kiểm tra tính toàn vẹn ngữ cảnh
        for idx, chunk in enumerate(parsed_chunks[12:14], 13):
            print(f"🏆 [Mẫu Sub-Chunk {idx}]")
            print(f"   📂 Citation: {chunk['citation']}")
            print(f"   📝 Rendered Text:\n{chunk['rendered_text']}")
            print("-" * 70)