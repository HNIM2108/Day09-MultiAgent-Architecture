from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import chromadb

# Import bộ phân tách liên tầng đã tối ưu hóa ở Task 2
from rag.parser import parse_policy_markdown


class ChromaPolicyStore:
    """
    Hệ thống quản lý và truy vấn kho dữ liệu chính sách VinShop 
    sử dụng cơ sở dữ liệu ChromaDB Local vĩnh viễn ổ cứng.
    """

    def __init__(
        self,
        persist_directory: Path,
        embedding_model: Any,
        collection_name: str = "policy_chunks",
    ) -> None:
        # TODO 1: Khởi tạo kết nối lưu trữ ChromaDB và cấu hình Collection
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        # Đảm bảo thư mục lưu trữ database tồn tại
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Ép ChromaDB ghi dữ liệu trực tiếp xuống ổ cứng cục bộ của máy
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def ensure_index(self, markdown_path: Path) -> None:
        """
        TODO 2: Đảm bảo chỉ mục luôn sẵn sàng. 
        Tự động kích hoạt luồng biên dịch dữ liệu (Rebuild) nếu database trống.
        """
        count = self.collection.count()
        if count == 0:
            print(f"🧹 Kho lưu trữ Vector hiện đang trống. Bắt đầu lập chỉ mục tự động...")
            self.rebuild(markdown_path)
        else:
            print(f"✅ Kho lưu trữ Vector sẵn sàng hoạt động với {count} chunks có sẵn.")

    def rebuild(self, markdown_path: Path) -> None:
        """
        TODO 3: Đọc tệp tin .md thật, chạy băm cấu trúc liên tầng, 
        sinh Vector nhúng và nạp dữ liệu hàng loạt xuống ChromaDB.
        """
        if not markdown_path.exists():
            print(f"❌ Không tìm thấy tệp tin chính sách tại đường dẫn: {markdown_path}")
            return

        # Đọc tệp tin Markdown thật
        markdown_text = markdown_path.read_text(encoding="utf-8")
        
        # Gọi bộ phân tách Parser tối ưu từ Task 2
        chunks = parse_policy_markdown(markdown_text)
        
        if not chunks:
            print("⚠ Không trích xuất được chunk nào hợp lệ từ tài liệu.")
            return

        print(f"🚀 Tiến hành sinh vector nhúng và nạp {len(chunks)} chunks vào ChromaDB...")
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            # 1. Sinh mã định danh duy nhất cho từng mảnh dữ liệu
            ids.append(f"policy_chunk_{idx:03d}")
            
            # 2. Sử dụng mô hình MiniLM nhúng chuỗi rendered_text giàu ngữ cảnh
            # (Hàm embed_query của LangChain Embeddings trả về mảng float)
            emb = self.embedding_model.embed_query(chunk["rendered_text"])
            embeddings.append(emb)
            
            # 3. Lưu nội dung thô hiển thị thực tế
            documents.append(chunk["content"])
            
            # 4. Đóng gói Metadata sạch phục vụ trích dẫn
            metadatas.append({
                "section_h2": chunk["section_h2"],
                "section_h3": chunk["section_h3"],
                "citation": chunk["citation"],
                "sub_index": chunk.get("sub_index", 0)
            })

        # Nạp dữ liệu hàng loạt xuống ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"✅ Đồng bộ chỉ mục thành công! ChromaDB đang lưu trữ {self.collection.count()} vectors.")

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """
        TODO 4: Tìm kiếm tương đồng vector và format dữ liệu đầu ra đồng nhất.
        """
        # Nhúng câu hỏi của người dùng thành vector nhúng ngữ nghĩa
        query_vector = self.embedding_model.embed_query(query)
        
        # Truy vấn khoảng cách gần nhất trên ChromaDB
        raw_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_hits = []
        if raw_results and raw_results["documents"] and len(raw_results["documents"][0]) > 0:
            docs = raw_results["documents"][0]
            metas = raw_results["metadatas"][0]
            distances = raw_results["distances"][0]

            for doc, meta, dist in zip(docs, metas, distances):
                formatted_hits.append({
                    "content": doc,
                    "citation": meta.get("citation", "Unknown Source"),
                    "metadata": meta,
                    "distance": float(dist)
                })
                
        return formatted_hits

if __name__ == "__main__":
    print("=" * 70)
    print("🎯 Testing Task 3: Production Chroma Policy Store with Project Embeddings")
    print("=" * 70)
    
    # Sử dụng lớp nhúng nội bộ chuẩn của dự án thay vì gọi thư viện bên ngoài
    from embeddings import SentenceTransformerEmbeddings

    print("⏳ Đang khởi tạo mô hình nhúng nội bộ từ src/rag/embeddings.py...")
    project_embedding_model = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    base_dir = Path(__file__).parent.parent.parent
    db_dir = base_dir / "data" / "policy_chroma_db"
    policy_file = base_dir / "data" / "policy_mock_vi.md"

    # Xóa database cũ để đồng bộ dữ liệu chuẩn hóa mới
    import shutil
    if db_dir.exists():
        shutil.rmtree(db_dir)

    # Khởi tạo Store kết nối
    store = ChromaPolicyStore(persist_directory=db_dir, embedding_model=project_embedding_model)
    store.ensure_index(policy_file)
    
    # Thử nghiệm truy vấn thực tế
    test_query = "Thời gian xử lý đơn hàng giao nhanh hỏa tốc"
    print(f"\n❓ Câu hỏi thử nghiệm: '{test_query}'")
    print("-" * 70)
    
    hits = store.search(test_query, top_k=2)
    for i, hit in enumerate(hits, 1):
        print(f"🏆 Top-{i} [Khoảng cách hình học: {hit['distance']:.4f}]")
        print(f"   📂 Nguồn trích dẫn: {hit['citation']}")
        print(f"   📝 Nội dung đoạn:\n{hit['content'].strip()}")
        print("-" * 70)