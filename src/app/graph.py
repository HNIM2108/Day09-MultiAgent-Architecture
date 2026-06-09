from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field

from app.config import Settings
from app.state import ShoppingState
from app.data_access import ShoppingDataStore
from rag.vector_store import ChromaPolicyStore
from rag.embeddings import SentenceTransformerEmbeddings

# ĐÃ SỬA: Import chính xác tên hàm từ provider gốc của trường
from provider import get_chat_model

# Import các thành phần xây dựng đồ thị của LangGraph
from langgraph.graph import StateGraph, START, END


# =============================================================================
# CẤU TRÚC ĐẦU RA BẮT BUỘC CHO ROUTER AGENT (TASK 5)
# =============================================================================
class RouterOutput(BaseModel):
    status: Literal["ok", "clarification_needed"] = Field(
        description="Chọn 'ok' nếu câu hỏi rõ ràng. Chọn 'clarification_needed' nếu thiếu thông tin cốt lõi cần làm rõ."
    )
    needs_policy: bool = Field(
        description="Đặt là True nếu câu hỏi cần tra cứu quy định chính sách đổi trả, hoàn tiền, voucher hoặc giao hàng."
    )
    needs_data: bool = Field(
        description="Đặt là True nếu câu hỏi cần tra cứu thông tin đơn hàng (order_id) hoặc hồ sơ khách hàng (customer_id) thực tế."
    )
    clarification_question: str | None = Field(
        default=None,
        description="Nếu thiếu thông tin định danh quan trọng, hãy viết câu hỏi lịch sự hỏi khách hàng. Nếu đủ thông tin, để None."
    )


# =============================================================================
# WORKER NODES IMPLEMENTATION (TASK 5, 6, 7, 8)
# =============================================================================

def supervisor_node(state: ShoppingState) -> dict[str, Any]:
    """Router Agent phân tích câu hỏi và ra quyết định định tuyến bằng mô hình tích hợp."""
    question = state.get("question", "")
    settings = Settings.load()
    
    llm = get_chat_model(settings)
    # Tiếp tục dùng json_mode để tránh lỗi 400 của DeepSeek
    structured_llm = llm.with_structured_output(RouterOutput, method="json_mode")
    
    # THAY ĐỔI: Thắt chặt Prompt ép buộc trả về đúng nhãn 'ok' hoặc 'clarification_needed'
    system_prompt = (
        "Bạn là điều phối viên thông minh của sàn TMĐT VinShop. Hãy trả về duy nhất một đối tượng JSON.\n"
        "Nhiệm vụ của bạn là phân tích câu hỏi của khách hàng và đưa ra quyết định định tuyến chính xác.\n\n"
        "BẮT BUỘC tuân thủ cấu trúc JSON sau đây và KHÔNG ĐƯỢC THAY ĐỔI giá trị hợp lệ:\n"
        "{\n"
        '  "status": "ok" hoặc "clarification_needed",\n'
        '  "needs_policy": true hoặc false,\n'
        '  "needs_data": true hoặc false,\n'
        '  "clarification_question": "Nội dung câu hỏi nếu status là clarification_needed, ngược lại để null"\n'
        "}\n\n"
        "Quy tắc logic:\n"
        "- Nếu khách hỏi về quy chế, chính sách vận chuyển, đổi trả chung chung -> needs_policy=true, status=\"ok\".\n"
        "- Nếu khách hỏi kèm mã đơn hàng cụ thể (ví dụ đơn 1971) hoặc mã khách hàng (C001) -> needs_data=true, status=\"ok\".\n"
        "- Nếu khách hỏi về trạng thái đơn hoặc voucher nhưng KHÔNG cung cấp mã ID -> bắt buộc đặt status=\"clarification_needed\" và điền câu hỏi làm rõ vào clarification_question.\n"
        "- TUYỆT ĐỐI KHÔNG tự bịa ra các giá trị khác cho trường 'status' như 'handled' hay 'success'."
    )
    
    response: RouterOutput = structured_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ])
    
    route_dict = response.model_dump()
    return {
        "route": route_dict,
        "trace": [{
            "agent": "RouterAgent",
            "action": f"Định tuyến câu hỏi. Kết quả phân tích: {route_dict}"
        }]
    }


def worker_1_policy_node(state: ShoppingState) -> dict[str, Any]:
    """Policy Specialist Agent - Gọi RAG ChromaDB thật tra cứu quy chế sàn."""
    question = state.get("question", "")
    base_dir = Path(__file__).parent.parent.parent
    
    embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db_dir = base_dir / "data" / "policy_chroma_db"
    
    store = ChromaPolicyStore(persist_directory=db_dir, embedding_model=embedding_model)
    rag_hits = store.search(query=question, top_k=4)
    
    return {
        "policy_result": {
            "status": "ok" if rag_hits else "empty",
            "hits": rag_hits
        },
        "trace": [{
            "agent": "PolicySpecialistAgent",
            "action": f"Đã tra cứu ChromaDB. Tìm thấy {len(rag_hits)} dữ kiện chính sách."
        }]
    }


def worker_2_data_node(state: ShoppingState) -> dict[str, Any]:
    """Order Specialist Agent - Trích xuất mã ID và truy vấn dữ liệu JSON phẳng chuẩn xác."""
    question = state.get("question", "")
    base_dir = Path(__file__).parent.parent.parent
    json_path = base_dir / "data" / "order_customer_mock_data.json"
    
    data_store = ShoppingDataStore(json_path)
    
    order_match = re.search(r"\b\d{4}\b", question)
    customer_match = re.search(r"\bC\d{3}\b", question, re.IGNORECASE)
    
    data_lookup_results = {"status": "no_identifiers_found"}
    action_note = "Không trích xuất được thông tin order_id hoặc customer_id từ câu hỏi."
    
    if order_match:
        order_id = order_match.group(0)
        res = data_store.get_order_detail_by_order_id(order_id)
        data_lookup_results = res
        action_note = f"Đã tra cứu chi tiết mã đơn hàng phẳng: {order_id}"
    elif customer_match:
        customer_id = customer_match.group(0).upper()
        if "voucher" in question.lower() or "mã giảm giá" in question.lower():
            res = data_store.get_vouchers_by_customer_id(customer_id, only_active=True)
        else:
            res = data_store.get_customer_by_id(customer_id)
        data_lookup_results = res
        action_note = f"Đã tra cứu hồ sơ khách hàng: {customer_id}"

    return {
        "data_result": data_lookup_results,
        "trace": [{
            "agent": "OrderSpecialistAgent",
            "action": f"Xử lý dữ liệu hệ thống. Hành động: {action_note}. Trạng thái kết quả: {data_lookup_results.get('status')}"
        }]
    }


def worker_3_response_node(state: ShoppingState) -> dict[str, Any]:
    """Response Agent - Tổng hợp dữ liệu từ RAG hoặc JSON sinh câu trả lời cá nhân hóa."""
    question = state.get("question", "")
    route = state.get("route", {})
    policy_result = state.get("policy_result", {})
    data_result = state.get("data_result", {})
    
    settings = Settings.load()
    # ĐÃ SỬA: Gọi đúng hàm get_chat_model từ provider
    llm = get_chat_model(settings)

    if route.get("status") == "clarification_needed" and route.get("clarification_question"):
        return {
            "final_answer": route["clarification_question"],
            "trace": [{
                "agent": "ResponseAgent",
                "action": "Đã chuyển tiếp yêu cầu làm rõ thông tin đến khách hàng."
            }]
        }

    context_chunks = []
    if data_result and data_result.get("status") == "ok":
        context_chunks.append(f"DỮ LIỆU HỆ THỐNG THẬT:\n{json.dumps(data_result, ensure_ascii=False, indent=2)}")
    elif data_result and data_result.get("status") == "not_found":
        context_chunks.append(f"THÔNG BÁO HỆ THỐNG: {data_result.get('message')}")
        
    if policy_result and policy_result.get("status") == "ok":
        context_chunks.append("QUY ĐỊNH CHÍNH SÁCH SÀN LIÊN QUAN:")
        for idx, hit in enumerate(policy_result.get("hits", []), 1):
            context_chunks.append(f"[{idx}] Nguồn trích dẫn: {hit['citation']}\nNội dung văn bản: {hit['content']}")

    full_context = "\n\n".join(context_chunks)

    system_prompt = (
        "Bạn là Chuyên viên Chăm sóc Khách hàng chuyên nghiệp và lịch sự của sàn TMĐT VinShop.\n"
        "Nhiệm vụ của bạn là dựa vào phần Ngữ cảnh hệ thống cung cấp để trả lời câu hỏi của khách.\n\n"
        "Quy tắc ứng xử quan trọng:\n"
        "1. Luôn chào hỏi lễ phép, xưng hô lịch sự (Ví dụ: Dạ VinShop xin chào anh/chị...).\n"
        "2. Chỉ sử dụng thông tin có sẵn trong Ngữ cảnh dữ liệu để trả lời. Không tự ý bịa đặt trạng thái, mã bưu cục hay ngày tháng.\n"
        "3. Nếu câu hỏi liên quan đến trạng thái đơn hàng (ví dụ đơn 1971), hãy đọc kỹ các trường dữ liệu thật như 'carrier' (đơn vị vận chuyển), 'latest_status_note' (ghi chú trạng thái mới nhất), 'estimated_delivery' (ngày giao dự kiến) để thông báo chính xác cho khách hàng.\n"
        "4. Nếu thông tin hệ thống báo không tìm thấy đơn hàng, hướng dẫn khách hàng kiểm tra lại mã ID."
    )

    user_message = f"Ngữ cảnh hệ thống tra cứu:\n{full_context}\n\n---\n\nCâu hỏi của khách hàng: {question}"

    ai_response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ])
    
    return {
        "final_answer": ai_response.content.strip(),
        "trace": [{
            "agent": "ResponseAgent",
            "action": "Đã biên soạn câu trả lời cuối cùng gửi khách hàng."
        }]
    }


# =============================================================================
# HÀM ĐIỀU HƯỚNG ĐIỀU KIỆN (CONDITIONAL ROUTER)
# =============================================================================
def route_decision(state: ShoppingState) -> Literal["policy", "data", "both", "response"]:
    """Đọc nhãn định tuyến từ Router để quyết định node tiếp theo."""
    route = state.get("route", {})
    if route.get("status") == "clarification_needed":
        return "response"
    
    policy = route.get("needs_policy", False)
    data = route.get("needs_data", False)
    
    if policy and data:
        return "both"
    elif policy:
        return "policy"
    elif data:
        return "data"
    return "response"


# =============================================================================
# ASSISTANT CLASS IMPLEMENTATION
# =============================================================================
class ShoppingAssistant:
    """Hệ thống trợ lý ảo đa tác nhân VinShop."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        # ĐÃ SỬA: Gọi đúng hàm get_chat_model từ provider
        self.llm = get_chat_model(self.settings)

        # Xây dựng cấu trúc hình học đồ thị LangGraph
        builder = StateGraph(ShoppingState)
        
        # 1. Thêm các Node chức năng vào Đồ thị
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("policy_specialist", worker_1_policy_node)
        builder.add_node("data_specialist", worker_2_data_node)
        builder.add_node("response_specialist", worker_3_response_node)
        
        # 2. Thiết lập điểm xuất phát
        builder.add_edge(START, "supervisor")
        
        # 3. Ráp nối luồng cạnh điều kiện từ điểm Supervisor Node
        builder.add_conditional_edges(
            "supervisor",
            route_decision,
            {
                "response": "response_specialist",
                "policy": "policy_specialist",
                "data": "data_specialist",
                "both": "policy_specialist"
            }
        )
        
        # 4. Hàm định tuyến phụ từ policy sang data khi có nhãn 'both'
        def post_policy_route(state: ShoppingState) -> Literal["data", "response"]:
            route = state.get("route", {})
            if route.get("needs_data", False):
                return "data"
            return "response"

        builder.add_conditional_edges("policy_specialist", post_policy_route, {
            "data": "data_specialist",
            "response": "response_specialist"
        })
        
        builder.add_edge("data_specialist", "response_specialist")
        builder.add_edge("response_specialist", END)
        
        self.graph = builder.compile()

    def ask(
        self,
        question: str,
        trace_file: Path | None = None,
        rebuild_index: bool = False,
    ) -> dict[str, Any]:
        """Thực thi đồ thị (Invoke) và ghi vết Trace lịch sử tư duy hệ thống."""
        initial_state = {"question": question, "trace": []}
        final_output = self.graph.invoke(initial_state)
        
        if trace_file:
            trace_data = final_output.get("trace", [])
            with open(trace_file, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, ensure_ascii=False, indent=2)
                
        return final_output