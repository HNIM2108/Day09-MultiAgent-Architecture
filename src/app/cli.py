from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.graph import ShoppingAssistant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Student scaffold CLI.")
    parser.add_argument("--question", help="Run one question through the graph.")
    parser.add_argument("--test-file", default="data/test.json")
    parser.add_argument("--trace-file", default=None)
    parser.add_argument("--batch", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assistant = ShoppingAssistant()

    # Thư mục gốc dự án
    base_dir = Path(__file__).parent.parent.parent

    # =========================================================================
    # CHẠY 1 CÂU HỎI ĐƠN LẺ (--question)
    # =========================================================================
    if args.question:
        print("\n" + "=" * 70)
        print(f"🚀 Chạy câu hỏi đơn lẻ: '{args.question}'")
        print("=" * 70)
        
        trace_path = Path(args.trace_file) if args.trace_file else None
        
        # Invoke đồ thị LangGraph
        output = assistant.ask(question=args.question, trace_file=trace_path)
        
        print("\n🎯 [CÂU TRẢ LỜI TỪ VINSHOP ASSISTANT]:")
        print("-" * 70)
        print(output.get("final_answer", "Không có câu trả lời."))
        print("-" * 70)
        
        if args.trace_file:
            print(f"💾 Đã lưu log vết tư duy tại: {args.trace_file}")
        print("=" * 70 + "\n")

    # =========================================================================
    # CHẠY BATCH TEST HÀNG LOẠT (--batch)
    # =========================================================================
    elif args.batch:
        test_file_path = base_dir / args.test_file
        if not test_file_path.exists():
            print(f"❌ Không tìm thấy file testcase tại: {test_file_path}")
            return

        with open(test_file_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        print("\n" + "=" * 80)
        print(f"📊 Bắt đầu Batch Test tự động: Lấy dữ liệu từ {args.test_file}")
        print(f"   Tổng số testcase: {len(test_cases)} câu hỏi.")
        print("=" * 80)

        total_cases = len(test_cases)
        passed_routes = 0
        passed_status = 0
        summary_results = []

        # Tạo thư mục con nếu lưu trace hàng loạt
        trace_dir = base_dir / "data" / "batch_traces"
        if args.trace_file:
            trace_dir.mkdir(parents=True, exist_ok=True)

        for case in test_cases:
            c_id = case.get("id", "N/A")
            q_text = case.get("question", "")
            exp_route = case.get("expected_route", [])
            exp_status = case.get("expected_status", "ok")

            # Chạy qua LangGraph
            single_trace_path = trace_dir / f"{c_id}_trace.json" if args.trace_file else None
            output = assistant.ask(question=q_text, trace_file=single_trace_path)
            
            # Trích xuất kết quả từ Router Agent
            actual_route_info = output.get("route", {})
            act_status = actual_route_info.get("status", "ok")
            
            # ĐỒNG BỘ TRẠNG THÁI HỆ THỐNG: Nếu database JSON báo không tìm thấy bản ghi, đồng bộ act_status về 'not_found'
            data_res = output.get("data_result", {})
            if data_res and data_res.get("status") == "not_found":
                act_status = "not_found"

            # Ánh xạ kết quả thực tế thành mảng nhãn
            act_route = []
            if actual_route_info.get("needs_policy"):
                act_route.append("policy")
            if actual_route_info.get("needs_data"):
                act_route.append("data")

            # So khớp chuẩn với barem trường sau khi đã chuẩn hóa trạng thái liên tầng
            route_match = sorted(act_route) == sorted(exp_route)
            status_match = act_status == exp_status

            if route_match:
                passed_routes += 1
            if status_match:
                passed_status += 1

            case_passed = route_match and status_match
            status_icon = "✅ PASS" if case_passed else "❌ FAIL"
            
            print(f"[{c_id}] {status_icon} | Mong muốn: {exp_route} (Status: {exp_status}) -> Thực tế: {act_route} (Status: {act_status})")

            summary_results.append({
                "id": c_id,
                "question": q_text,
                "expected": {"route": exp_route, "status": exp_status},
                "actual": {"route": act_route, "status": act_status},
                "passed": case_passed
            })

        # Lưu file summary.json theo đúng Guide yêu cầu
        summary_path = base_dir / "data" / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": {
                    "total": total_cases,
                    "route_accuracy": passed_routes / total_cases,
                    "status_accuracy": passed_status / total_cases
                },
                "results": summary_results
            }, f, ensure_ascii=False, indent=2)

        # In bảng tổng kết Metrics
        print("-" * 80)
        print("📈 KẾT QUẢ ĐÁNH GIÁ (ACCURACY METRICS):")
        print(f"   ✓ Route Accuracy:  {passed_routes}/{total_cases} ({passed_routes/total_cases*100:.1f}%)")
        print(f"   ✓ Status Accuracy: {passed_status}/{total_cases} ({passed_status/total_cases*100:.1f}%)")
        print(f"📝 Đã ghi file tổng hợp tại: data/summary.json")
        print("=" * 80 + "\n")
        
    else:
        print("💡 Hướng dẫn: Thêm cờ `--question \"nội dung\"` hoặc `--batch` để kích hoạt.")


if __name__ == "__main__":
    main()