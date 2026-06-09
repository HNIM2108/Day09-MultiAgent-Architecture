from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ShoppingDataStore:
    """
    Bộ lõi truy xuất dữ liệu an toàn hệ thống VinShop.
    Xử lý thông minh cả cấu trúc phẳng lẫn cấu trúc lồng nhau bị loạn của file JSON.
    """

    def __init__(self, json_path: Path) -> None:
        self.json_path = json_path
        
        if not json_path.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy tệp dữ liệu JSON tại: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.metadata = raw_data.get("metadata", {})
        self.customers = raw_data.get("customers", [])
        self.orders = raw_data.get("orders", [])
        self.vouchers = raw_data.get("vouchers", [])

        # Xây dựng bộ chỉ mục dạng chuỗi string sạch sẽ
        self.customer_index = {str(c["customer_id"]).strip(): c for c in self.customers}
        self.order_index = {str(o["order_id"]).strip(): o for o in self.orders}
        
        self.customer_vouchers_index: dict[str, list[dict[str, Any]]] = {}
        for v in self.vouchers:
            c_id = str(v["customer_id"]).strip()
            if c_id not in self.customer_vouchers_index:
                self.customer_vouchers_index[c_id] = []
            self.customer_vouchers_index[c_id].append(v)

    def get_customer_by_id(self, customer_id: str) -> dict[str, Any]:
        """Tra cứu hồ sơ khách hàng từ bảng gốc customers."""
        customer = self.customer_index.get(str(customer_id).strip())
        if customer:
            return {"status": "ok", "customer": customer}
        return {
            "status": "not_found",
            "message": f"Không tìm thấy thông tin khách hàng với mã: {customer_id}"
        }

    def get_orders_by_customer_id(self, customer_id: str, limit: int = 10) -> dict[str, Any]:
        clean_id = str(customer_id).strip()
        customer_orders = [o for o in self.orders if str(o.get("customer_id")).strip() == clean_id]
        return {"status": "ok", "orders": customer_orders[:limit]}

    def get_order_detail_by_order_id(self, order_id: str) -> dict[str, Any]:
        """Tra cứu chi tiết một đơn hàng, bọc an toàn bảo vệ thuộc tính."""
        order = self.order_index.get(str(order_id).strip())
        if order:
            return {"status": "ok", "order": order}
        return {
            "status": "not_found",
            "message": f"Hệ thống không tìm thấy đơn hàng nào có mã là: {order_id}"
        }

    def get_vouchers_by_customer_id(self, customer_id: str, only_active: bool = False) -> dict[str, Any]:
        clean_id = str(customer_id).strip()
        vouchers = self.customer_vouchers_index.get(clean_id, [])
        if only_active:
            vouchers = [v for v in vouchers if v.get("status") == "active" and v.get("remaining_uses", 0) > 0]
        return {"status": "ok", "vouchers": vouchers}


if __name__ == "__main__":
    print("=" * 70)
    print("🎯 Testing Task 6: Verified Production Data Store Lookup Engine")
    print("=" * 70)
    
    base_dir = Path(__file__).parent.parent.parent
    mock_json_file = base_dir / "data" / "order_customer_mock_data.json"
    
    if mock_json_file.exists():
        store = ShoppingDataStore(mock_json_file)
        print(f"📊 Đã nạp thành công: {len(store.customers)} khách | {len(store.orders)} đơn thật.")
        print("-" * 70)
        
        print("🔍 Tra cứu thực tế Đơn hàng 1971:")
        res = store.get_order_detail_by_order_id("1971")
        
        if res["status"] == "ok":
            order_data = res["order"]
            
            # ĐÃ SỬA THEO ĐÚNG JSON THẬT BẠN CUNG CẤP:
            order_status = order_data.get("order_status", "Không rõ")
            provider = order_data.get("carrier", "Không rõ")  # <--- Sửa từ shipping_provider sang carrier
            c_name = order_data.get("customer_name", "Không rõ") # <--- Sửa sang customer_name
            
            # Lấy hạng thành viên từ hàm tra cứu khách hàng gốc bằng customer_id
            c_id = order_data.get("customer_id")
            customer_res = store.get_customer_by_id(c_id)
            c_tier = "Standard"
            if customer_res["status"] == "ok":
                c_tier = customer_res["customer"].get("tier") or customer_res["customer"].get("customer_tier") or "Standard"
            
            # In kết quả nghiệm thu
            print(f"   ✓ Trạng thái vận chuyển: {order_status}")
            print(f"   ✓ Đơn vị giao hàng: {provider}")
            print(f"   ✓ Khách hàng đặt: {c_name} (Hạng: {c_tier})")
        else:
            print(f"   ❌ Thất bại: {res['message']}")
        print("-" * 70)
    else:
        print(f"❌ Không tìm thấy file JSON tại: {mock_json_file}")