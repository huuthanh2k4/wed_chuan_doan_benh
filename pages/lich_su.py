import streamlit as st
import requests

# ==== CẤU HÌNH FIREBASE ====
FIREBASE_URL = "https://bai-test-2ae56-default-rtdb.asia-southeast1.firebasedatabase.app"

def fetch_diagnosis_history():
    """
    Lấy lịch sử chẩn đoán từ Firebase RTDB
    """
    url = f"{FIREBASE_URL}/diagnoses.json"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        return data if data else {}
    except Exception as e:
        st.error(f"Không thể tải lịch sử chẩn đoán: {e}")
        return {}

# ==== GIAO DIỆN STREAMLIT ====
st.title("📜 Lịch Sử Chẩn Đoán")
st.markdown("Xem lại lịch sử chẩn đoán và thời gian thực hiện.")

# Lấy dữ liệu từ Firebase
history = fetch_diagnosis_history()

if not history:
    st.info("Không có lịch sử chẩn đoán nào.")
else:
    for key, record in history.items():
        st.subheader(f"Người dùng: {record.get('user_name', 'Không xác định')}")
        st.write(f"**Loại chẩn đoán:** {record.get('type', 'Không xác định')}")
        st.write(f"**Kết quả:** {record.get('result', 'Không xác định')}")
        st.write(f"**Thời gian:** {record.get('timestamp', 'Không xác định')}")
        st.write("---")