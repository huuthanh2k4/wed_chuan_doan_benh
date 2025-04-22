import pickle
import joblib
import numpy as np
import streamlit as st
import requests
import json
from datetime import datetime
import pytz
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ==== TẢI MODEL ====
@st.cache_resource
def load_models():
    with open('Model/ML_heartattack.sav', 'rb') as f:
        heart_model = pickle.load(f)
    knn_depression = joblib.load('Model/CDTC_knn.sav')
    with open('Model/NutriAI.sav', 'rb') as f:
        obesity_model, scaler = pickle.load(f)
    return heart_model, knn_depression, obesity_model, scaler

heart_model, knn_depression, obesity_model, scaler = load_models()

# ==== HÀM DỰ ĐOÁN + MESSAGE ====
def predict_heart(features):
    pred = heart_model.predict(np.array(features).reshape(1, -1))[0]
    if pred == 0:
        return "Chúc mừng bạn không có nguy cơ mắc bệnh tim mạch"
    else:
        return "Bạn có nguy cơ cao mắc bệnh tim mạch, hãy đi khám ngay!"

def predict_depression(features):
    pred = knn_depression.predict(np.array(features).reshape(1, -1))[0]
    if pred == 0:
        return "Chúc mừng bạn không có nguy cơ trầm cảm"
    else:
        return "Bạn có nguy cơ trầm cảm, hãy đi gặp chuyên gia tâm lý!"

def predict_obesity(features):
    scaled = scaler.transform(np.array(features).reshape(1, -1))
    pred = obesity_model.predict(scaled)[0]
    messages = {
        0: " Thiếu cân",
        1: " Cân nặng bình thường",
        2: " Thừa cân cấp độ I",
        3: " Thừa cân cấp độ II",
        4: " Béo phì loại I",
        5: " Béo phì loại II",
        6: " Béo phì loại III"
    }
    return messages.get(pred, "Kết quả không xác định")


# Firebase configuration
FIREBASE_URL = "https://bai-test-2ae56-default-rtdb.asia-southeast1.firebasedatabase.app"

# Helper functions for Firebase operations
def get_from_firebase(path):
    url = f"{FIREBASE_URL}/{path}.json"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Không thể lấy dữ liệu từ Firebase: {e}")
        return None

def push_to_firebase(path, data):
    url = f"{FIREBASE_URL}/{path}.json"
    try:
        resp = requests.post(url, json=data)
        resp.raise_for_status()
        return resp.json()  # Trả về key của bản ghi mới
    except Exception as e:
        st.error(f"Không lưu được dữ liệu lên Firebase: {e}")
        return None

def update_in_firebase(path, data):
    url = f"{FIREBASE_URL}/{path}.json"
    try:
        resp = requests.patch(url, json=data)  # Sử dụng PATCH để cập nhật một phần
        resp.raise_for_status()
    except Exception as e:
        st.error(f"Không cập nhật được dữ liệu trên Firebase: {e}")

def delete_from_firebase(path):
    url = f"{FIREBASE_URL}/{path}.json"
    try:
        resp = requests.delete(url)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"Không xóa được dữ liệu trên Firebase: {e}")

# Get client IP address using ipify
def get_client_ip():
    try:
        resp = requests.get("https://api.ipify.org?format=json")
        resp.raise_for_status()
        return resp.json()["ip"]
    except Exception as e:
        st.error(f"Không thể lấy địa chỉ IP: {e}")
        return "unknown"

def main():
    st.title("🛠️ Trang Quản Lý (Admin)")

    # Get timestamp for logging
    tz = pytz.timezone("Asia/Bangkok")
    timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    # Get client IP and create a valid key for Firebase
    client_ip = get_client_ip()
    ip_key = client_ip.replace('.', '_')  # Thay thế '.' thành '_' để hợp lệ với Firebase key

    # Kiểm tra và lưu IP vào Firebase
    ip_data = get_from_firebase(f"ips/{ip_key}")
    if ip_data is None:
        # IP chưa tồn tại, tạo mới với quyền là 0 và lưu thời gian truy cập
        ip_data = {
            "role": 0,
            "access_times": [timestamp]
        }
        update_in_firebase(f"ips/{ip_key}", ip_data)
    else:
        # IP đã tồn tại, thêm thời gian truy cập mới vào danh sách
        if "access_times" not in ip_data:
            ip_data["access_times"] = []
        ip_data["access_times"].append(timestamp)
        update_in_firebase(f"ips/{ip_key}", {"access_times": ip_data["access_times"]})

    # Lấy quyền của người dùng từ Firebase
    user_role = ip_data["role"]

    # Role-based access control
    if user_role == 0:
        st.warning("Bạn không có quyền truy cập trang này. Chuyển hướng đến trang lịch sử...")
        st.markdown("### Trang Lịch Sử Chẩn Đoán")
        diagnoses = get_from_firebase("diagnoses")
        if diagnoses:
            for diag_id, diag in diagnoses.items():
                st.write(f"**{diag['user_name']}** ({diag['type']}): {diag['result']} - {diag['timestamp']}")
        return

    # Admin interface (role=1)
    st.subheader("Quản Lý Thông Tin Bệnh Nhân")

    # Fetch all diagnoses
    diagnoses = get_from_firebase("diagnoses")
    if not diagnoses:
        st.info("Không có dữ liệu chẩn đoán nào.")
        return

    # Display diagnoses
    for diag_id, diag in diagnoses.items():
        with st.expander(f"{diag['user_name']} - {diag['type']} - {diag['timestamp']}"):
            st.write("**Thông tin chi tiết:**")
            inputs_df = pd.DataFrame([diag["inputs"]])
            st.table(inputs_df)
            st.write(f"**Kết quả:** {diag['result']}")

            # Edit functionality
            if st.button("Sửa", key=f"edit_{diag_id}"):
                st.session_state.editing_id = diag_id
                st.session_state.editing_data = diag

            # Delete functionality
            if st.button("Xóa", key=f"delete_{diag_id}"):
                delete_from_firebase(f"diagnoses/{diag_id}")
                st.success("Đã xóa chẩn đoán.")
                st.rerun()

    # Editing form
    if "editing_id" in st.session_state:
        st.subheader("Chỉnh Sửa Chẩn Đoán")
        editing_id = st.session_state.editing_id
        editing_data = st.session_state.editing_data

        user_name = st.text_input("Tên người dùng:", value=editing_data["user_name"])
        result = st.text_input("Kết quả:", value=editing_data["result"])

        if editing_data["type"] == "heart":
            inputs = edit_heart_form(editing_data["inputs"])
        elif editing_data["type"] == "depression":
            inputs = edit_depression_form(editing_data["inputs"])
        elif editing_data["type"] == "obesity":
            inputs = edit_obesity_form(editing_data["inputs"])

        if st.button("Lưu thay đổi"):
            # Tính lại kết quả chuẩn đoán theo loại chẩn đoán
            if editing_data["type"] == "heart":
                # Tạo features cho tim mạch (chú ý thứ tự theo mô hình ban đầu)
                features = [
                    inputs["age"],
                    1 if "Nam" in inputs["gender"] else 0,
                    int(inputs["chest_pain"].split("(")[1].rstrip(")")),
                    inputs["blood_pressure"],
                    inputs["cholesterol"],
                    inputs["heartbeat"],
                    int(inputs["thalassemia"].split("(")[1].rstrip(")"))
                ]
                new_result = predict_heart(features)
            elif editing_data["type"] == "depression":
                # Tạo features cho trầm cảm (đảm bảo thứ tự như khi dự đoán ban đầu)
                features = [
                    1 if "Nam" in inputs["gender"] else 0,
                    inputs["age"],
                    inputs["study_pressure"],
                    inputs["cgpa"],
                    inputs["satisfaction"],
                    # Giả sử 'sleep' và 'diet' cần chuyển đổi nếu cần,
                    int(inputs["sleep"].split("(")[1].rstrip(")")),
                    int(inputs["diet"].split("(")[1].rstrip(")")),
                    1 if inputs["suicide_thoughts"] == "Có (1)" else 0,
                    inputs["study_hours"],
                    inputs["financial_pressure"],
                    1 if inputs["family_history"] == "Có (1)" else 0
                ]
                new_result = predict_depression(features)
            elif editing_data["type"] == "obesity":
                # Tạo features cho béo phì (chuyển đổi các giá trị theo định dạng ban đầu)
                features = [
                    1 if "Nam" in inputs["gender"] else 0,                              # feature 1: Giới tính
                    inputs["age"],                                                       # feature 2: Tuổi
                    inputs["height"],                                                    # feature 3: Chiều cao
                    inputs["weight"],                                                    # feature 4: Cân nặng
                    int(inputs["family_history"].split("(")[1].rstrip(")")),               # feature 5: Gia đình có thừa cân?
                    int(inputs["caloric_food"].split("(")[1].rstrip(")")),                 # feature 6: Tiêu thụ thực phẩm giàu calo?
                    int(inputs["veg_intake"].split("(")[1].rstrip(")")),                   # feature 7: Ăn rau
                    int(inputs["meals_per_day"].split("(")[1].rstrip(")")),                # feature 8: Số bữa chính/ngày
                    int(inputs["snacking"].split("(")[1].rstrip(")")),                     # feature 9: Ăn vặt
                    int(inputs["smoking"].split("(")[1].rstrip(")")),                      # feature 10: Hút thuốc?
                    inputs["water_liter"],                                                 # feature 11: Nước uống (lít)
                    int(inputs["track_calories"].split("(")[1].rstrip(")")),               # feature 12: Theo dõi calo?
                    int(inputs["activity"].split("(")[1].rstrip(")")),                     # feature 13: Hoạt động thể chất
                    int(inputs["device_time"].split("(")[1].rstrip(")")),                  # feature 14: Giờ dùng thiết bị
                    int(inputs["alcohol"].split("(")[1].rstrip(")")),                      # feature 15: Tiêu thụ rượu
                    int(inputs["transport"].split("(")[1].rstrip(")"))                     # feature 16: Phương tiện chính
                ]
                new_result = predict_obesity(features)
            else:
                new_result = "Kết quả không xác định"

            updated_data = {
                "user_name": user_name,
                "type": editing_data["type"],
                "inputs": inputs,
                "result": new_result,
                "timestamp": editing_data["timestamp"]
            }
            update_in_firebase(f"diagnoses/{editing_id}", updated_data)
            st.success("Đã cập nhật chẩn đoán.")
            del st.session_state.editing_id
            del st.session_state.editing_data
            st.rerun()

# Các hàm edit form (giữ nguyên)
def edit_heart_form(inputs):
    st.subheader("Chỉnh sửa thông tin tim mạch")
    age = st.number_input("Tuổi:", min_value=1, step=1, value=inputs["age"])
    gender = st.selectbox("Giới tính:", ["Nam (0)", "Nữ (1)"], index=0 if inputs["gender"] == "Nam (0)" else 1)
    chest = st.selectbox("Đau ngực:", ["Typical angina (1)", "Asymptomatic (0)", "Non-anginal pain (3)", "Atypical angina (2)"], index=["Typical angina (1)", "Asymptomatic (0)", "Non-anginal pain (3)", "Atypical angina (2)"].index(inputs["chest_pain"]))
    bp = st.number_input("Huyết áp:", min_value=1, step=1, value=inputs["blood_pressure"])
    chol = st.number_input("Cholesterol:", min_value=1, step=1, value=inputs["cholesterol"])
    hr = st.number_input("Nhịp tim:", min_value=1, step=1, value=inputs["heartbeat"])
    thal = st.selectbox("Thalassemia:", ["Bình thường (3)", "Khiếm khuyết cố định (6)", "Khuyết có thể đảo ngược (7)"], index=["Bình thường (3)", "Khiếm khuyết cố định (6)", "Khuyết có thể đảo ngược (7)"].index(inputs["thalassemia"]))
    return {
        "age": age,
        "gender": gender,
        "chest_pain": chest,
        "blood_pressure": bp,
        "cholesterol": chol,
        "heartbeat": hr,
        "thalassemia": thal
    }

def edit_depression_form(inputs):
    st.subheader("Chỉnh sửa thông tin trầm cảm")
    gender = st.selectbox("Giới tính:", ["Nam (1)", "Nữ (0)"], index=0 if inputs["gender"] == "Nam (1)" else 1)
    age = st.number_input("Tuổi:", min_value=1, step=1, value=inputs["age"])
    stress_study = st.slider("Áp lực học tập (0-5):", 0, 5, inputs["study_pressure"])
    cgpa = st.number_input("Điểm trung bình (0.0-10.0):", min_value=0.0, max_value=10.0, step=0.01, value=inputs["cgpa"])
    satisfaction = st.slider("Mức độ hài lòng (0-5):", 0, 5, inputs["satisfaction"])
    sleep = st.selectbox("Giờ ngủ:", ["Dưới 5 giờ (1)", "5-6 giờ (2)", "7-8 giờ (3)", "Trên 8 giờ (4)"], index=["Dưới 5 giờ (1)", "5-6 giờ (2)", "7-8 giờ (3)", "Trên 8 giờ (4)"].index(inputs["sleep"]))
    diet = st.selectbox("Thói quen ăn uống:", ["Không lành mạnh (1)", "Trung bình (2)", "Lành mạnh (3)"], index=["Không lành mạnh (1)", "Trung bình (2)", "Lành mạnh (3)"].index(inputs["diet"]))
    suicide = st.selectbox("Từng nghĩ tự tử?", ["Không (0)", "Có (1)"], index=0 if inputs["suicide_thoughts"] == "Không (0)" else 1)
    hours_study = st.number_input("Giờ học/ngày:", min_value=1, step=1, value=inputs["study_hours"])
    stress_fin = st.slider("Áp lực tài chính (0-5):", 0, 5, inputs["financial_pressure"])
    fh = st.selectbox("Tiền sử bệnh tâm thần gia đình:", ["Không (0)", "Có (1)"], index=0 if inputs["family_history"] == "Không (0)" else 1)
    return {
        "gender": gender,
        "age": age,
        "study_pressure": stress_study,
        "cgpa": cgpa,
        "satisfaction": satisfaction,
        "sleep": sleep,
        "diet": diet,
        "suicide_thoughts": suicide,
        "study_hours": hours_study,
        "financial_pressure": stress_fin,
        "family_history": fh
    }

def edit_obesity_form(inputs):
    st.subheader("Chỉnh sửa thông tin béo phì")
    gender = st.selectbox("Giới tính:", ["Nam (1)", "Nữ (0)"], index=0 if inputs["gender"] == "Nam (1)" else 1)
    age = st.number_input("Tuổi:", min_value=1, step=1, value=inputs["age"])
    height = st.number_input("Chiều cao (m):", min_value=0.5, max_value=2.5, step=0.01, value=inputs["height"])
    weight = st.number_input("Cân nặng (kg):", min_value=1.0, step=0.1, value=inputs["weight"])
    fh = st.selectbox("Gia đình có thừa cân?", ["Yes (1)", "No (0)"], index=0 if inputs["family_history"] == "Yes (1)" else 1)
    favc = st.selectbox("Tiêu thụ thực phẩm giàu calo?", ["Yes (1)", "No (0)"], index=0 if inputs["caloric_food"] == "Yes (1)" else 1)
    fcvc = st.selectbox("Ăn rau:", ["Ăn ít (0)", "Ăn đủ (1)", "Ăn nhiều (2)"], index=["Ăn ít (0)", "Ăn đủ (1)", "Ăn nhiều (2)"].index(inputs["veg_intake"]))
    ncp = st.selectbox("Số bữa chính/ngày:", ["1 (1)", "2 (2)", "3 (3)", "4+ (4)"], index=["1 (1)", "2 (2)", "3 (3)", "4+ (4)"].index(inputs["meals_per_day"]))
    caec = st.selectbox("Ăn vặt:", ["Không (0)", "Thi thoảng (1)", "Thường xuyên (2)", "Luôn (3)"], index=["Không (0)", "Thi thoảng (1)", "Thường xuyên (2)", "Luôn (3)"].index(inputs["snacking"]))
    smoke = st.selectbox("Hút thuốc?", ["Yes (1)", "No (0)"], index=0 if inputs["smoking"] == "Yes (1)" else 1)
    water = st.number_input("Nước uống (lít):", min_value=0.1, step=0.1, value=inputs["water_liter"])
    scc = st.selectbox("Theo dõi calo?", ["Yes (1)", "No (0)"], index=0 if inputs["track_calories"] == "Yes (1)" else 1)
    faf = st.selectbox("Hoạt động thể chất:", ["Không (0)", "Thấp (1)", "Bình thường (2)", "Cao (3)"], index=["Không (0)", "Thấp (1)", "Bình thường (2)", "Cao (3)"].index(inputs["activity"]))
    tue = st.selectbox("Giờ dùng thiết bị:", ["Thấp (0)", "Trung bình (1)", "Cao (2)"], index=["Thấp (0)", "Trung bình (1)", "Cao (2)"].index(inputs["device_time"]))
    calc = st.selectbox("Tiêu thụ rượu:", ["Không (0)", "Thi thoảng (1)", "Thường xuyên (2)", "Luôn (3)"], index=["Không (0)", "Thi thoảng (1)", "Thường xuyên (2)", "Luôn (3)"].index(inputs["alcohol"]))
    mtrans = st.selectbox("Phương tiện chính:", ["Public (0)", "Automobile (1)", "Walking (2)", "Motorbike (3)", "Bike (4)"], index=["Public (0)", "Automobile (1)", "Walking (2)", "Motorbike (3)", "Bike (4)"].index(inputs["transport"]))
    return {
        "gender": gender,
        "age": age,
        "height": height,
        "weight": weight,
        "family_history": fh,
        "caloric_food": favc,
        "veg_intake": fcvc,
        "meals_per_day": ncp,
        "snacking": caec,
        "smoking": smoke,          # thêm trường này
        "water_liter": water,
        "track_calories": scc,
        "activity": faf,
        "device_time": tue,
        "alcohol": calc,
        "transport": mtrans
    }

if __name__ == "__main__":
    main()