import streamlit as st
import numpy as np
import pickle
import joblib
import requests
from datetime import datetime
import pytz

# ==== CẤU HÌNH FIREBASE ====
FIREBASE_URL = "https://bai-test-2ae56-default-rtdb.asia-southeast1.firebasedatabase.app"

def push_to_firebase(path, data):
    """
    Ghi một object `data` vào đường dẫn `path` trên Firebase RTDB
    """
    url = f"{FIREBASE_URL}/{path}.json"
    try:
        resp = requests.post(url, json=data)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"Không lưu được dữ liệu lên Firebase: {e}")

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

# ==== GIAO DIỆN STREAMLIT ====
st.title("🏥 Chuẩn Đoán Bệnh Bằng Machine Learning")
st.markdown("Vui lòng nhập thông tin để nhận kết quả chẩn đoán và lưu lại lịch sử.")

# Nhập tên
user_name = st.text_input("Tên người dùng:")

# Chọn loại chẩn đoán
diagnosis_type = st.selectbox(
    "Chọn loại chẩn đoán:",
    ["-- Chọn --", "Kiểm tra tim mạch", "Chuẩn đoán trầm cảm", "Chuẩn đoán bệnh béo phì"]
)
if diagnosis_type == "-- Chọn --":
    st.stop()

# Lấy timestamp theo timezone Asia/Bangkok
tz = pytz.timezone("Asia/Bangkok")
timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

# ===== PHẦN TIM MẠCH =====
if diagnosis_type == "Kiểm tra tim mạch":
    st.subheader("❤️ Thông số Tim Mạch")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Tuổi:", min_value=1, step=1)
        gender_str = st.selectbox("Giới tính:", ["-- Chọn --", "Nam (0)", "Nữ (1)"])
        chest_str = st.selectbox("Đau ngực:", ["-- Chọn --", "Typical angina (1)", "Asymptomatic (0)",
                                               "Non-anginal pain (3)", "Atypical angina (2)"])
        bp = st.number_input("Huyết áp:", min_value=1, step=1)
    with col2:
        chol = st.number_input("Cholesterol:", min_value=1, step=1)
        hr = st.number_input("Nhịp tim:", min_value=1, step=1)
        thal_str = st.selectbox("Thalassemia:", ["-- Chọn --", "Bình thường (3)",
                                                 "Khiếm khuyết cố định (6)", "Khuyết có thể đảo ngược (7)"])
    if st.button("Chuẩn đoán Tim Mạch"):
        if "-- Chọn --" in [gender_str, chest_str, thal_str]:
            st.error("Vui lòng chọn đầy đủ thông tin!")
        else:
            features = [
                age,
                0 if "Nam" in gender_str else 1,
                int(chest_str.split("(")[1].rstrip(")")),
                bp,
                chol,
                hr,
                int(thal_str.split("(")[1].rstrip(")"))
            ]
            result = predict_heart(features)
            st.success(f"{user_name}: {result}")

            # Ghi vào Firebase
            data = {
                "user_name": user_name,
                "type": "heart",
                "inputs": {
                    "age": age, "gender": gender_str, "chest_pain": chest_str,
                    "blood_pressure": bp, "cholesterol": chol, "heartbeat": hr,
                    "thalassemia": thal_str
                },
                "result": result,
                "timestamp": timestamp
            }
            push_to_firebase("diagnoses", data)

# ===== PHẦN TRẦM CẢM =====
elif diagnosis_type == "Chuẩn đoán trầm cảm":
    st.subheader("🧠 Thông số Trầm Cảm")
    col1, col2 = st.columns(2)
    with col1:
        dg = st.selectbox("Giới tính:", ["-- Chọn --", "Nam (1)", "Nữ (0)"])
        age_d = st.number_input("Tuổi:", min_value=1, step=1)
        stress_study = st.slider("Áp lực học tập (0-5):", 0, 5, 0)
        cgpa = st.number_input("Điểm trung bình (0.0-10.0):", min_value=0.0, max_value=10.0, step=0.01)
        satisfaction = st.slider("Mức độ hài lòng (0-5):", 0, 5, 0)
        sleep_str = st.selectbox("Giờ ngủ:", ["-- Chọn --", "Dưới 5 giờ (1)", "5-6 giờ (2)", "7-8 giờ (3)", "Trên 8 giờ (4)"])
    with col2:
        diet_str = st.selectbox("Thói quen ăn uống:", ["-- Chọn --", "Không lành mạnh (1)", "Trung bình (2)", "Lành mạnh (3)"])
        suicide_str = st.selectbox("Từng nghĩ tự tử?", ["-- Chọn --", "Không (0)", "Có (1)"])
        hours_study = st.number_input("Giờ học/ngày:", min_value=1, step=1)
        stress_fin = st.slider("Áp lực tài chính (0-5):", 0, 5, 0)
        fh_str = st.selectbox("Tiền sử bệnh tâm thần gia đình:", ["-- Chọn --", "Không (0)", "Có (1)"])
    if st.button("Chuẩn đoán Trầm Cảm"):
        if "-- Chọn --" in [dg, sleep_str, diet_str, suicide_str, fh_str]:
            st.error("Vui lòng chọn đầy đủ thông tin!")
        else:
            features = [
                1 if "Nam" in dg else 0,
                age_d, stress_study, cgpa, satisfaction,
                int(sleep_str.split("(")[1].rstrip(")")),
                int(diet_str.split("(")[1].rstrip(")")),
                int(suicide_str.split("(")[1].rstrip(")")),
                hours_study, stress_fin,
                int(fh_str.split("(")[1].rstrip(")"))
            ]
            result = predict_depression(features)
            st.success(f"{user_name}: {result}")

            # Ghi vào Firebase
            data = {
                "user_name": user_name,
                "type": "depression",
                "inputs": {
                    "gender": dg, "age": age_d, "study_pressure": stress_study,
                    "cgpa": cgpa, "satisfaction": satisfaction, "sleep": sleep_str,
                    "diet": diet_str, "suicide_thoughts": suicide_str,
                    "study_hours": hours_study, "financial_pressure": stress_fin,
                    "family_history": fh_str
                },
                "result": result,
                "timestamp": timestamp
            }
            push_to_firebase("diagnoses", data)

# ===== PHẦN BÉO PHÌ =====
elif diagnosis_type == "Chuẩn đoán bệnh béo phì":
    st.subheader("⚖️ Thông số Béo Phì")
    col1, col2 = st.columns(2)
    with col1:
        og = st.selectbox("Giới tính:", ["-- Chọn --", "Nam (1)", "Nữ (0)"])
        age_o = st.number_input("Tuổi:", min_value=1, step=1)
        height = st.number_input("Chiều cao (m):", min_value=0.5, max_value=2.5, step=0.01)
        weight = st.number_input("Cân nặng (kg):", min_value=1.0, step=0.1)
        fh_o = st.selectbox("Gia đình có thừa cân?", ["-- Chọn --", "Yes (1)", "No (0)"])
        favc = st.selectbox("Tiêu thụ thực phẩm giàu calo?", ["-- Chọn --", "Yes (1)", "No (0)"])
        fcvc = st.selectbox("Ăn rau:", ["-- Chọn --", "Ăn ít (0)", "Ăn đủ (1)", "Ăn nhiều (2)"])
        ncp = st.selectbox("Số bữa chính/ngày:", ["1 (1)", "2 (2)", "3 (3)", "4+ (4)"])
    with col2:
        caec = st.selectbox("Ăn vặt:", ["Không (0)", "Thi thoảng (1)", "Thường xuyên (2)", "Luôn (3)"])
        smoke = st.selectbox("Hút thuốc?", ["Yes (1)", "No (0)"])
        water = st.number_input("Nước uống (lít):", min_value=0.1, step=0.1)
        scc = st.selectbox("Theo dõi calo?", ["Yes (1)", "No (0)"])
        faf = st.selectbox("Hoạt động thể chất:", ["Không (0)", "Thấp (1)", "Bình thường (2)", "Cao (3)"])
        tue = st.selectbox("Giờ dùng thiết bị:", ["Thấp (0)", "Trung bình (1)", "Cao (2)"])
        calc = st.selectbox("Tiêu thụ rượu:", ["Không (0)", "Thi thoảng (1)", "Thường xuyên (2)", "Luôn (3)"])
        mtrans = st.selectbox("Phương tiện chính:", ["Public (0)", "Automobile (1)", "Walking (2)", "Motorbike (3)", "Bike (4)"])
    if st.button("Chuẩn đoán Béo Phì"):
        selects = [og, fh_o, favc, fcvc, ncp, caec, smoke, scc, faf, tue, calc, mtrans]
        if any("-- Chọn --" in s for s in selects):
            st.error("Vui lòng chọn đầy đủ thông tin!")
        else:
            features = [
                1 if "Nam" in og else 0,
                age_o, height, weight,
                int(fh_o.split("(")[1].rstrip(")")),
                int(favc.split("(")[1].rstrip(")")),
                int(fcvc.split("(")[1].rstrip(")")),
                int(ncp.split("(")[1].rstrip(")")), 
                int(caec.split("(")[1].rstrip(")")), 
                int(smoke.split("(")[1].rstrip(")")),
                water,
                int(scc.split("(")[1].rstrip(")")),
                int(faf.split("(")[1].rstrip(")")), 
                int(tue.split("(")[1].rstrip(")")), 
                int(calc.split("(")[1].rstrip(")")), 
                int(mtrans.split("(")[1].rstrip(")"))
            ]
            result = predict_obesity(features)
            st.success(f"{user_name}: {result}")

            # Ghi vào Firebase
            data = {
                "user_name": user_name,
                "type": "obesity",
                "inputs": {
                    "gender": og, "age": age_o, "height": height, "weight": weight,
                    "family_history": fh_o, "caloric_food": favc, "veg_intake": fcvc,
                    "meals_per_day": ncp, "snacking": caec, "smoking": smoke,
                    "water_liter": water, "track_calories": scc,
                    "activity": faf, "device_time": tue, "alcohol": calc, "transport": mtrans
                },
                "result": result,
                "timestamp": timestamp
            }
            push_to_firebase("diagnoses", data)
