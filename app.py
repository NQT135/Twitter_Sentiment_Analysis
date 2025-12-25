import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦")

# --- HÀM XỬ LÝ TEXT (GIỮ NGUYÊN NHƯ CŨ) ---
def clean_text(text):
    # Đảm bảo logic này GIỐNG HỆT file train
    ps = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    
    # Dùng whitelist như đã bàn
    all_stopwords = stopwords.words('english')
    whitelist = ["n't", "not", "no", "nor"]
    final_stopwords = [word for word in all_stopwords if word not in whitelist]
    
    text = [ps.stem(word) for word in text if not word in set(final_stopwords)]
    text = ' '.join(text)
    return text

# --- HÀM TẢI VECTORIZER & SCALER (CHỈ TẢI 1 LẦN) ---
@st.cache_resource
def load_preprocessors():
    try:
        cv = joblib.load('vectorizer.pkl')
        sc = joblib.load('scaler.pkl')
        return cv, sc
    except FileNotFoundError:
        st.error("Thiếu file vectorizer.pkl hoặc scaler.pkl")
        return None, None

# --- GIAO DIỆN CHÍNH ---
st.title("🐦 Multi-Model Sentiment Analysis")

# 1. Tải bộ xử lý chung
cv, sc = load_preprocessors()

# 2. MENU CHỌN MODEL (SIDEBAR)
st.sidebar.header("🔧 Control Panel")
model_options = {
    "Logistic Regression": "model_LogisticRegression.pkl",
    "Random Forest": "model_RandomForest.pkl",
    "Decision Tree": "model_DecisionTree.pkl",
    "Support Vector Machine (SVM)": "model_SVM.pkl",
    "XGBoost": "model_XGBoost.pkl"
}

# Tạo Dropdown để chọn
selected_model_name = st.sidebar.selectbox("Chọn thuật toán:", list(model_options.keys()))

# Lấy tên file tương ứng
selected_model_file = model_options[selected_model_name]

# 3. TẢI MODEL ĐƯỢC CHỌN
try:
    model = joblib.load(selected_model_file)
    st.sidebar.success(f"Đã tải: {selected_model_name}")
except FileNotFoundError:
    st.error(f"Không tìm thấy file {selected_model_file}. Hãy chạy file train lại!")
    model = None

# --- PHẦN DỰ ĐOÁN ---
st.write(f"Đang sử dụng mô hình: **{selected_model_name}**")
user_input = st.text_area("Nhập nội dung Tweet tại đây:", height=100)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập nội dung!")
    elif model is None or cv is None:
        st.error("Lỗi: Chưa tải được model hoặc bộ xử lý.")
    else:
        # Xử lý
        processed_text = clean_text(user_input)
        vectorized_text = cv.transform([processed_text]).toarray()
        scaled_text = sc.transform(vectorized_text)
        
        # Dự đoán
        prediction = model.predict(scaled_text)
        
        # Hiển thị kết quả
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("Input Text")
            st.write(f"_{user_input}_")
            
        with col2:
            st.info("Prediction Result")
            # Logic hiển thị (Kiểm tra lại dataset của bạn 0 hay 1 là tích cực nhé)
            # Nếu dataset của bạn là 0: Tốt, 1: Xấu (Hate Speech) thì sửa lại dòng dưới
            if prediction[0] == 1:
                st.markdown("### 😡 Negative / Hate")
            else:
                st.markdown("### 😊 Positive / Normal")

# --- THÔNG TIN NHÓM ---
st.sidebar.divider()
st.sidebar.text("Group: [Insert Group No]")