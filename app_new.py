import streamlit as st
import numpy as np
import pickle

# ----------------------------
# ✅ Load Your Saved Model
# ----------------------------
with open("bestcancer_ml.pkl", "rb") as f:   # NOTE: space before .pkl
    model = pickle.load(f)

# ----------------------------
# ✅ Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# ✅ Custom CSS Styling
# ----------------------------
st.markdown("""
<style>

body {
    background-color: #f5f7fa;
}

.title {
    font-size: 40px;
    text-align: center;
    font-weight: 900;
    color: #2b2d42;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
}

.input-box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}

.result-good {
    padding: 20px;
    border-radius: 15px;
    background: #d4f8e8;
    color: #065f46;
    font-size: 22px;
    text-align: center;
    font-weight: 700;
}

.result-bad {
    padding: 20px;
    border-radius: 15px;
    background: #fde2e4;
    color: #8b0000;
    font-size: 22px;
    text-align: center;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #888;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# ✅ Title Section
# ----------------------------
st.markdown("<h1 class='title'>🩺 Breast Cancer Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter tumor measurements to predict whether cancer is <b>Benign</b> or <b>Malignant</b>.</p>", unsafe_allow_html=True)
st.write("")

# ----------------------------
# ✅ Sidebar Information
# ----------------------------
st.sidebar.header("📌 About This App")
st.sidebar.info("""
This interactive dashboard uses a trained Machine Learning model  
to predict whether a breast tumor is **Malignant (Cancer)** or **Benign**  
based on 30 microscopic features.
""")

st.sidebar.header("⚙ Model Info")
st.sidebar.success("✅ Model Loaded Successfully")

# --------------------------------------------
# ✅ Feature Names (30 Inputs)
# --------------------------------------------
feature_names = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean",
    "fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se",
    "smoothness_se","compactness_se","concavity_se","concave points_se",
    "symmetry_se","fractal_dimension_se","radius_worst","texture_worst",
    "perimeter_worst","area_worst","smoothness_worst","compactness_worst",
    "concavity_worst","concave points_worst","symmetry_worst",
    "fractal_dimension_worst"
]

# --------------------------------------------
# ✅ UI Input Area
# --------------------------------------------
st.markdown("<div class='input-box'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

values = []

for i, feature in enumerate(feature_names):
    if i % 3 == 0:
        val = col1.number_input(feature, value=0.0)
    elif i % 3 == 1:
        val = col2.number_input(feature, value=0.0)
    else:
        val = col3.number_input(feature, value=0.0)

    values.append(val)

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------
# ✅ Prediction Button
# --------------------------------------------
if st.button("🔍 Predict", use_container_width=True):
    data = np.array(values).reshape(1, -1)
    prediction = model.predict(data)[0]

    st.write("")
    if prediction == 1:
        st.markdown("<div class='result-bad'>🔴 Result: Malignant (Cancer Detected)</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-good'>🟢 Result: Benign (No Cancer)</div>", unsafe_allow_html=True)

# --------------------------------------------
# ✅ Footer
# --------------------------------------------
st.markdown("<p class='footer'>Developed with ❤️ using Streamlit & Machine Learning</p>", unsafe_allow_html=True)
