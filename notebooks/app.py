import os
import pandas as pd
import joblib
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Survival Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}

    /* Page background */
    .stApp { background-color: #f8f9fb; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e8ecf0;
    }

    /* Section label inside sidebar */
    .sidebar-section {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8a94a6;
        margin: 1.1rem 0 0.35rem 0;
    }

    /* Top hero banner */
    .hero {
        background: linear-gradient(135deg, #1a1f5e 0%, #3a4db7 100%);
        border-radius: 14px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { font-size: 1.9rem; font-weight: 800; margin: 0 0 0.3rem 0; }
    .hero p  { font-size: 0.95rem; opacity: 0.85; margin: 0; }

    /* Stat pill */
    .stat-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
    .stat-pill {
        background: white;
        border: 1px solid #e3e8f0;
        border-radius: 10px;
        padding: 0.75rem 1.2rem;
        flex: 1;
        text-align: center;
    }
    .stat-pill .val { font-size: 1.35rem; font-weight: 800; color: #1a1f5e; }
    .stat-pill .lbl { font-size: 0.72rem; color: #8a94a6; text-transform: uppercase;
                      letter-spacing: 0.06em; }

    /* Result card — Living */
    .result-living {
        background: linear-gradient(135deg, #0f9b58 0%, #17c675 100%);
        border-radius: 14px;
        padding: 2rem 2.5rem;
        color: white;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    /* Result card — Deceased */
    .result-deceased {
        background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%);
        border-radius: 14px;
        padding: 2rem 2.5rem;
        color: white;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .result-icon  { font-size: 3rem; margin-bottom: 0.4rem; }
    .result-label { font-size: 2rem; font-weight: 900; margin: 0; letter-spacing: 0.02em; }
    .result-sub   { font-size: 0.9rem; opacity: 0.85; margin: 0.3rem 0 0 0; }

    /* Probability bar cards */
    .prob-card {
        background: white;
        border: 1px solid #e3e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
    }
    .prob-label { font-size: 0.8rem; font-weight: 700; color: #555e6e;
                  text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem; }
    .prob-val   { font-size: 1.6rem; font-weight: 900; margin-bottom: 0.6rem; }
    .prob-living   .prob-val { color: #0f9b58; }
    .prob-deceased .prob-val { color: #ef4444; }

    /* Instruction card */
    .instruction-card {
        background: white;
        border: 1px solid #e3e8f0;
        border-radius: 14px;
        padding: 2.5rem;
        text-align: center;
        color: #8a94a6;
    }
    .instruction-card .icon { font-size: 2.5rem; margin-bottom: 0.6rem; }
    .instruction-card p { margin: 0; font-size: 0.95rem; }

    /* Disclaimer */
    .disclaimer {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        font-size: 0.82rem;
        color: #92400e;
        margin-top: 1.5rem;
    }

    /* Predict button */
    div[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1a1f5e, #3a4db7);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem;
        font-size: 1rem;
        font-weight: 700;
        margin-top: 0.8rem;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    div[data-testid="stSidebar"] .stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model  = joblib.load(os.path.join(BASE_DIR, "models", "breast_cancer_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

FEATURE_NAMES = [
    "age_at_diagnosis", "tumor_size", "neoplasm_histologic_grade",
    "lymph_nodes_examined_positive", "mutation_count", "nottingham_prognostic_index",
    "er_status", "her2_status", "pr_status", "chemotherapy",
    "hormone_therapy", "radio_therapy", "type_of_breast_surgery", "inferred_menopausal_state"
]

# ── Sidebar — inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Patient Data")
    st.caption("Fill in the clinical details and press **Predict**.")

    # — Clinical Measurements —
    st.markdown('<p class="sidebar-section">Clinical Measurements</p>', unsafe_allow_html=True)
    age_at_diagnosis              = st.number_input("Age at Diagnosis",             20.0, 100.0, 60.0, step=0.5)
    tumor_size                    = st.number_input("Tumor Size (mm)",               0.0, 200.0, 25.0, step=0.5)
    neoplasm_histologic_grade     = st.selectbox(   "Histologic Grade",             [1, 2, 3], index=1)
    lymph_nodes_examined_positive = st.number_input("Positive Lymph Nodes",         0,   50,   0)
    mutation_count                = st.number_input("Mutation Count",               0,  200,   2)
    nottingham_prognostic_index   = st.number_input("Nottingham Prognostic Index",  0.0, 10.0,  4.0, step=0.01)

    # — Receptor Status —
    st.markdown('<p class="sidebar-section">Receptor Status</p>', unsafe_allow_html=True)
    er_status   = st.selectbox("ER Status",   ["Positive", "Negative"])
    her2_status = st.selectbox("HER2 Status", ["Negative", "Positive"])
    pr_status   = st.selectbox("PR Status",   ["Positive", "Negative"])

    # — Treatment History —
    st.markdown('<p class="sidebar-section">Treatment History</p>', unsafe_allow_html=True)
    chemotherapy    = st.selectbox("Chemotherapy",    ["No", "Yes"])
    hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])
    radio_therapy   = st.selectbox("Radio Therapy",   ["Yes", "No"])

    # — Patient Profile —
    st.markdown('<p class="sidebar-section">Patient Profile</p>', unsafe_allow_html=True)
    surgery_type     = st.selectbox("Type of Surgery",       ["Breast Conserving", "Mastectomy"])
    menopausal_state = st.selectbox("Menopausal State",      ["Post", "Pre"])

    st.markdown("---")
    predict_clicked = st.button("🔍 Predict Survival")

# ── Build input DataFrame ─────────────────────────────────────────────────────
input_data = pd.DataFrame([[
    age_at_diagnosis,
    tumor_size,
    float(neoplasm_histologic_grade),
    float(lymph_nodes_examined_positive),
    float(mutation_count),
    nottingham_prognostic_index,
    1.0 if er_status == "Positive" else 0.0,
    1.0 if her2_status == "Positive" else 0.0,
    1.0 if pr_status == "Positive" else 0.0,
    1.0 if chemotherapy == "Yes" else 0.0,
    1.0 if hormone_therapy == "Yes" else 0.0,
    1.0 if radio_therapy == "Yes" else 0.0,
    1.0 if surgery_type == "Mastectomy" else 0.0,
    1.0 if menopausal_state == "Post" else 0.0,
]], columns=FEATURE_NAMES)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧬 Breast Cancer Survival Predictor</h1>
    <p>Clinical survival prediction powered by a Random Forest model trained on the METABRIC dataset.</p>
</div>
""", unsafe_allow_html=True)

# Dataset stats row
st.markdown("""
<div class="stat-row">
    <div class="stat-pill"><div class="val">2,509</div><div class="lbl">Patients</div></div>
    <div class="stat-pill"><div class="val">14</div><div class="lbl">Features</div></div>
    <div class="stat-pill"><div class="val">70.8%</div><div class="lbl">Model Accuracy</div></div>
    <div class="stat-pill"><div class="val">0.73</div><div class="lbl">ROC-AUC</div></div>
</div>
""", unsafe_allow_html=True)

# ── Prediction output ─────────────────────────────────────────────────────────
if predict_clicked:
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)
    probability  = model.predict_proba(input_scaled)

    living_pct   = float(probability[0][0]) * 100
    deceased_pct = float(probability[0][1]) * 100
    confidence   = max(living_pct, deceased_pct)
    is_deceased  = prediction[0] == 1

    # Result card
    if is_deceased:
        st.markdown(f"""
        <div class="result-deceased">
            <div class="result-icon">⚠️</div>
            <p class="result-label">Deceased</p>
            <p class="result-sub">Confidence {confidence:.1f}% &nbsp;·&nbsp; Random Forest · METABRIC</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-living">
            <div class="result-icon">✅</div>
            <p class="result-label">Living</p>
            <p class="result-sub">Confidence {confidence:.1f}% &nbsp;·&nbsp; Random Forest · METABRIC</p>
        </div>
        """, unsafe_allow_html=True)

    # Probability bars
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="prob-card prob-living">
            <div class="prob-label">Living Probability</div>
            <div class="prob-val">{living_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(living_pct))

    with col2:
        st.markdown(f"""
        <div class="prob-card prob-deceased">
            <div class="prob-label">Deceased Probability</div>
            <div class="prob-val">{deceased_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(deceased_pct))

    # Input summary
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("View Input Summary"):
        display_df = pd.DataFrame({
            "Feature": [
                "Age at Diagnosis", "Tumor Size (mm)", "Histologic Grade",
                "Positive Lymph Nodes", "Mutation Count", "Nottingham Prognostic Index",
                "ER Status", "HER2 Status", "PR Status",
                "Chemotherapy", "Hormone Therapy", "Radio Therapy",
                "Type of Surgery", "Menopausal State"
            ],
            "Value": [
                age_at_diagnosis, tumor_size, neoplasm_histologic_grade,
                lymph_nodes_examined_positive, mutation_count, nottingham_prognostic_index,
                er_status, her2_status, pr_status,
                chemotherapy, hormone_therapy, radio_therapy,
                surgery_type, menopausal_state
            ]
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

else:
    # Placeholder before prediction
    st.markdown("""
    <div class="instruction-card">
        <div class="icon">👈</div>
        <p><strong>Fill in the patient details</strong> in the sidebar and press <strong>Predict Survival</strong> to see the result.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only.
    Predictions are not a substitute for professional medical diagnosis or clinical judgement.
</div>
""", unsafe_allow_html=True)
