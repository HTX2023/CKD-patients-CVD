import streamlit as st
import joblib
import json
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CVD Risk Prediction in CKD Patients",
    page_icon="🩺",
    layout="wide"
)

# ──────────────────────────────────────────────────────────────────────
# Load model and feature names
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("RandomForest.pkl")

@st.cache_data
def load_features():
    with open("feature_names.json", "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
feature_names = load_features()

# ──────────────────────────────────────────────────────────────────────
# Sidebar: Indicator Descriptions
# ──────────────────────────────────────────────────────────────────────
st.sidebar.header("🧾 Indicator Descriptions")
st.sidebar.markdown("""
**Hypertension**  
Persistently elevated arterial blood pressure, commonly defined as systolic ≥ 140 mmHg or diastolic ≥ 90 mmHg.  
Hypertension is a major risk factor for cardiovascular disease (CVD); prolonged hypertension can lead to stroke, coronary artery disease, heart failure, and chronic kidney disease.

**Dyslipidemia**  
Abnormal levels of lipids in the blood (e.g., elevated total cholesterol, low‐density lipoprotein cholesterol, triglycerides, or low high‐density lipoprotein cholesterol).  
Dyslipidemia is a critical risk factor for atherosclerosis and CVD.

**Self-Rated Health (SRH)**  
An individual’s overall assessment of their health status, typically measured by asking, “In general, how would you rate your health?” with response options such as “Excellent, Good, Fair, Poor, Very Poor.”  
SRH has been shown to be an independent and valid predictor of all-cause mortality, functional decline, and hospitalization risk.

**Number of ADL Difficulties**  
Measures a person’s ability to perform basic self-care tasks, including bathing, dressing, eating, toileting, transferring, and continence (six domains).  
Inability to complete any ADL task independently indicates functional impairment and potential need for assistance or long-term care.

**TyG-BMI Index**  
The TyG index is a surrogate marker for insulin resistance calculated from fasting triglyceride and glucose levels.  
The TyG-BMI index integrates body mass index (BMI) with the TyG index, providing a composite indicator reflecting lipid, glucose metabolism, and adiposity. It is a cost-effective and practical measure for assessing insulin resistance.

**Waist Circumference (cm)**  
An indicator of abdominal fat accumulation that more directly reflects visceral adiposity than BMI.  
Elevated waist circumference signifies central obesity, substantially increasing the risk of CVD, type 2 diabetes, and metabolic syndrome.

**White Blood Cell Count (×10³/μL)**  
The number of leukocytes in peripheral blood. Leukocytes, produced in the bone marrow, function in immune defense and clearance of pathogens.  
Elevated WBC count often indicates infection or inflammation; low counts may suggest bone marrow suppression or immunodeficiency.

**Platelet Count (×10⁹/L)**  
The number of platelets in peripheral blood. Platelets are the smallest cell fragments in blood, critical for hemostasis and coagulation.  
Abnormal platelet counts can arise from bone marrow disorders, inflammatory responses, infections, or medication effects.

**Serum Creatinine/Cystatin C Ratio**  
The ratio of serum creatinine to serum cystatin C. This ratio reflects both muscle mass and renal function; a lower ratio has been associated with sarcopenia and increased mortality or postoperative complications in CKD patients.
""")

# ──────────────────────────────────────────────────────────────────────
# Main area: Title and Input Form
# ──────────────────────────────────────────────────────────────────────
st.title("🧠 Machine Learning–Based CVD Risk Prediction Model for CKD Patients")
st.markdown(
    "This application aims to predict the cardiovascular disease (CVD) risk level in chronic kidney disease (CKD) patients "
    "based on input health indicators and to provide corresponding health recommendations."
)

# Categorical variable options
yes_no_options = {"Yes": 1, "No": 0}
srh_options = {
    "Excellent": 1,
    "Good": 2,
    "Fair": 3,
    "Poor": 4,
    "Very Poor": 5
}
adlab_cADL_options = {
    "0 Difficulties": 0,
    "1 Difficulty": 1,
    "2 Difficulties": 2,
    "3 Difficulties": 3,
    "4 Difficulties": 4,
    "5 Difficulties": 5,
    "6 Difficulties": 6
}

# ──────────────────────────────────────────────────────────────────────
# User input form
# ──────────────────────────────────────────────────────────────────────
with st.form("input_form", clear_on_submit=False):
    st.subheader("Please provide the following information:")
    col1, col2 = st.columns(2)

    with col1:
        hibpe = st.selectbox("Hypertension (Yes/No)", list(yes_no_options.keys()))
        dyslipe = st.selectbox("Dyslipidemia (Yes/No)", list(yes_no_options.keys()))
        srh = st.selectbox("Self-Rated Health", list(srh_options.keys()))
        adlab_cADL = st.selectbox("Number of ADL Difficulties", list(adlab_cADL_options.keys()))
        tyg_bmi = st.number_input("TyG-BMI Index (unitless)", min_value=0.0, step=0.1)
        mwaist = st.number_input("Waist Circumference (cm)", min_value=0.0, step=0.1)
        bl_wbc = st.number_input("White Blood Cell Count (×10³/μL)", min_value=0.0, step=0.1)
        bl_pltPlatelets = st.number_input("Platelet Count (×10⁹/L)", min_value=0.0, step=0.1)
        Cr_CysC_Ratio = st.number_input("Serum Creatinine/Cystatin C Ratio (unitless)", min_value=0.0, step=0.01)

    with col2:
        st.markdown("_All detailed definitions and clinical context can be found in the sidebar._")
        st.markdown("")

    submit = st.form_submit_button("🔍 Predict Risk")

# ──────────────────────────────────────────────────────────────────────
# Process inputs and display results
# ──────────────────────────────────────────────────────────────────────
if submit:
    input_data = {
        "hibpe": yes_no_options[hibpe],
        "dyslipe": yes_no_options[dyslipe],
        "srh": srh_options[srh],
        "adlab_cADL": adlab_cADL_options[adlab_cADL],
        "tyg_bmi": tyg_bmi,
        "mwaist": mwaist,
        "bl_wbc": bl_wbc,
        "bl_pltPlatelets": bl_pltPlatelets,
        "Cr_CysC_Ratio": Cr_CysC_Ratio
    }

    # Validate numeric inputs
    numeric_fields = ["tyg_bmi", "mwaist", "bl_wbc", "bl_pltPlatelets", "Cr_CysC_Ratio"]
    for field in numeric_fields:
        if input_data[field] == 0:
            st.error(f"❌ The value for '{field}' is 0. Please enter a valid measurement.")
            st.stop()

    # Ensure the input order matches feature names
    input_df = pd.DataFrame([[input_data[f] for f in feature_names]], columns=feature_names)

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # Assume 1 indicates high risk

    # Display results
    st.subheader("🩺 Prediction Result:")
    if probability < 0.3:
        st.success(f"Risk Level: Low Risk ({probability:.2%})")
        st.markdown("✅ Recommendation: Continue maintaining healthy lifestyle habits and undergo regular check-ups.")
        st.markdown("🍎 Recommendation: Maintain a balanced diet with adequate nutrition.")
        st.markdown("🏃 Recommendation: Engage in moderate exercise regularly to improve fitness.")
    elif 0.3 <= probability < 0.7:
        st.warning(f"Risk Level: Moderate Risk ({probability:.2%})")
        st.markdown("⚠️ Recommendation: Be mindful of diet and exercise, monitor health indicators regularly, and consult a physician.")
        st.markdown("🩺 Recommendation: Schedule regular health check-ups, focus on blood pressure and lipid levels.")
        st.markdown("🧘 Recommendation: Maintain mental well-being and manage stress appropriately.")
    else:
        st.error(f"Risk Level: High Risk ({probability:.2%})")
        st.markdown("🚨 Recommendation: Seek medical attention promptly for comprehensive evaluation and treatment.")
        st.markdown("💊 Recommendation: Adhere strictly to medical advice and medication regimens.")
        st.markdown("🛌 Recommendation: Ensure adequate rest and avoid overexertion.")
