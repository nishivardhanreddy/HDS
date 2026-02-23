from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project-root imports work when launching from subdirectories.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import load_symptom_dataset, load_symptom_reference_tables
from engine.prediction_engine import PredictionEngine
from engine.risk_scoring import classify_risk, compute_health_risk_score, preventive_recommendations


st.set_page_config(page_title="Context-Aware Disease Prediction", layout="wide")


@st.cache_resource
def get_engine() -> PredictionEngine:
    return PredictionEngine(model_dir="artifacts/models")


@st.cache_data
def get_symptom_options() -> list[str]:
    symptom_df, symptom_cols = load_symptom_dataset("dataset.csv")
    options = sorted(
        {
            token.replace("_", " ")
            for col in symptom_cols
            for token in symptom_df[col].dropna().astype(str).tolist()
            if token and token != "nan"
        }
    )
    return options


@st.cache_data
def get_reference_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_symptom_reference_tables()


def _bool_to_binary(value: bool) -> int:
    return 1 if value else 0


def _smoking_label_to_code(label: str) -> float:
    mapping = {"Never": 0.0, "Former": 1.0, "Current": 2.0}
    return mapping[label]


def main() -> None:
    st.title("Context-Aware Disease Prediction & Health Risk Assessment")
    st.caption("Clinical decision support prototype with multi-modal modeling and explainable predictions.")

    engine = get_engine()
    if not engine.available_models():
        st.error("No trained model artifacts found in `artifacts/models`. Run `python train_models.py` first.")
        st.stop()

    symptom_options = get_symptom_options()
    descriptions_df, precautions_df = get_reference_tables()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Symptoms and Immediate Illness Context")
        symptoms = st.multiselect(
            "Select active symptoms",
            options=symptom_options,
            help="Symptoms drive immediate disease screening via the symptom model.",
        )

        st.subheader("Lifestyle and Preventive Context")
        smoking_status = st.selectbox("Smoking status", ["Never", "Former", "Current"])
        alcohol_per_week = st.number_input("Alcohol intake (drinks/week)", min_value=0.0, max_value=70.0, value=2.0)
        physically_inactive = st.checkbox("Physically inactive in last 30 days", value=False)
        sleep_hours = st.number_input("Sleep duration (hours/night)", min_value=0.0, max_value=14.0, value=7.0)
        bmi_lifestyle = st.number_input("BMI (lifestyle context)", min_value=10.0, max_value=60.0, value=24.0)

        st.subheader("Medical History")
        hypertension_history = st.checkbox("Have you ever been told you have high blood pressure?", value=False)
        diabetes_history = st.checkbox("Have you ever been told you have diabetes?", value=False)
        heart_disease_history = st.checkbox("Do you have a history of heart disease?", value=False)
        age_group = st.slider(
            "Age group (younger to older)",
            min_value=1,
            max_value=13,
            value=6,
            help="A grouped age scale used by the lifestyle model.",
        )
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        st.subheader("Optional Diabetes Clinical Inputs (Patient-Friendly)")
        d_gender = st.selectbox("Sex assigned at birth", ["Female", "Male", "Other"], index=0)
        d_age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0)
        d_hypertension_label = st.selectbox("High blood pressure diagnosis", ["No", "Yes"], index=0)
        d_heart_disease_label = st.selectbox("Heart disease diagnosis", ["No", "Yes"], index=0)
        d_smoking_history = st.selectbox(
            "Smoking history",
            ["never", "former", "current", "not current", "No Info", "ever"],
            index=0,
        )
        d_bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=24.0)
        d_hba1c = st.number_input(
            "Average blood sugar over 3 months (HbA1c %)",
            min_value=3.0,
            max_value=15.0,
            value=5.8,
            help="HbA1c estimates your average blood sugar level over the last 2-3 months.",
        )
        d_glucose = st.number_input(
            "Current blood sugar level (mg/dL)",
            min_value=40.0,
            max_value=400.0,
            value=110.0,
        )
        d_blood_pressure = st.number_input("Blood pressure (mmHg, optional)", min_value=60.0, max_value=240.0, value=120.0)
        d_insulin = st.number_input("Insulin level (optional)", min_value=0.0, max_value=900.0, value=80.0)

        st.subheader("Optional Heart Clinical Inputs (Patient-Friendly)")
        h_age = st.number_input("Age (heart assessment)", min_value=0.0, max_value=120.0, value=52.0)
        h_sex_label = st.selectbox("Sex assigned at birth (heart model)", ["Female", "Male"], index=1)
        h_cp_label = st.selectbox(
            "Chest pain pattern",
            [
                "Typical angina pain",
                "Atypical chest pain",
                "Non-anginal chest pain",
                "No chest pain symptoms",
            ],
            index=1,
        )
        h_trestbps = st.number_input("Resting blood pressure (mmHg)", min_value=80.0, max_value=250.0, value=130.0)
        h_chol = st.number_input("Total cholesterol (mg/dL)", min_value=80.0, max_value=650.0, value=220.0)
        h_fbs_label = st.selectbox("Fasting blood sugar above 120 mg/dL?", ["No", "Yes"], index=0)
        h_restecg_label = st.selectbox(
            "Resting ECG result",
            [
                "Normal",
                "Mild electrical abnormality (ST-T change)",
                "Left ventricular hypertrophy pattern",
            ],
            index=0,
        )
        h_thalach = st.number_input("Maximum heart rate reached", min_value=60.0, max_value=230.0, value=150.0)
        h_exang_label = st.selectbox("Chest pain during exercise?", ["No", "Yes"], index=0)
        h_oldpeak = st.number_input(
            "Exercise-related ECG depression",
            min_value=0.0,
            max_value=7.0,
            value=1.0,
            help="Higher values can indicate reduced blood flow to the heart during exercise.",
        )
        h_slope_label = st.selectbox("Exercise ECG slope", ["Upsloping", "Flat", "Downsloping"], index=1)
        h_ca = st.selectbox("Number of major vessels seen on imaging", [0, 1, 2, 3, 4], index=0)
        h_thal_label = st.selectbox(
            "Thalassemia scan result",
            ["Unknown/Not measured", "Normal blood flow", "Fixed perfusion defect", "Reversible perfusion defect"],
            index=2,
        )

    run_prediction = st.button("Run Context-Aware Assessment", type="primary", use_container_width=True)
    if not run_prediction:
        return

    payload = {
        "symptoms": [s.strip().lower().replace(" ", "_") for s in symptoms],
        "lifestyle": {
            "smoking_status": _smoking_label_to_code(smoking_status),
            "alcohol_per_week": alcohol_per_week,
            "physically_inactive": _bool_to_binary(physically_inactive),
            "sleep_hours": sleep_hours,
            "bmi": bmi_lifestyle,
            "age_group": age_group,
            "gender": gender,
        },
        "history": {
            "hypertension_history": _bool_to_binary(hypertension_history),
            "diabetes_history": _bool_to_binary(diabetes_history),
            "heart_disease_history": _bool_to_binary(heart_disease_history),
        },
        "diabetes_clinical": {
            "gender": d_gender,
            "age": d_age,
            "hypertension": _bool_to_binary(d_hypertension_label == "Yes"),
            "heart_disease": _bool_to_binary(d_heart_disease_label == "Yes"),
            "smoking_history": d_smoking_history,
            "bmi": d_bmi,
            "HbA1c_level": d_hba1c,
            "blood_glucose_level": d_glucose,
            "blood_pressure": d_blood_pressure,
            "insulin": d_insulin,
        },
        "heart_clinical": {
            "age": h_age,
            "sex": 1 if h_sex_label == "Male" else 0,
            "cp": {
                "Typical angina pain": 0,
                "Atypical chest pain": 1,
                "Non-anginal chest pain": 2,
                "No chest pain symptoms": 3,
            }[h_cp_label],
            "trestbps": h_trestbps,
            "chol": h_chol,
            "fbs": _bool_to_binary(h_fbs_label == "Yes"),
            "restecg": {
                "Normal": 0,
                "Mild electrical abnormality (ST-T change)": 1,
                "Left ventricular hypertrophy pattern": 2,
            }[h_restecg_label],
            "thalach": h_thalach,
            "exang": _bool_to_binary(h_exang_label == "Yes"),
            "oldpeak": h_oldpeak,
            "slope": {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[h_slope_label],
            "ca": h_ca,
            "thal": {
                "Unknown/Not measured": 0,
                "Normal blood flow": 1,
                "Fixed perfusion defect": 2,
                "Reversible perfusion defect": 3,
            }[h_thal_label],
        },
    }

    result = engine.predict(payload)
    rule_score, rule_risk_factors = compute_health_risk_score(payload["lifestyle"], payload["history"])
    rule_level = classify_risk(rule_score)
    recommendations = preventive_recommendations(rule_risk_factors)

    st.subheader("Prediction Dashboard")
    top_col1, top_col2, top_col3 = st.columns(3)
    top_col1.metric("Combined AI Risk Probability", f"{result['combined_probability']:.2%}")
    top_col2.metric("Combined AI Risk Level", result["combined_level"])
    top_col3.metric("Preventive Rule-Based Risk", f"{rule_score:.1f}/100 ({rule_level})")

    st.progress(min(max(rule_score / 100.0, 0.0), 1.0), text=f"Rule-based risk meter: {rule_level}")
    st.write("Active context:", ", ".join(result["context"]["active_modalities"]) or "None")

    rows = []
    for pred in result["predictions"]:
        rows.append(
            {
                "Model": pred["model"],
                "Prediction": pred["prediction"],
                "Probability": f"{pred['probability']:.2%}",
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Explainability: Key Contributing Factors")
    for pred in result["predictions"]:
        st.markdown(f"**{pred['model'].capitalize()} model contributors**")
        explanation = pred.get("explanation", [])
        if not explanation:
            st.write("No explanation available.")
            continue
        for item in explanation:
            st.write(f"- {item['feature']}: {item['score']:.4f}")

    if symptoms and any(p["model"] == "symptom" for p in result["predictions"]):
        symptom_prediction = next(p for p in result["predictions"] if p["model"] == "symptom")
        disease_name = symptom_prediction["prediction"]
        st.subheader("Disease Information")
        desc_row = descriptions_df[descriptions_df["Disease"] == disease_name]
        if not desc_row.empty:
            st.write(desc_row.iloc[0]["Description"])
        precaution_row = precautions_df[precautions_df["Disease"] == disease_name]
        if not precaution_row.empty:
            precautions = [str(precaution_row.iloc[0][c]) for c in precaution_row.columns if c.startswith("Precaution_")]
            precautions = [p for p in precautions if p and p.lower() != "nan"]
            if precautions:
                st.write("Recommended precautions:")
                for p in precautions:
                    st.write(f"- {p}")

    st.subheader("Preventive Recommendations")
    for rec in recommendations:
        st.write(f"- {rec}")


if __name__ == "__main__":
    main()
