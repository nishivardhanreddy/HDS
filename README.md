https://hds-3n9x.onrender.com/
# Context-Aware Disease Prediction & Health Risk Assessment

Clinical decision support prototype built with Python + Streamlit.

## Implemented Research Principles

- Context-aware prediction (model selected by available input modality)
- Multi-modal integration (symptoms + lifestyle/history + clinical indicators)
- Modality-specific models
- Explainability with top contributing factors
- Preventive health risk scoring

## Folder Structure

```text
data/
  data_loader.py
  preprocessing.py
  feature_engineering.py
models/
  symptom_model.py
  diabetes_model.py
  heart_model.py
  lifestyle_risk_model.py
engine/
  context_builder.py
  model_selector.py
  prediction_engine.py
  risk_scoring.py
  explainability.py
app/
  streamlit_app.py
train_models.py
requirements.txt
```

## Run Locally

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app/streamlit_app.py
```

## Model Mapping

- Symptom model: Multinomial Naive Bayes + TF-IDF
- Diabetes clinical model: Random Forest
- Heart physiological model: Random Forest
- Lifestyle risk model: Random Forest (BRFSS-derived features)

## Preprocessing Included

- Missing value imputation
- Categorical encoding (OneHot)
- Feature scaling (StandardScaler/MinMaxScaler)
- TF-IDF for symptoms
- Low-variance feature removal
- Class-imbalance handling with SMOTE

## Notes

- BRFSS columns vary by year; loader uses available files and harmonizes fields.
- Lifestyle target is derived from BRFSS `_RFHLTH` (fair/poor health risk proxy).
- Rule-based preventive risk score is shown alongside model probabilities.
- Diabetes UI includes blood pressure and insulin fields for clinical completeness; the current provided diabetes dataset does not include these columns, so they are reserved for extension datasets.
