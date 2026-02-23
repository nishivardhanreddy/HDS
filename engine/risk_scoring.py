from __future__ import annotations


def compute_health_risk_score(lifestyle: dict, history: dict) -> tuple[float, list[str]]:
    score = 0.0
    risk_factors: list[str] = []

    smoking_raw = lifestyle.get("smoking_status", "")
    smoking = str(smoking_raw).lower()
    smoking_numeric = None
    try:
        smoking_numeric = float(smoking_raw)
    except Exception:
        smoking_numeric = None

    if smoking in {"current", "2", "2.0"} or smoking_numeric == 2:
        score += 22
        risk_factors.append("Current smoking")

    bmi = lifestyle.get("bmi")
    if bmi is not None:
        try:
            bmi = float(bmi)
            if bmi >= 30:
                score += 22
                risk_factors.append("Obesity (BMI >= 30)")
            elif bmi >= 25:
                score += 10
                risk_factors.append("Overweight (BMI 25-29.9)")
        except Exception:
            pass

    if int(history.get("hypertension_history", 0) or 0) == 1:
        score += 20
        risk_factors.append("Hypertension history")

    if int(history.get("diabetes_history", 0) or 0) == 1:
        score += 20
        risk_factors.append("Diabetes history")

    if int(history.get("heart_disease_history", 0) or 0) == 1:
        score += 16
        risk_factors.append("Heart disease history")

    inactive = int(lifestyle.get("physically_inactive", 0) or 0)
    if inactive == 1:
        score += 12
        risk_factors.append("Physical inactivity")

    sleep = lifestyle.get("sleep_hours")
    if sleep is not None:
        try:
            sleep = float(sleep)
            if sleep < 6 or sleep > 9:
                score += 8
                risk_factors.append("Suboptimal sleep duration")
        except Exception:
            pass

    return min(score, 100.0), risk_factors


def classify_risk(score: float) -> str:
    if score < 30:
        return "LOW"
    if score < 60:
        return "MODERATE"
    return "HIGH"


def preventive_recommendations(risk_factors: list[str]) -> list[str]:
    recommendations: list[str] = []

    if any("smoking" in factor.lower() for factor in risk_factors):
        recommendations.append("Start a smoking cessation plan and clinical counseling.")
    if any("obesity" in factor.lower() or "overweight" in factor.lower() for factor in risk_factors):
        recommendations.append("Use a structured weight management plan with nutrition support.")
    if any("inactivity" in factor.lower() for factor in risk_factors):
        recommendations.append("Target at least 150 minutes/week of moderate activity.")
    if any("hypertension" in factor.lower() for factor in risk_factors):
        recommendations.append("Monitor blood pressure regularly and reduce sodium intake.")
    if any("diabetes history" in factor.lower() for factor in risk_factors):
        recommendations.append("Schedule periodic fasting glucose and HbA1c checks.")
    if any("sleep" in factor.lower() for factor in risk_factors):
        recommendations.append("Optimize sleep hygiene and maintain 7-8 hours/night.")

    if not recommendations:
        recommendations.append("Maintain current healthy lifestyle and regular preventive screening.")
    return recommendations
