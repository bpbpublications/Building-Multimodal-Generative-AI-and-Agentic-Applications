
import requests

def call_fraud_model(features: dict) -> str:
    try:
        response = requests.post("http://localhost:8000/predict_fraud", json=features)
        if response.status_code != 200:
            return f"Model API error: {response.text}"
        prob = response.json()["fraud_probability"]

        explanation = []
        if features.get("IS_MISSING_MOBILE") == 1:
            explanation.append("mobile number is missing")
        if features.get("HOUR_TO_RAISE_CLAIM", 0) < 6:
            explanation.append("claim was submitted during off-hours")
        if features.get("TOTAL_VERIFICATIONS", 2) < 2:
            explanation.append("low verification count")

        reason = ", ".join(explanation) if explanation else "model factors"
        return f"The fraud probability is {prob:.2f}. Likely flagged due to {reason}."

    except Exception as e:
        return f"Error calling fraud model: {str(e)}"