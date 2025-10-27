from langchain_core.tools import Tool
from tools.fraud_tool import call_fraud_model

print("✅ langchain_fraud_tool module loaded")

fraud_detection_tool = Tool(
    name="FraudDetectionTool",
    func=call_fraud_model,
    description="Use this to check if a telecom claim is likely fraudulent. Provide structured features like IS_MISSING_MOBILE, HOUR_TO_RAISE_CLAIM, and TOTAL_VERIFICATIONS."
)

