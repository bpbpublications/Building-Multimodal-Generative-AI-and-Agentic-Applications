# === agent/run_agent.py ===

from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import Ollama  # modern import for local Ollama support
from tools.langchain_fraud_tool import fraud_detection_tool

# Load local LLM (Mistral via Ollama)
llm = Ollama(model="mistral", temperature=0.3)

# Initialize the agent with our fraud detection tool
agent = initialize_agent(
    tools=[fraud_detection_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True  # ✅ lets the agent recover from LLM format issues
)

# Sample query to evaluate a telecom claim
query = "Can you evaluate this claim: it was submitted at 2AM, the mobile number is missing, and verification steps were skipped."

# Run the agent and print the response
print("\n🧠 Sending query to LLM agent...")
response = agent.run(query)
print("\n🔍 Agent Response:\n", response)
