from langchain.agents import initialize_agent, Tool, AgentType
from tasks.sql_generator import generate_sql
from core.cache import cache_query
from core.llm import get_llm

llm = get_llm()

tools = [
    Tool(name="SQL Generator", func=generate_sql, description="Generates SQL queries."),
    Tool(name="SQL Executor", func=cache_query, description="Executes SQL queries with caching.")
]

sql_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # ✅ Replaces deprecated AgentType.REACT
    verbose=True
)