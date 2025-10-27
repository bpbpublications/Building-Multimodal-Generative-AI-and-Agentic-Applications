# agents/summarization_schema_agent.py

from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from tasks.summarizer import summarize_and_store, pre_filter
from tasks.aggregator import aggregate_summarized_data
from tasks.schema_matcher import schema_matching
from core.llm import get_llm

llm = get_llm()

tools = [
    StructuredTool.from_function(pre_filter, name="Pre-filter", description="Vector search for row summaries. Accepts a 'query' string."),
    StructuredTool.from_function(schema_matching, name="Schema Matcher", description="Matches query against global schema."),
    StructuredTool.from_function(summarize_and_store, name="Row Summarizer", description="Summarizes rows and stores embeddings."),
    StructuredTool.from_function(aggregate_summarized_data, name="Summarization Aggregator", description="Aggregates relevant summaries for insight.")
]

summarization_schema_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
