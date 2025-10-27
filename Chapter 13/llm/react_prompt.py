from langchain.prompts import PromptTemplate

react_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent assistant using the ReAct (Reasoning + Acting) technique.
Break down the user query into reasoning steps and retrieve relevant information accordingly.

Question: {question}
Relevant Context:
{context}

First, list your reasoning steps clearly.
Then, provide a final answer based on those steps and the retrieved context.

Reasoning Steps:
1.
"""
)