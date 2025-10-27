router_instructions = """
You are a router deciding whether a question should be answered using the user's private PDFs or from a web search.

If the question is about LangChain, prompt engineering, or other topics covered in the provided documents → use 'vectorstore'.

If the question is about recent events, weather, people, locations, or real-world data → use 'websearch'.

Return ONLY a JSON like:
{ "datasource": "websearch" }
or
{ "datasource": "vectorstore" }
"""


rag_prompt = """
You are a helpful assistant. Use ONLY the information in the CONTEXT below to answer the QUESTION.
If the CONTEXT does not contain the answer, respond exactly with:
"I don’t know based on the context."

---
CONTEXT:
{context}
---
QUESTION:
{question}
---
INSTRUCTIONS:
- Do NOT use prior knowledge.
- Do NOT make up any answers.
- ONLY use information in the context above.
- Answer in 2–3 concise sentences.
- Say “I don’t know based on the context” if unsure.
---
Answer:
"""