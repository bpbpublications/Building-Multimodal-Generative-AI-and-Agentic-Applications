from fpdf import FPDF
import os

os.makedirs("data/documents", exist_ok=True)

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, """
LangChain Agent Types:
- ReAct
- Conversational
- Tool-using

LangChain Memory Types:
- ConversationBufferMemory
- VectorStoreRetrieverMemory
- SummaryMemory

Prompt engineering is the process of crafting inputs to guide LLMs effectively.
""")
pdf.output("data/documents/langchain_test.pdf")
print("✅ Sample PDF created at data/documents/langchain_test.pdf")
