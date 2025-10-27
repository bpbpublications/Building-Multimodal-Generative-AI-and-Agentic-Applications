from utils.pdf_parser import load_and_chunk_pdfs
from vectorstore.db_handler import get_vectorstore
from retriever.hybrid_search import get_hybrid_retriever
from llm.generate import get_llm
from memory.conversation_buffer import memory
from llm.react_prompt import react_prompt
from langchain.chains import ConversationalRetrievalChain

def get_rag_chain():
    chunks = load_and_chunk_pdfs()
    vectorstore = get_vectorstore(chunks)
    llm = get_llm()

    # Return a wrapper to process input query at runtime
    def invoke_rag_chain(query: str):
        hybrid_retriever = get_hybrid_retriever(chunks, vectorstore, topic=query)
        rag = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=hybrid_retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": react_prompt},
            output_key="answer"
        )
        return rag.invoke({"question": query})

    return invoke_rag_chain
