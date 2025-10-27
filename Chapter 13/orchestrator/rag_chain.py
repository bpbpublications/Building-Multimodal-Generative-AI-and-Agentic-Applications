from utils.data_loader import load_and_chunk_all_data
from vectorstore.db_handler import get_vectorstore
from retriever.hybrid_search import get_hybrid_retriever
from llm.generate import get_llm
from memory.conversation_buffer import memory
from llm.react_prompt import react_prompt
from langchain.chains import ConversationalRetrievalChain

def get_rag_chain():
    chunks = load_and_chunk_all_data()
    vectorstore = get_vectorstore(chunks)
    retriever = get_hybrid_retriever(chunks, vectorstore)
    llm = get_llm()

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": react_prompt},
        output_key="answer"
    )