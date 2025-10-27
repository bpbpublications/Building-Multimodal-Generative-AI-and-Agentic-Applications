from langchain_community.vectorstores import SKLearnVectorStore

def create_vectorstore(docs, embedding):
    return SKLearnVectorStore.from_documents(docs, embedding)