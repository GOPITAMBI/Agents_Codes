from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def build_rag_chain(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Embed and store in FAISS
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-4", temperature=0)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain
