import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


# load_dotenv()m
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# model = genai.GenerativeModel("gemini-2.0-flash")

# response = model.generate_content("whoc is the nelson mandela")
# print(response.text)

##Document_loaders
##1.TextLoader(TEXT_BASED_RAG)
class TextAgent:
    def __init__(self, text_path: str):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model_name = "gemini-2.0-flash"
        self.text_path = text_path

        # Load and process
        self.docs = self._load_text()
        self.chunks = self._split_documents()
        self.vector_index = self._create_vector_index()
        self.qa_chain = self._create_qa_chain()

    def _load_text(self):
        """Loads the text document"""
        loader = TextLoader(self.text_path)
        return loader.load()

    def _split_documents(self):
        """Splits documents into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=['\n\n', '\n', ' ']
        )
        return splitter.split_documents(self.docs)

    def _create_vector_index(self):
        """Creates Chroma vector index"""
        embeddings = GoogleGenerativeAIEmbeddings(
            model='models/embedding-001',
            google_api_key=self.api_key
        )
        vector_store = Chroma.from_documents(documents=self.chunks, embedding=embeddings)
        return vector_store.as_retriever(search_kwargs={'k': 8})

    def _create_qa_chain(self):
        """Builds RAG chain"""
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.vector_index,
            return_source_documents=True
        )

    def ask(self, question: str) -> str:
        """Get answer from the RAG chain"""
        response = self.qa_chain.invoke({"query": question})
        return response['result']

# ===== Usage Example =====
if __name__ == "__main__":
    text_path = r"D:\desktop\Agriculture\Agricultural Statistics_Books_recomendataions.txt"
    rag_agent = TextAgent(text_path)
    question = "What is the tittle of BL AGGARWAL book?"
    answer = rag_agent.ask(question)

    sys.stdout.reconfigure(encoding='utf-8')
    print(answer)
