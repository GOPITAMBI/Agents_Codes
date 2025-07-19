import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# model = genai.GenerativeModel("gemini-2.0-flash")

# # response = model.generate_content("whoc is the nelson mandela")
# # print(response.text)


#3.CSVLoader(CSV_BASED_RAG)
class CSVAgent:
    def __init__(self, csv_path: str):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model_name = "gemini-2.0-flash"
        self.csv_path = csv_path

        # Load and process
        self.docs = self._load_csv()
        self.chunks = self._split_documents()
        self.vector_index = self._create_vector_index()
        self.qa_chain = self._create_qa_chain()

    def _load_csv(self):
        """Loads the CSV document"""
        loader = CSVLoader(self.csv_path)
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
    csv_path = r"D:\desktop\Agriculture\GxE_Analysis\Gen_data.csv"
    rag_agent = CSVAgent(csv_path)
    question = "what is the g1 value at GRW?"
    answer = rag_agent.ask(question)

    sys.stdout.reconfigure(encoding='utf-8')
    print(answer)
