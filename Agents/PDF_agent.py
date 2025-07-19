import os
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


class PDFAgent:
    def __init__(self, uploaded_file):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model_name = "gemini-2.0-flash"
        self.uploaded_file = uploaded_file

        # Save file to temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(self.uploaded_file.read())
            self.temp_path = tmp.name

        # Load and process
        self.docs = self._load_pdf()
        self.chunks = self._split_documents()
        self.vector_index = self._create_vector_index()
        self.qa_chain = self._create_qa_chain()

    def _load_pdf(self):
        loader = PyPDFLoader(self.temp_path)
        return loader.load()

    def _split_documents(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=['\n\n', '\n', ' ']
        )
        return splitter.split_documents(self.docs)

    def _create_vector_index(self):
        embeddings = GoogleGenerativeAIEmbeddings(
            model='models/embedding-001',
            google_api_key=self.api_key
        )
        vector_store = Chroma.from_documents(documents=self.chunks, embedding=embeddings)
        return vector_store.as_retriever(search_kwargs={'k': 8})

    def _create_qa_chain(self):
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
        response = self.qa_chain.invoke({"query": question})
        return response['result']
