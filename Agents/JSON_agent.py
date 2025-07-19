import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import sys

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# response = model.generate_content("whoc is the nelson mandela")
# print(response.text)

#4.Json (WEB_BASED_RAG)
Json_path=r"D:\desktop\Prd_info.json"
jq_schema = ".[] | {text: (.tittle + \" \" + .description)}"
json_file_loader=JSONLoader(file_path=Json_path, jq_schema=jq_schema, text_content=False)
loaded_json_doc=json_file_loader.load()
print(len(loaded_json_doc))

#Split into chunks
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = recursive_splitter.split_documents(loaded_json_doc)

#Set up embeddings (GoogleGenerativeAI)
embeddings = GoogleGenerativeAIEmbeddings(
    model='models/embedding-001',
    google_api_key=os.getenv("GOOGLE_API_KEY") 
)

#Vector store
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
vector_index = vector_store.as_retriever(search_kwargs={'k': 8})

llm= ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
#Define your LLM (Chat model)
rag = RetrievalQA.from_chain_type(
    llm=llm,
    retriever = vector_index,
    return_source_documents=True
)

question = 'list out all the productName'
response = rag.invoke({"query":question})

print(response['result'])
# with open(Json_path, 'r', encoding='utf-8') as f:
#     data = json.load(f)
#     print(type(data))