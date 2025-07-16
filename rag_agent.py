import streamlit as st
import os
import base64
import tempfile
import time
import gc
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader, JSONLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

from crewai import Agent, Crew, Task, Process

# Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_loader(file_path, ext):
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path)
    elif ext == ".csv":
        return CSVLoader(file_path)
    elif ext == ".json":
        return JSONLoader(file_path)
    elif ext in [".doc", ".docx"]:
        return UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)

def create_agents_and_tasks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    combined_context = "\n".join([t.page_content for t in texts])

    retriever_agent = Agent(
        role="Context Retriever",
        goal="Retrieve relevant information from uploaded document",
        backstory="You are great at identifying relevant info from user-uploaded files.",
        verbose=True,
        llm=load_llm()
    )

    answer_agent = Agent(
        role="Answer Synthesizer",
        goal="Answer user queries from context",
        backstory="You turn retrieved context into useful answers.",
        verbose=True,
        llm=load_llm()
    )

    retrieval_task = Task(
        description="Retrieve relevant content for query: {query}",
        expected_output="Relevant context as text.",
        agent=retriever_agent
    )

    response_task = Task(
        description="Generate a final answer based on: {query} and retrieved content",
        expected_output="Final user-friendly answer",
        agent=answer_agent
    )

    crew = Crew(
        agents=[retriever_agent, answer_agent],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,
        verbose=True
    )

    return crew, combined_context

# Remainder of the app (uploading, chat interface, session state, etc.)
# remains unchanged from previous version.
def preview_text(content, file_name):
    st.markdown(f"### Preview of `{file_name}`")
    st.text_area("Content", content[:2000], height=300)

# ==== Session Setup ====
if "messages" not in st.session_state:
    st.session_state.messages = []

if "crew" not in st.session_state:
    st.session_state.crew = None

if "file_text" not in st.session_state:
    st.session_state.file_text = ""

def reset_chat():
    st.session_state.messages = []
    st.session_state.crew = None
    st.session_state.file_text = ""
    gc.collect()

# ==== Sidebar: Upload Any File ====
with st.sidebar:
    st.header("Upload Any File")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "csv", "json", "doc", "docx"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name.split('.')[-1]) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        ext = os.path.splitext(temp_path)[-1].lower()

        try:
            loader = get_loader(temp_path, ext)
            with st.spinner("Loading and parsing file..."):
                documents = loader.load()
                crew, combined_text = create_agents_and_tasks(documents)

            st.session_state.crew = crew
            st.session_state.file_text = combined_text
            st.success("File loaded and indexed!")
            preview_text(combined_text, uploaded_file.name)

        except Exception as e:
            st.error(f"Failed to load file: {e}")

    st.button("Clear Chat", on_click=reset_chat)

# ==== Main Chat ====
st.title("Multi-File Agentic Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about your uploaded file...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        spinner = st.empty()
        placeholder = st.empty()

        with st.spinner("Thinking..."):
            result = st.session_state.crew.kickoff(inputs={"query": prompt}).raw

        response_lines = result.split('\n')
        full_response = ""

        for i, line in enumerate(response_lines):
            full_response += line + '\n'
            placeholder.markdown(full_response + "▌")
            time.sleep(0.1)

        placeholder.markdown(full_response.strip())
        st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})
