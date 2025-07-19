import streamlit as st
import os
from dotenv import load_dotenv
from Agents.CSV_agent import CSVAgent
from Agents.PDF_agent import PDFAgent
from Agents.Text_agent import TextAgent
from src.custom_tool import FireCrawlWebSearchTool
#from utils.parser import extract_text_from_uploaded_file


# Load environment variables
load_dotenv()

# File Upload
uploaded_file = st.file_uploader("Upload PDF, CSV, or TXT", type=["pdf", "csv", "txt"])
question = st.text_input("Enter your question")

if uploaded_file and question:
    file_type = uploaded_file.name.lower().split(".")[-1]
    

    if file_type == "pdf":
        answer = PDFAgent(uploaded_file)
    elif file_type == "csv":
        answer = CSVAgent(uploaded_file)
    elif file_type == "txt":
        answer = TextAgent(uploaded_file)
    else:
        answer = "Unsupported file type."

    if "not found" in answer.lower() or answer.strip() == "":
        st.warning("Answer not found in document.")
        if st.radio("Do you want to search the web?", ["Yes", "No"]) == "Yes":
            firecrawl = FireCrawlWebSearchTool()
            web_answer = firecrawl.run(question)
            st.success("Answer from web:")
            st.write(web_answer)
        else:
            st.info("No web search selected. Please try another question.")
    else:
        st.success("Answer from document:")
        st.write(answer)
