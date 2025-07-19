import fitz  # PyMuPDF
import pandas as pd
import os
import tempfile


from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader

def extract_text_from_uploaded_file(uploaded_file, file_type: str):
    """Unified function to load content from PDF, CSV, or TXT files"""
    if file_type == "pdf":
        loader = PyPDFLoader(uploaded_file)
    elif file_type == "csv":
        loader = CSVLoader(uploaded_file)
    elif file_type == "txt":
        loader = TextLoader(uploaded_file)
    else:
        raise ValueError("Unsupported file type")
    
    return loader.load()
