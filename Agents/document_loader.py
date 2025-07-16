import os
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader, JSONLoader

def load_document(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    elif ext == ".json":
        loader = JSONLoader(file_path)
    else:
        raise ValueError("Unsupported file format.")
    return loader.load()
