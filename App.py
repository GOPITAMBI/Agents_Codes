import streamlit as st
import tempfile
from agents.document_loader import load_document
from agents.rag_agent import build_rag_chain
from agents.web_agent import web_search

st.set_page_config(page_title="RAG Agent with Web Fallback", layout="centered")

st.title("RAG Agent with Web Fallback")
st.markdown("Upload a document, ask a question, and get document-based answers with optional web fallback.")

# Upload
uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt", "csv", "json"])

# Question
query = st.text_input("Ask your question")

if uploaded_file and query:
    with st.spinner("Reading document..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        try:
            documents = load_document(file_path)
            rag_chain = build_rag_chain(documents)
            result = rag_chain.run(query)

            # Check if RAG failed
            if "i don't know" in result.lower() or len(result.strip()) < 25:
                st.warning("The answer may not be in the document.")
                search = st.radio("Would you like to search the web?", ["No", "Yes"])
                if search == "Yes":
                    with st.spinner("Searching the web..."):
                        web_result = web_search(query)
                    st.markdown("### Web Answer:")
                    st.write(web_result)
                else:
                    st.info("Here’s the best we could find from the document:")
                    st.write(result)
            else:
                st.markdown("### Answer from Document:")
                st.write(result)

        except Exception as e:
            st.error(f"Error processing the document: {e}")
