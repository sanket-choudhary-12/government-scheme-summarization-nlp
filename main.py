import streamlit as st
import fitz  # PyMuPDF for reading PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.schema import Document
from transformers import pipeline
import torch
import requests
from bs4 import BeautifulSoup
import pickle  # To save the FAISS index

# Set up Hugging Face pipeline and embeddings
device = 0 if torch.cuda.is_available() else -1
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=device)
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
embeddings = HuggingFaceEmbeddings()

# Function to read and extract text from a PDF file
def read_pdf(uploaded_file):
    text = ""
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in pdf_document:
        page_text = page.get_text()
        text += page_text
    pdf_document.close()
    return text

# Function to extract text from a URL
def fetch_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join(p.get_text() for p in soup.find_all('p'))
    return text

# Function to summarize text in chunks
def generate_summary(text):
    if len(text) < 100:
        return "Text is too short for summarization."
    elif len(text) > 2000:
        chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
        summary = " ".join(
            [summarizer_pipeline(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks])
    else:
        summary = summarizer_pipeline(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return summary

# Streamlit app layout
st.title("Document Question Answering and Summarization App")
st.sidebar.title("Upload Document, Enter URL, and Ask Questions")

# Option to upload a PDF file or enter a URL
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])
url = st.sidebar.text_input("Or enter a URL to extract content:")

# Process PDF upload or URL text
document_text = ""
if uploaded_file is not None:
    document_text = read_pdf(uploaded_file)
    st.write("File uploaded and processed successfully!")
elif url:
    document_text = fetch_text_from_url(url)
    st.write("URL content fetched successfully!")

# If document text is available, generate summary and set up QA
if document_text:
    # Display the summary
    summary = generate_summary(document_text)
    st.write("Summary of the Document:")
    st.write(summary)

    # Wrap document text in LangChain's Document class
    documents = [Document(page_content=document_text)]

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Check if texts were generated
    if not texts:
        st.error("No text chunks created from the document.")
    else:
        # Create a FAISS vector store for embeddings
        vector_store = FAISS.from_documents(texts, embeddings)
        st.write("Document successfully embedded!")

        # Save the FAISS index to a pickle file
        with open("faiss_store_openai.pkl", "wb") as f:
            pickle.dump(vector_store, f)
        st.write("FAISS index saved to faiss_store_openai.pkl.")

        # Input question section
        question = st.sidebar.text_input("Enter your question:")
        if st.sidebar.button("Get Answer"):
            if question:
                # Perform similarity search and question answering
                docs = vector_store.similarity_search(question, k=3)  # Retrieve top 3 documents
                if docs:
                    context = " ".join(doc.page_content for doc in docs)  # Combine content from the top results
                    result = qa_pipeline(question=question, context=context)
                    st.write("Answer:", result['answer'])
                else:
                    st.write("No relevant context found in the document.")
            else:
                st.write("Please enter a question to get an answer.")
else:
    st.sidebar.write("Please upload a PDF document or enter a URL.")