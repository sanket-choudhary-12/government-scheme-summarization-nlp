Document Question Answering and Summarization App
This project is a Streamlit-based web application for extracting text, summarizing content, and answering questions from documents or URLs. It leverages powerful NLP models such as Hugging Face's BART for summarization and DistilBERT for question answering, along with LangChain for chunking and embedding.

Features
PDF Text Extraction:

Upload a PDF document, and the app extracts and processes the text for further operations.
URL Content Extraction:

Input a URL to extract text content from its paragraphs.
Text Summarization:

Summarizes the content into concise and readable summaries using the facebook/bart-large-cnn model.
Question Answering:

Ask context-aware questions about the document or URL content using the distilbert-base-uncased-distilled-squad model.
Embeddings and Vector Search:

Chunks the text using LangChain's RecursiveCharacterTextSplitter.
Embeds chunks using Hugging Face embeddings and stores them in a FAISS index for efficient similarity searches.
FAISS Index:

Saves the document embeddings to a pickle file for reuse.
Installation
Prerequisites
Python 3.8 or above
GPU (optional but recommended for faster processing)
Clone the Repository
bash
Copy code
git clone https://github.com/your-username/nlp-summarization-app.git
cd nlp-summarization-app
Install Dependencies
Use the requirements.txt file to install all dependencies:

bash
Copy code
pip install -r requirements.txt
Key Libraries
Streamlit for the web interface.
PyMuPDF (fitz) for reading PDFs.
BeautifulSoup for extracting content from URLs.
Hugging Face's transformers for summarization and question answering.
LangChain for document chunking and embeddings.
FAISS for efficient similarity searches.
Usage
Start the App: Run the Streamlit app:

bash
Copy code
streamlit run main.py
Upload a PDF or Enter a URL:

Use the sidebar to upload a PDF document or provide a URL.
Generate a Summary:

View a concise summary of the document or webpage content.
Ask Questions:

Enter a question related to the document or webpage in the sidebar.
The app retrieves the most relevant chunks and generates an answer.
Directory Structure
graphql
Copy code
.
├── main.py               # Main application script
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
├── faiss_store_openai.pkl # Saved FAISS index (generated after embedding)
Example
Input:
PDF Document: A research paper on NLP.
URL: https://example.com/article-about-nlp
Summary Output:
vbnet
Copy code
The document discusses the importance of NLP in modern AI applications. Key topics include text processing, summarization, and embeddings.
Question:
What are the key challenges in NLP mentioned in the document?

Answer:
javascript
Copy code
The document highlights challenges such as ambiguity in language, lack of labeled data, and computational limitations.
Future Enhancements
Add support for multi-language documents.
Enhance summarization for larger documents using chunk stitching.
Integrate other QA models like GPT-4 for better accuracy.
Support for real-time updates to FAISS indexes.
