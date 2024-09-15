import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from docx import Document
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)


# Initialize Vertex AI model
#
class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file):
        text = ""
        pdf_reader = PdfReader(BytesIO(file.read()))
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    @staticmethod
    def extract_text_from_docx(file):
        doc = Document(BytesIO(file.read()))
        full_text = [para.text for para in doc.paragraphs]
        return '\n'.join(full_text)

    @staticmethod
    def get_text(files):
        if not files:
            return False

        uploaded_file = files[0]

        if "pdf" in uploaded_file.name.lower():
            return DocumentProcessor.extract_text_from_pdf(uploaded_file)
        elif any(ext in uploaded_file.name.lower() for ext in ("docx", "doc")):
            return DocumentProcessor.extract_text_from_docx(uploaded_file)
        else:
            return False


#
class VectorStoreManager:
    @staticmethod
    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)

    @staticmethod
    def create_vector_store(text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")


def main():
    st.set_page_config(page_title="Upload Your Docs/Pdf", page_icon="💁", layout="wide")

    st.markdown("""

        <style>
        /* Hide the GitHub icon in the Streamlit app */
           #MainMenu {visibility: hidden;}
            footer {visibility:hidden;}
            .GithubIcon {visibility: hidden;}
        .intro-text {
            font-size: 18px;
            color: #4F4F4F;
            text-align: center;
            margin: 20px 0;
        }
        .highlight {
            font-weight: bold;
            color: #FF6347;
        }
        </style>
        <div class="intro-text">
            Upload your <span class="highlight">PDF</span> or <span class="highlight">DOCX</span> files here...
            
        </div>
        """, unsafe_allow_html=True)

    st.title("Upload and Process Documents")
    pdf_docs = st.file_uploader("Upload your PDF or DOCX files", accept_multiple_files=True)

    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = DocumentProcessor.get_text(pdf_docs)
            if raw_text:
                text_chunks = VectorStoreManager.get_text_chunks(raw_text)
                VectorStoreManager.create_vector_store(text_chunks)
                st.success("Documents processed successfully!")
            else:
                st.error("Please upload a valid PDF or DOCX file.")


if __name__ == "__main__":
    main()
