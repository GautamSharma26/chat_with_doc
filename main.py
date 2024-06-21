# import streamlit as st
# import os
# # from langchain_openai import ChatOpenAI
# from langchain import hub
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_vertexai import ChatVertexAI
# from langchain_google_genai import G
# import vertexai
# from dotenv import load_dotenv
# load_dotenv()
#
# PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
# vertexai.init(project=PROJECT_ID, location="us-central1")
#
# llm = ChatVertexAI(model="gemini-1.5-flash")
#
# # Set environment variables
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "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"
#
# # Initialize OpenAI model
# # llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
#
# # Load, chunk and index the contents of the PDF
# def load_and_process_pdf(pdf_path):
#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splits = text_splitter.split_documents(docs)
#     vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
#     return vectorstore
#
# # Retrieve and generate using the relevant snippets of the PDF
# def generate_response(vectorstore, question):
#     retriever = vectorstore.as_retriever()
#     prompt = hub.pull("rlm/rag-prompt")
#
#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)
#
#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#
#     response = rag_chain.invoke(question)
#     return response
#
# # Streamlit app
# st.title("PDF QA with LangChain and OpenAI")
#
# uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
# question = st.text_input("Enter your question")
#
# if uploaded_file and question:
#     pdf_path = os.path.join("/tmp", uploaded_file.name)
#     with open(pdf_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#
#     with st.spinner("Processing..."):
#         vectorstore = load_and_process_pdf(pdf_path)
#         response = generate_response(vectorstore, question)
#
#     st.write("## Response")
#     st.write(response)
#

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from docx import Document
from io import BytesIO
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)


# Initialize Vertex AI model

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

    @staticmethod
    def load_vector_store():
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        index_path = "faiss_index"
        if os.path.exists(index_path):
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            return None


class ChatManager:
    @staticmethod
    def get_conversational_chain():
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say "answer is not available in the context" and don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """

        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)

    @staticmethod
    def get_response(user_question):
        if vector_store := VectorStoreManager.load_vector_store():

            docs = vector_store.similarity_search(user_question, k=10)

            chain = ChatManager.get_conversational_chain()

            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )

            if "Answer is not available in the context" in response["output_text"]:
                st.write("Context-specific answer not found. Fetching a general answer.")
                general_answer = genai.GenerativeModel("gemini-1.5-flash").generate_content(user_question)
                return general_answer.text
            return response["output_text"]
        else:
            st.write("No Docs are available.Data is fetching from general context...")
            general_answer = genai.GenerativeModel("gemini-1.5-flash").generate_content(user_question)
            return general_answer.text


def main():
    st.set_page_config(page_title="Chat with Documents", page_icon="💁", layout="wide")

    st.title("Chat with Your Documents 💁")
    st.markdown("""
    Upload your PDF or DOCX files, and then ask any questions you have about the content.
    The system will provide detailed answers based on the uploaded documents.
    """)

    with st.sidebar:
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

    st.subheader("Ask a Question")
    user_question = st.text_input("Type your question here:")

    if user_question:
        with st.spinner("Generating response..."):
            response = ChatManager.get_response(user_question)
            st.markdown("### Reply:")
            st.markdown(response)


if __name__ == "__main__":
    main()
