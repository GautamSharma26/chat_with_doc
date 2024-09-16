import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)


class VectorStoreManager:
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
        You are a highly skilled resume analyzer. Given the context extracted from a resume, answer the question based on the information provided. If the same entity (such as a company name, skill, or degree) is mentioned multiple times, consider it as a single entity. If the question is not covered in the resume, respond with "The answer is not available in the resume."

        Context:
        {context}

        Question:
        {question}

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

            if "The answer is not available in the resume." in response["output_text"].lower():
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

    st.title("ResumeAnswerBot 💁")
    st.subheader("Ask any professional question related to me..")
    user_question = st.text_input("Type your question here:")

    if user_question:
        with st.spinner("Generating response..."):
            response = ChatManager.get_response(user_question)
            st.markdown("### Reply:")
            st.markdown(response)


if __name__ == "__main__":
    main()
