
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st


load_dotenv()
os.environ['HF_TOKEN']=st.secrets['HF_TOKEN']
groq_api_key = st.secrets['GROQ_API_KEY']

#Step1: LLM model

llm_model = ChatGroq(model='Llama3-8b-8192',groq_api_key=groq_api_key)
llm_model.invoke([HumanMessage(content="tell me joke")])

#Step2: RAG implementation
# Data injection and chunking
#Document Embedding and Vectore stores
def create_vectore_store():
    if "vectore_store" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("wars_pdf")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("World wars Q&A with Llama3 based AI")
st.write("First Press start and wait for AI to be trainied")
st.write("Trainied on only small data of world wars.")
user_prompt = st.text_input("Enter question regarding world wars")

if st.button("Start"):
    create_vectore_store()
    st.write("Trainied and ready, please enter questions!")

#Step3: Chains and Retrevials
prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """

)

if user_prompt:
    document_chain=create_stuff_documents_chain(llm_model,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    response = retrieval_chain.invoke({'input': user_prompt})

    st.write(response['answer'])

    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')


