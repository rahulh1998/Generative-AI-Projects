import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
embeddings = HuggingFaceEmbeddings(model_name = 'All-MiniLM-L6-v2')

st.title("Conversational RAG with PDF")
st.write("Upload PDF's and chat with contents")

# Defining the model
llm = ChatGroq(model="Gemma2-9b-It")

# Get the session id
session_id = st.text_input("Session_id", value='default_user')

# Managing chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Choose a PDF file", type = "pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []

    for uploaded_file in uploaded_files:
        temppdf = f"./temp.pdf"
        with open(temppdf, 'wb') as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)


    # Chunking and creating embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 50)
    splits = text_splitter.split_documents(documents=documents)
    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vector_store.as_retriever()

    # Creating System Prompt
    contextualize_system_prompt = (
        "Given a chat history and the user question which might reference context in chat history,"
        "formulate a standalone question which can be understood without chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is"
    )

    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system",contextualize_system_prompt),
        MessagesPlaceholder("history"),
        ("human","{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)

    ## Answer question
    system_prompt = (
        "You are an assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer the question"
        "If you dont know the answer, simply say that you DO NOT know"
        "Use maximim of 3 sentences and keep the answer concise"
        "\n\n"
        "{context}"
    )

    question_answer_prompt = ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        MessagesPlaceholder("history"),
        ("human","{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm,question_answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_id.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Enter your question")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input":user_input},
            config={"configurable":{"session_id":session_id}}
        )
        st.write(st.session_state.store)
        st.write("Assistance:", response['answer'])
        st.write("Chat History:",session_history.messages)