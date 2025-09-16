import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = 'true'

st.set_page_config(page_title="QnA with Conversational History", page_icon="💬"
                   , layout="wide"
                   )

st.title("💬 QnA Chatbot with Conversational History")

# Define LLM
groq_model = "groq/compound-mini"
llm = ChatGroq(model=groq_model, max_tokens=1024)

# Get question from user
question = st.text_input("Enter your question:", key="question_input")

# Define prompt
chat_prompt =ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant"),
    MessagesPlaceholder(variable_name="history"),
    ("human","{question}")
])

# Define store for histories
store = {}


# Create a chain to handle the prompt and LLM
chain = chat_prompt | llm | StrOutputParser()

# Display the answer
if question:
    st.write(chain.invoke(question))
