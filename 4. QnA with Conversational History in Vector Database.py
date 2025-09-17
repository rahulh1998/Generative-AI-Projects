import streamlit as st
import os
import uuid
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings

# Load environment
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = 'true'

# Streamlit config
st.set_page_config(page_title="QnA with Conversational History", page_icon="💬", layout="wide")
st.title("💬 QnA Chatbot with Persistent Conversational History (Chroma)")

# ---------------- LLM ----------------
llm = ChatGroq(model="groq/compound-mini", max_tokens=1024)

# ---------------- Embeddings + Chroma ----------------
embedding_model = OllamaEmbeddings(model="embeddinggemma:300m")
persist_directory = "vector_store"

chroma = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

# ---------------- Prompt Template ----------------
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# ---------------- Chain ----------------
chain = chat_prompt | llm | StrOutputParser()

# ---------------- Session State ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = "rahul"  # fixed user ID or uuid.uuid4().hex

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Message Store ----------------
# This will live in RAM during runtime
ram_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Return ChatMessageHistory, restoring from Chroma if empty."""
    if session_id not in ram_store:
        ram_store[session_id] = ChatMessageHistory()

        # Restore from Chroma on first load
        results = chroma.similarity_search(session_id, k=50)
        for doc in results:
            role = doc.metadata.get("role", "human")
            content = doc.page_content
            if role == "user":
                ram_store[session_id].add_user_message(content)
            else:
                ram_store[session_id].add_ai_message(content)

    return ram_store[session_id]

# Wrap with RunnableWithMessageHistory
chat_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

config = {"configurable": {"session_id": st.session_state.session_id}}

# ---------------- UI: Display Past Messages ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Handle User Input ----------------
if user_input := st.chat_input("Ask me anything..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from model
    response = chat_with_history.invoke({"question": user_input}, config=config)

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    # Persist messages into Chroma
    chroma.add_texts(
        [user_input],
        metadatas=[{"role": "user", "session_id": st.session_state.session_id}],
        ids=[f"user-{uuid.uuid4()}"]
    )
    chroma.add_texts(
        [response],
        metadatas=[{"role": "assistant", "session_id": st.session_state.session_id}],
        ids=[f"assistant-{uuid.uuid4()}"]
    )
    chroma.persist()
