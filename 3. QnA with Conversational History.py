import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

# Load env
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Streamlit
st.set_page_config(page_title="QnA with Conversational History", page_icon="💬", layout="wide")
st.title("💬 QnA Chatbot with Conversational History")

# LLM
llm = ChatGroq(model="groq/compound-mini", max_tokens=1024)

# Prompt
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# Chain (prompt -> llm -> parse to str)
chain = chat_prompt | llm | StrOutputParser()

# Ensure session_state containers exist
if "session_id" not in st.session_state:
    st.session_state.session_id = "rahul"  # or use uuid.uuid4().hex for unique sessions

if "messages" not in st.session_state:
    # keep UI copy of the messages (simple list of dicts) to render chat
    st.session_state.messages = []

if "history_store" not in st.session_state:
    # will hold ChatMessageHistory objects per session_id
    st.session_state.history_store = {}

# get_session_history persists the ChatMessageHistory inside session_state
def get_session_history(session_id: str) -> ChatMessageHistory:
    store = st.session_state.history_store
    if session_id not in store:
        # create new ChatMessageHistory and restore from st.session_state.messages (if any)
        hist = ChatMessageHistory()
        for m in st.session_state.get("messages", []):
            if m.get("role") == "user":
                hist.add_user_message(m["content"])
            else:
                hist.add_ai_message(m["content"])
        store[session_id] = hist
    return store[session_id]

# Wrap with RunnableWithMessageHistory
chat_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

config = {"configurable": {"session_id": st.session_state.session_id}}

# Render previous messages (UI)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask me anything..."):
    # show user message in UI and append to messages list
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # invoke LLM with history (RunnableWithMessageHistory will append to ChatMessageHistory)
    try:
        with st.spinner("Thinking..."):
            response = chat_with_history.invoke({"question": user_input}, config=config)
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        raise

    # append assistant reply to UI messages and render it
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
