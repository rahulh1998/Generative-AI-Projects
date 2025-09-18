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
st.set_page_config(page_title="GenAI Playground", page_icon="🤖", layout="wide")
st.title("🤖 GenAI Playground — Powered by Groq & LangChain")

# Sidebar navigation
menu = st.sidebar.radio("Choose a Mode:", [
    "💬 Chatbot with Memory",
    "📄 Smart Document QnA",
    "🎨 Creative Playground",
    "🔍 Semantic Search",
    "⚡ Quick Tools"
])

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

ram_store = {}

# ---------------- Message History ----------------
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in ram_store:
        ram_store[session_id] = ChatMessageHistory()
        results = chroma.similarity_search(session_id, k=50)
        for doc in results:
            role = doc.metadata.get("role", "human")
            content = doc.page_content
            if role == "user":
                ram_store[session_id].add_user_message(content)
            else:
                ram_store[session_id].add_ai_message(content)
    return ram_store[session_id]

chat_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

config = {"configurable": {"session_id": st.session_state.session_id}}

# ---------------- Features ----------------
if menu == "💬 Chatbot with Memory":
    st.subheader("💬 Conversational AI with Memory")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response = chat_with_history.invoke({"question": user_input}, config=config)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        chroma.add_texts([user_input], metadatas=[{"role": "user", "session_id": st.session_state.session_id}], ids=[f"user-{uuid.uuid4()}"])
        chroma.add_texts([response], metadatas=[{"role": "assistant", "session_id": st.session_state.session_id}], ids=[f"assistant-{uuid.uuid4()}"])
        chroma.persist()

elif menu == "📄 Smart Document QnA":
    st.subheader("📄 Upload a Document and Ask Questions")
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        chroma.add_texts([content], metadatas=[{"role": "doc", "session_id": st.session_state.session_id}], ids=[f"doc-{uuid.uuid4()}"])
        st.success("Document added to knowledge base!")

    question = st.text_input("Ask a question about your docs:")
    if question:
        docs = chroma.similarity_search(question, k=5)
        context = "\n".join([d.page_content for d in docs])
        final_prompt = f"Answer based on the docs below:\n{context}\n\nQuestion: {question}"
        answer = llm.invoke(final_prompt)
        st.markdown(f"**Answer:** {answer.content}")

elif menu == "🎨 Creative Playground":
    st.subheader("🎨 Creative AI Fun")
    option = st.selectbox("Choose a style:", ["Joke", "Poem", "Story", "Rap Song"])
    topic = st.text_input("Enter a topic:")
    if st.button("Generate") and topic:
        prompt = f"Write a {option.lower()} about {topic} in a funny and creative way."
        response = llm.invoke(prompt)
        st.markdown(response.content)

elif menu == "🔍 Semantic Search":
    st.subheader("🔍 Search Across Your Conversations/Docs")
    query = st.text_input("Search query:")
    if query:
        results = chroma.similarity_search(query, k=5)
        for i, r in enumerate(results):
            st.markdown(f"**Result {i+1}:** {r.page_content}")

elif menu == "⚡ Quick Tools":
    st.subheader("⚡ AI-Powered Quick Tools")
    tool = st.selectbox("Select a tool:", ["Summarizer", "Keyword Extractor", "Sentiment Analyzer"])
    text_input = st.text_area("Paste text here:")

    if st.button("Run") and text_input:
        if tool == "Summarizer":
            prompt = f"Summarize the following text in simple words:\n{text_input}"
        elif tool == "Keyword Extractor":
            prompt = f"Extract the top 5 keywords from this text:\n{text_input}"
        else:
            prompt = f"Analyze the sentiment (positive, negative, neutral) of this text:\n{text_input}"

        response = llm.invoke(prompt)
        st.markdown(f"**Output:** {response.content}")