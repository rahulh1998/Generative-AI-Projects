# 💬 QnA Chatbot with Persistent Conversational History

This project is a **Streamlit-based conversational chatbot** that combines **LLMs, embeddings, and vector storage** to enable **question answering with long-term memory**. Unlike a typical chatbot that forgets past interactions, this system saves conversation history into a **Chroma vector database** and restores it across sessions, making interactions more natural and context-aware.

---

## 🚀 How It Works (Step-by-Step)

### 1. **Environment Setup**
- Loads required API keys (`GROQ_API_KEY`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`) from `.env`.
- Enables **LangChain tracing** for better observability.

### 2. **Streamlit UI Initialization**
- Configures the app with a title, icon, and wide layout.
- Sets up a **chat interface** where both user and assistant messages are displayed in a conversational style.

### 3. **Language Model (LLM)**
- Uses **ChatGroq** with the `groq/compound-mini` model.
- Limited to **1024 tokens per response** for concise outputs.

### 4. **Embeddings & Vector Store**
- Uses **Ollama embeddings (`embeddinggemma:300m`)** to convert text into vector representations.
- Stores and retrieves vectors via **ChromaDB**, persisted in a `vector_store/` directory.
- Ensures conversations are **retrievable across sessions**.

### 5. **Prompt Template**
- Defines a structured prompt using `ChatPromptTemplate`:
  - **System role**: instructs the model to act as a helpful assistant.
  - **History placeholder**: injects previous messages for context.
  - **Human role**: captures the latest user question.

### 6. **Chain Assembly**
- Combines the **prompt → LLM → output parser** into a processing pipeline.
- Ensures consistent input/output flow.

### 7. **Session Management**
- Assigns each user a **session ID** (fixed or via `uuid`).
- Maintains `st.session_state.messages` for the current session.
- Uses **`ram_store`** (in-memory) plus **Chroma persistence** for hybrid memory.

### 8. **Message History Restoration**
- On first load, checks if session history exists.
- If empty, **retrieves up to 50 past messages from Chroma** and reconstructs chat memory (`ChatMessageHistory`).
- Allows seamless continuation of previous conversations.

### 9. **Conversation Flow**
- Displays all past messages in the chat interface.
- Accepts user input via `st.chat_input`.
- Passes input into the **RunnableWithMessageHistory** chain.
- Returns model response and updates both:
  - **UI messages** (displayed immediately).
  - **ChromaDB** (persisted for long-term memory).

### 10. **Persistence**
- Each new message (user + assistant) is saved to Chroma with:
  - Text content
  - Role metadata (`user` / `assistant`)
  - Session ID
- Database is persisted to disk (`chroma.persist()`).

---

## 🔑 Key Components

- **LLM**: `ChatGroq` (Groq-powered model for responses).
- **Embeddings**: `OllamaEmbeddings` (`embeddinggemma:300m`) for semantic search & memory.
- **Vector Store**: `ChromaDB` for storing/retrieving conversation history.
- **Prompts**: `ChatPromptTemplate` with system, history, and user slots.
- **Memory**: 
  - **In-memory RAM store** for fast access during runtime.
  - **Chroma persistence** for restoring past conversations.
- **Chain**: `RunnableWithMessageHistory` manages conversational context seamlessly.
- **UI**: Streamlit chat components (`st.chat_message`, `st.chat_input`) for interactive frontend.

---

## 📊 Workflow Summary

1. **User enters a question** in the Streamlit chatbox.  
2. **Session history is retrieved** from in-memory or Chroma vector store.  
3. **Prompt is assembled** with system instructions, history, and new question.  
4. **LLM generates a response** based on context.  
5. **Response is displayed** in the chat UI.  
6. **Both user and assistant messages are persisted** in Chroma for long-term storage.  
7. On next interaction (or reload), the **chat history is restored**, allowing continuity.  

---

## ⚡ Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
