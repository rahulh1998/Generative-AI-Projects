# 📄 RAG Q&A Conversation With PDF (Including Chat History)

An interactive **Streamlit-based Conversational RAG (Retrieval-Augmented Generation)** app that allows you to **upload PDFs, ask questions about their content, and maintain multi-turn conversational history**.  
It leverages **LangChain, HuggingFace embeddings, Chroma vector database, and Groq LLMs** to provide context-aware answers.

---

## 🚀 How It Works

The project is organized into several key steps:

### 1. **Environment & Setup**
- Loads environment variables (e.g., HuggingFace API token).  
- Initializes the **HuggingFace sentence transformer embeddings** (`all-MiniLM-L6-v2`) for semantic vectorization.

### 2. **Streamlit UI**
- **Title & Description**: Displays the app header and short instructions.  
- **Groq API Key Input**: Requires the user to provide their **Groq API key**.  
- **Session ID Input**: Allows multiple conversations to be maintained separately.  
- **PDF Uploader**: Supports uploading one or more PDF files.  

### 3. **PDF Processing & Embeddings**
- Each uploaded PDF is temporarily stored.  
- `PyPDFLoader` extracts text from the PDF.  
- `RecursiveCharacterTextSplitter` splits documents into overlapping chunks for better retrieval.  
- `Chroma` vector store indexes the chunks using HuggingFace embeddings.  
- A **Retriever** is created from the vector store for semantic search.

### 4. **Retriever with Context Awareness**
- A **History-Aware Retriever** is created using `create_history_aware_retriever`.  
- It reformulates the latest user query into a **standalone question** (if necessary) by leveraging chat history.

### 5. **Question Answering Chain**
- Defines a concise **system prompt**:  
  > Use retrieved context to answer questions in **three sentences or less**. If unknown, say you don’t know.  
- Uses `create_stuff_documents_chain` to generate answers with retrieved context.  
- Wraps this into a **Retrieval Chain** (`create_retrieval_chain`) for seamless Q&A.

### 6. **Chat History Management**
- `ChatMessageHistory` is used to store and retrieve conversation history.  
- Each **session_id** maintains its own conversation state in `st.session_state.store`.  
- `RunnableWithMessageHistory` links the RAG chain with stored chat history so conversations remain contextual.

### 7. **User Interaction**
- Users can enter a question in the input box.  
- The system retrieves relevant chunks from uploaded PDFs and provides concise, context-aware answers.  
- Full **chat history** is displayed after every turn.

---

## 🔑 Key Components

- **Models**:  
  Uses Groq’s `Gemma2-9b-It` model for inference.  

- **Retriever**:  
  - Built with **Chroma** vector DB.  
  - History-aware retriever reformulates follow-up queries for accuracy.  

- **Prompts**:  
  - **Contextualization Prompt**: Reformulates user questions into standalone queries.  
  - **QA Prompt**: Ensures concise, context-based answers.  

- **Chains**:  
  - **Retrieval Chain** for fetching relevant context.  
  - **Stuff Documents Chain** for combining retrieved chunks into responses.  

- **Memory**:  
  - **ChatMessageHistory** for persisting conversations.  
  - **RunnableWithMessageHistory** to integrate history into the RAG pipeline.  

- **UI**:  
  Simple Streamlit interface for PDF upload, chat input, and displaying responses + conversation history.

---

## 📋 Summary of Workflow

1. User enters **Groq API Key** and uploads one or more PDFs.  
2. PDFs are processed → split into chunks → embedded → stored in Chroma DB.  
3. User asks a question.  
4. History-aware retriever reformulates the query if needed.  
5. Retriever fetches relevant chunks from PDFs.  
6. The LLM generates a concise answer using the context.  
7. Answer and conversation history are displayed.  
8. Multiple sessions can be maintained with unique session IDs.  

---

## ⚡ Quickstart: How to Run

### 1. **Install Dependencies**
Make sure you have the following installed:
- [Python 3.9+](https://www.python.org/downloads/)  
- [Streamlit](https://streamlit.io/)  
- [LangChain](https://www.langchain.com/)  
- [Chroma](https://docs.trychroma.com/)  
- [HuggingFace Hub](https://huggingface.co/)  
- [Groq](https://groq.com/)  

Install dependencies:
```bash
pip install streamlit langchain langchain-community langchain-chroma langchain-huggingface langchain-groq chromadb
