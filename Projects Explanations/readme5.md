# 📄 Conversational RAG with PDF  

This project implements a **Conversational RAG (Retrieval-Augmented Generation)** pipeline using PDFs as a knowledge source.  
You can upload PDFs and then chat with their contents. The system remembers previous questions and answers, so it can handle follow-up queries in a natural way.

---

## 🚀 How It Works  

### 1. Embeddings  
We use **HuggingFaceEmbeddings (All-MiniLM-L6-v2)** to convert PDF text into vector representations.  
These embeddings allow us to search and retrieve the most relevant document chunks when answering user queries.

---

### 2. Upload & Process PDFs  
- PDFs are uploaded by the user.  
- They are read using **PyPDFLoader**.  
- The text is split into manageable chunks with **RecursiveCharacterTextSplitter**.  
- These chunks are stored in a **Chroma vector database** for retrieval.  

This preprocessing step ensures the retriever can efficiently search through the documents.

---

### 3. Retriever  
The **retriever** acts like a search engine over your PDF content.  
When a question is asked, it looks up the most relevant chunks from the vector database.  

This step ensures only relevant sections of the documents are considered for answering.

---

### 4. Contextualization System Prompt  
We define a **system prompt** that reformulates vague or incomplete follow-up questions into standalone ones.  

For example:  
- User: *"What about it?"*  
- Reformulated: *"What about Infosys mentioned earlier in the document?"*  

This improves the accuracy of retrieval since the query now makes sense without needing prior context.

---

### 5. History-Aware Retriever  
The **history-aware retriever** connects the language model with the retriever and the contextualization prompt.  

It uses the LLM to transform follow-up questions into complete queries before retrieving documents.  
This allows conversations to flow naturally while still pulling the right context.

---

### 6. Question-Answer System Prompt  
We define a second system prompt to guide the LLM when generating answers:  
- Only use the retrieved context.  
- Keep answers concise (max 3 sentences).  
- If the answer is not found, say *“I don’t know.”*  

This ensures responses are grounded in the provided documents rather than hallucinated.

---

### 7. Question-Answer Chain  
The **question-answer chain** combines the retrieved chunks with the user’s question and passes them to the LLM.  

The LLM then generates the final answer, restricted by the system prompt’s rules.  

---

### 8. RAG Chain  
The **retrieval chain (RAG)** connects everything together:  
1. Reformulate the user’s question if necessary.  
2. Retrieve the most relevant chunks.  
3. Use the LLM to answer the question.  

This is the core pipeline that powers the whole application.

---

### 9. Chat History Manager  
To enable conversational memory, we use a **chat history manager**.  
Each session has its own chat history, meaning different users (or different sessions) can have independent conversations.  

This history is used by both the retriever and the LLM to handle follow-up questions naturally.

---

### 10. RunnableWithMessageHistory  
We wrap the RAG chain in **RunnableWithMessageHistory**.  

- This ensures that both the **inputs (user queries)** and the **outputs (AI answers)** are stored in the chat history.  
- The history is passed automatically into the pipeline every time the user asks a new question.  

This allows the system to provide context-aware answers across multiple turns in a conversation.

---

### 11. Streamlit User Interface  
Finally, everything is tied together with a **Streamlit app**:  
- Title and description shown at the top.  
- File uploader for PDFs.  
- Text input for session ID (to separate conversations).  
- Text input for user questions.  
- Display of answers and chat history in the UI.  

📌 With this, the project becomes interactive and easy to use directly from the browser.

---

## ✅ Summary  

1. Convert PDF text into embeddings.  
2. Store embeddings in a Chroma vector store.  
3. Retrieve relevant chunks for each query.  
4. Reformulate vague questions using chat history.  
5. Pass context + question to LLM with clear instructions.  
6. Generate concise, context-aware answers.  
7. Maintain separate histories for different sessions.  
8. Wrap everything with `RunnableWithMessageHistory` for memory.  
9. Deploy the full pipeline as a **Streamlit web app**.  

This creates a powerful **Conversational RAG system with memory** that allows smooth interaction with uploaded PDFs.

---
flowchart TD
    A[User Uploads PDFs] --> B[PyPDFLoader Loads Text]
    B --> C[RecursiveCharacterTextSplitter Splits Text]
    C --> D[HuggingFace Embeddings Create Vectors]
    D --> E[Chroma Vector Store]

    F[User Question] --> G[History-Aware Retriever]
    G --> H[Reformulated Question]
    H --> I[Retriever Fetches Relevant Chunks]
    I --> J[QA Prompt + LLM]
    J --> K[Answer Generated]

    E --> I
    K --> L[Assistant Response]
    L --> M[Chat History Stored]
    M --> G