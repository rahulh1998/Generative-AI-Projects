# 🦙 Ollama Multi-Model Chat (Streaming)

An interactive **Streamlit-based chat interface** for running and comparing multiple **Ollama language models**.  
This app allows you to query different models, stream responses in real-time, and keep a persistent conversation history — all inside a clean and intuitive chat UI.

---

## 🚀 How It Works

The project is structured into **three main sections**:

### 1. **Utility Functions**
- **Model Discovery**:  
  The app runs `ollama list` in the background to fetch all installed Ollama models and displays them in the sidebar for selection.  
  If no models are found, the app stops gracefully with a warning.

- **Streaming Query Execution**:  
  User prompts are sent to the chosen model via `subprocess.Popen`.  
  The output is captured line by line, streamed back to the UI, and displayed progressively for a real-time chat experience.

---

### 2. **Streamlit User Interface**
- **Page Setup**:  
  Defines the title, layout, and sidebar settings.

- **Sidebar**:  
  - Model selector dropdown (auto-populated from installed models).  
  - "Clear Chat" button to reset the session state.  
  - Current model indicator.

- **Chat Display**:  
  - Messages are styled like a chat app.  
  - **User messages** appear on the right with green bubbles.  
  - **Model responses** appear on the left with grey bubbles and are tagged with the model’s name.

---

### 3. **Chat Workflow**
- **User Input**:  
  Uses `st.chat_input` for a conversational experience.  

- **Streaming Response**:  
  - As soon as a user enters a message, it’s appended to the session history.  
  - The chosen model processes the query, and the response is displayed incrementally in the chat window.  
  - Once completed, the full response is saved in session state for persistence.

- **Session Memory**:  
  The app stores the entire chat history in `st.session_state`, so users can scroll through past messages even after multiple queries.

---

## 🔑 Key Components

- **Models**  
  Powered by Ollama — supports multiple locally installed LLMs.  
  You can switch between them anytime via the sidebar.

- **Retriever & Prompt Handling**  
  Prompts are directly sent to the backend model process. No additional retriever logic is applied in this version.

- **Chains & Memory**  
  - The app uses Streamlit’s `session_state` to maintain chat history.  
  - Each assistant response is tagged with the **model name** to avoid confusion when switching between models.

- **UI**  
  Built entirely with **Streamlit**, featuring styled chat bubbles for a modern conversational experience.  
  Real-time streaming makes the model responses feel interactive and natural.

---

## 📋 Summary of Workflow

1. User selects a model from the sidebar.  
2. User enters a query in the chat input.  
3. The query is sent to the selected Ollama model.  
4. The response streams back in real-time and is displayed in the UI.  
5. Both user and model messages are stored in session memory for a seamless chat history.  
6. The user can clear the chat and start fresh at any point.

---

## ⚡ Quickstart: How to Run

### 1. **Install Dependencies**
Make sure you have the following installed:
- [Python 3.8+](https://www.python.org/downloads/)  
- [Streamlit](https://streamlit.io/)  
- [Ollama](https://ollama.ai/)

Install Streamlit:
```bash
pip install streamlit
