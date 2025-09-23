import streamlit as st
import subprocess

# ----------------------------
# Utility functions
# ----------------------------
def get_installed_models():
    """Fetch installed Ollama models"""
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    if result.returncode != 0:
        st.error("❌ Error fetching models from Ollama")
        return []

    lines = result.stdout.strip().split("\n")
    models = []
    for line in lines[1:]:  # Skip header row
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def stream_query(model, prompt):
    """Stream response from the selected Ollama model"""
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    # Send the prompt
    process.stdin.write(prompt)
    process.stdin.close()

    # Yield output as it comes in
    for line in process.stdout:
        yield line

    process.stdout.close()
    process.wait()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Ollama Multi-Model Chat", layout="wide")

st.title("🦙 Ollama Multi-Model Chat (Streaming)")
st.write("Chat with multiple Ollama models. Switch models anytime — each reply remembers which model generated it.")

# Sidebar for model selection
st.sidebar.header("⚙️ Settings")
models = get_installed_models()

if models:
    selected_model = st.sidebar.selectbox("Select a model", models)
    st.sidebar.success(f"✅ Current model: {selected_model}")
else:
    st.sidebar.warning("⚠️ No models found. Please install Ollama models first.")
    st.stop()

# Clear chat button
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state["messages"] = []
    st.rerun()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style="text-align:right; padding:8px; border-radius:10px; 
            background-color:#DCF8C6; margin:5px 0; display:inline-block; float:right; clear:both; max-width:75%;">
            <b>You:</b> {msg["content"]}
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="text-align:left; padding:8px; border-radius:10px; 
            background-color:#f1f0f0; margin:5px 0; display:inline-block; float:left; clear:both; max-width:75%;">
            <b>{msg["model"]}:</b> {msg["content"]}
            """,
            unsafe_allow_html=True
        )

# ----------------------------
# Chat input with streaming response
# ----------------------------
user_query = st.chat_input("💬 Type your message...")

if user_query:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_query})

    # Placeholder for streaming response
    with st.spinner(f"Running `{selected_model}`..."):
        response_placeholder = st.empty()
        streamed_text = ""

        for chunk in stream_query(selected_model, user_query):
            streamed_text += chunk
            response_placeholder.markdown(
                f"""
                <div style="text-align:left; padding:8px; border-radius:10px; 
                background-color:#f1f0f0; margin:5px 0; display:inline-block; float:left; clear:both; max-width:75%;">
                <b>{selected_model}:</b> {streamed_text}
                """,
                unsafe_allow_html=True
            )

    # Add final assistant response with model name
    st.session_state["messages"].append(
        {"role": "assistant", "content": streamed_text, "model": selected_model}
    )

    # Refresh so response stays in history
    st.rerun()
