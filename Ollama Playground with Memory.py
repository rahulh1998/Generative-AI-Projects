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


def query_model(model, prompt):
    """Run query on the selected Ollama model"""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    if result.returncode != 0:
        return f"❌ Error: {result.stderr}"
    return result.stdout.strip()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Ollama Chat", layout="wide")

st.title("🦙 Ollama Chat")
st.write("Chat with your locally installed Ollama models.")

# Sidebar for model selection
st.sidebar.header("⚙️ Settings")
models = get_installed_models()

if models:
    selected_model = st.sidebar.selectbox("Select a model", models)
    st.sidebar.success(f"✅ Using model: {selected_model}")
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
            </div><div style="clear:both;"></div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="text-align:left; padding:8px; border-radius:10px; 
            background-color:#f1f0f0; margin:5px 0; display:inline-block; float:left; clear:both; max-width:75%;">
            <b>{selected_model}:</b> {msg["content"]}
            </div><div style="clear:both;"></div>
            """,
            unsafe_allow_html=True
        )

# ----------------------------
# Chat input (auto-send on Enter, clears after send)
# ----------------------------
user_query = st.chat_input("💬 Type your message...")

if user_query:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_query})

    # Query model
    with st.spinner(f"Running `{selected_model}`..."):
        response = query_model(selected_model, user_query)

    # Add assistant response
    st.session_state["messages"].append({"role": "assistant", "content": response})

    # Force refresh so input clears and new messages show
    st.rerun()
