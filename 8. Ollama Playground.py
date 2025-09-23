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
        encoding="utf-8",   # ensure UTF-8
        errors="replace"    # replace unsupported chars
    )
    if result.returncode != 0:
        return f"❌ Error: {result.stderr}"
    return result.stdout.strip()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Ollama Playground", layout="wide")

st.title("🦙 Ollama Playground")
st.write("Run queries on your locally installed Ollama models.")

# Sidebar for model selection
st.sidebar.header("⚙️ Settings")
models = get_installed_models()

if models:
    selected_model = st.sidebar.selectbox("Select a model", models)
    st.sidebar.success(f"✅ Using model: {selected_model}")
else:
    st.sidebar.warning("⚠️ No models found. Please install Ollama models first.")
    st.stop()


# Input area
user_query = st.text_area("💬 Enter your query:", height=150, placeholder="Ask me anything...")

# Run query button
if st.button("🚀 Run Query", type="primary"):
    if user_query.strip() == "":
        st.warning("Please enter a query before running.")
    else:
        with st.spinner(f"Running model `{selected_model}`..."):
            response = query_model(selected_model, user_query)

        # Display chat-style output
        st.subheader("📢 Model Response")
        st.markdown(
            f"""
            <div style="padding:10px; border-radius:10px; background-color:#f5f5f5; border:1px solid #ddd;">
            {response}
            </div>
            """,
            unsafe_allow_html=True
        )
