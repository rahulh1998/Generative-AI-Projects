import validators
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# Streamlit App
st.set_page_config(page_title="LangChain: Website Link Summarizer", page_icon= '💥', layout='wide')
st.title("LangChain: Link Summarizer")
st.subheader("Summarise URL's")

# llama-3.1-8b Model
llm = ChatGroq(model="llama-3.1-8b-instant")

# Initialising Prompt
template = """
    Provide the summary of the following content in 400 words
    {text}
"""
prompt = PromptTemplate(template=template,
                        input_variables=['text'])

# Get Groq API Key
# with st.sidebar:
#     api_key = st.text_input("Groq Api Key", value='', type='password')

url = st.text_input("Enter URL", label_visibility="collapsed")

if st.button("Summarize the content of Youtube or Website"):
    if not url.strip():
        st.error("URL Field is Empty")
    elif not validators.url(url):
        st.error("Please enter a valid URL")

    else:
        try:
            with st.spinner("Waiting..."):
                # Loading the data
                if 'youtube.com' in url:
                    loader = YoutubeLoader.from_youtube_url(url, 
                                                            add_video_info = True)
                else:
                    loader = UnstructuredURLLoader(urls = [url] , 
                                                   ssl_verify = True)
                docs = loader.load()

                # Chain Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt = prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)

        except Exception as e: 
            st.error(f"Exception : {e}")