import streamlit as st
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
DB_PATH = "text_database"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
MAX_SUMMARY_WORDS=8000
def summarizer(text,tokens):
    #here we are going to use the groq for the summarizeation task
    llm=ChatGroq(model='llama3-70b-8192')

    
    
    temp=PromptTemplate.from_template('''You are a summarizer, you have to  {tokens}\n\n
                                      
                                      and the give text is -- {text}--''')
    
    parser=StrOutputParser()
    chain=temp|llm|parser
    return chain.invoke({'text':text,'tokens':tokens})

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

def store_text_to_vector_db(text):
    chunks = get_text_chunks(text)
    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, encode_kwargs={"normalize_embeddings": True})
    vectordb = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    vectordb.save_local(DB_PATH)

def get_context_from_vector_db(query):
    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, encode_kwargs={"normalize_embeddings": True})
    vectordb = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    context_docs = vectordb.similarity_search(query)
    return [doc.page_content for doc in context_docs]

def generate_answer(query):
    context = get_context_from_vector_db(query)
    llm = ChatGroq(model='llama3-70b-8192')

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a helpful assistant. Use the context provided below to answer the question. 
        If the answer cannot be found in the context, say \"I don't know.\"

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"context": "\n".join(context), "question": query})



def add_footer():
    footer_html = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #555;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        z-index: 1000;
    }
    .footer a {
        color: #0366d6;
        text-decoration: none;
        margin: 0 10px;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        Developed by A. K. Saini | 
        <a href="https://github.com/aksaini2003" target="_blank">GitHub</a> | 
        <a href="https://www.linkedin.com/in/aashish-kumar-saini-03946b296/" target="_blank">LinkedIn</a>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
# Streamlit UI

st.set_page_config(page_title="Text Genie", layout="wide")

st.sidebar.title("Navigation Menu")
st.markdown("""
    <style>
    /* Style the sidebar title */
    .sidebar .sidebar-content {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Style for sidebar header and radio buttons */
    section[data-testid="stSidebar"] .css-1v0mbdj,  /* sidebar title */
    section[data-testid="stSidebar"] .css-1cpxqw2,  /* radio container */
    section[data-testid="stSidebar"] label {
        font-size: 20px !important;
        font-weight: 600;
        color: #2C3E50;
        padding: 8px 4px;
        margin-bottom: 6px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }

    /* Hover effect */
    section[data-testid="stSidebar"] label:hover {
        background-color: #f0f0f5;
        color: #1A5276;
        cursor: pointer;
    }

    /* Highlight selected option */
    section[data-testid="stSidebar"] input:checked + div {
        background-color: #1A5276 !important;
        color: white !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Go to", ["Question Answering System", "Summarizer"], index=0)
#for enhancing the navigation menu


add_footer()
if page == "Question Answering System":
    st.title("📄 Question Answering Chatbot")
    
    uploaded_files = st.file_uploader("Upload one or more .txt files", type=["txt"], accept_multiple_files=True)
    process_button = st.button("Process Text Files")

    full_text = ""

    if uploaded_files and process_button:
        for file in uploaded_files:
            full_text += file.read().decode("utf-8") + "\n"
        store_text_to_vector_db(full_text)
        st.success(f"{len(uploaded_files)} file(s) processed and indexed.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not os.path.exists("text_database"):
        st.info("Upload and process files to start asking questions.")
    else:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        question = st.chat_input("Ask a question about the uploaded text...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            answer = generate_answer(question)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
elif page == "Summarizer":
    st.title("📝 Text Summarizer")

    st.markdown(f"Paste or type up to **{MAX_SUMMARY_WORDS} words** below:")

    input_text = st.text_area("Enter your text here:", height=300)

    summary_size = st.selectbox(
        "Select summary size:",
        options=["Short (1-2 lines)", "Medium (1 paragraph)", "Detailed (multi-paragraph)"]
    )

    summarize_button = st.button("Summarize")

    if summarize_button:
        word_count = len(input_text.split())
        if word_count > MAX_SUMMARY_WORDS:
            st.warning(f"Text exceeds {MAX_SUMMARY_WORDS} word limit. Currently: {word_count} words.")
        elif word_count == 0:
            st.info("Please enter some text to summarize.")
        else:
            with st.spinner("Generating summary..."):

                # Add instructions based on summary size
                size_prompt = {
                    "Short (1-2 lines)": "Write a very short 1-2 line summary.",
                    "Medium (1 paragraph)": "Write a concise summary in one paragraph.",
                    "Detailed (multi-paragraph)": "Write a detailed multi-paragraph summary covering all important points."
                }

                llm = ChatGroq(model="llama3-70b-8192")
                summary=summarizer(input_text,size_prompt[summary_size])

                st.subheader("Summary:")
                st.write(summary)