import os
import streamlit as st
import logging
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENVIRONMENT, PINECONE_INDEX_NAME
from langchain_openai import ChatOpenAI # type: ignore
from langchain_pinecone import PineconeVectorStore # type: ignore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec #type: ignore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import SecretStr

# Streamlit page setup
st.set_page_config(
    page_title="Interactive AI Assistant",  
    page_icon="🤖",  
    layout="wide",  
    initial_sidebar_state="expanded"  
)

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Initialize Pinecone
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in [index.name for index in pinecone_client.list_indexes()]:
    pinecone_client.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_API_ENVIRONMENT
        )
    )

# Connect to the Pinecone index
index = pinecone_client.Index(PINECONE_INDEX_NAME)

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_type=OPENAI_API_KEY)
embeddings = embedding_model

# Initialize Pinecone VectorStore
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Ensure the temp directory exists
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Function to load and index PDFs
def load_pdf_to_vector_db(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)
        vector_store.add_documents(documents=split_docs)
        return vector_store
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        return None

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# LLM setup
llm_chat = ChatOpenAI(
    temperature=0.9, 
    model='gpt-4o-mini', 
    api_key=SecretStr(OPENAI_API_KEY)
)

def qa_function(query):
    qa_prompt_template = """
    You are an assistant that helps the user with dynamic interactions.

    If context is provided below, use it to answer the question concisely.

    Context:
    {context}

    Question: {input}
    """

    qa_prompt = PromptTemplate.from_template(qa_prompt_template)
    qa_chain = create_stuff_documents_chain(llm_chat, qa_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vector_store.as_retriever(), combine_docs_chain=qa_chain)
    result = retrieval_chain.invoke({"input": query})
    return result["answer"], result.get("context", [])

# Function to display chat history with interactive elements
def display_chat():
    for message in st.session_state.history:
        role = message["role"]
        if role == "user":
            st.chat_message("user").markdown(message["content"])
        elif role == "assistant":
            st.chat_message("assistant").markdown(message["content"])

# Main Interface
st.markdown('<div class="main-title">🤖 Interactive AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload documents, ask questions, and receive instant personalized answers!</div>', unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.header("Upload PDF Files")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    sanitized_file_name = uploaded_file.name.lower().replace(" ", "_")
    temp_path = os.path.join(temp_dir, sanitized_file_name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    vector_store = load_pdf_to_vector_db(temp_path)
    st.sidebar.success(f"File '{sanitized_file_name}' successfully uploaded and indexed!")

if st.sidebar.button("🔄 Reset Conversation"):
    st.session_state.history.clear()

# Display chat
display_chat()

# Chat Input Handler
user_query = st.chat_input("💬 Type your question here...")

if user_query:
    st.session_state.history.append({"role": "user", "content": user_query})

    # Generate response
    with st.spinner("🔍 Generating response..."):
        response, docs = qa_function(query=user_query)

    # Fallback handling
    if response:
        st.session_state.history.append({"role": "assistant", "content": response})
    else:
        fallback_message = "Sorry, I couldn't understand your question. Could you please rephrase or ask something else?"
        logger.warning(f"Misinterpreted input: '{user_query}'")
        st.session_state.history.append({"role": "assistant", "content": fallback_message})

    # Redisplay chat after input
    display_chat()
