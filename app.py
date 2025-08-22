import os
import streamlit as st
import logging
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENVIRONMENT, PINECONE_INDEX_NAME
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec # type: ignore
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain.prompts import PromptTemplate


# Streamlit page config
st.set_page_config(
    page_title="AI Q&A Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# Load env variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Initialize Pinecone
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in [index.name for index in pinecone_client.list_indexes()]:
    pinecone_client.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_API_ENVIRONMENT)
    )
index = pinecone_client.Index(PINECONE_INDEX_NAME)

# Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Ensure temp directory exists
temp_dir = "temp"
os.makedirs(temp_dir, exist_ok=True)

# Initialize conversation history in session
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize vector store in session
if "vector_store" not in st.session_state:
    st.session_state.vector_store = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

# LLM setup
llm_chat = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
    api_key=SecretStr(OPENAI_API_KEY)
)

# Function to load and index PDF
def load_pdf_to_vector_db(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        # Index into Pinecone
        st.session_state.vector_store = PineconeVectorStore.from_documents(
            documents=split_docs,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME
        )
        logger.info(f"✅ Indexed {len(split_docs)} chunks from PDF.")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading PDF: {e}")
        return False

# Prompt template: combines context (if available) with general answering ability
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent assistant. Use the provided context to answer the question. 
If the context does not contain the answer, still try to provide a helpful and relevant response 
using your general knowledge.

Context:
{context}

Question: {question}

Answer:
"""
)

def qa_function(query):
    vector_store = st.session_state.get("vector_store")

    try:
        docs = []
        if vector_store:
            retriever = vector_store.as_retriever(search_kwargs={"k": 10})
            docs = retriever.get_relevant_documents(query)

        # Join retrieved docs into context text
        context_text = "\n\n".join([d.page_content for d in docs]) if docs else "No context available."

        chain = load_qa_chain(llm=llm_chat, chain_type="stuff", prompt=qa_prompt)

        logger.info(f"🔍 Retrieved {len(docs)} documents for query: {query[:50]}...")

        response = chain.run(input_documents=docs, question=query)
        return response or "Sorry, I could not generate an answer."

    except Exception as e:
        logger.error(f"❌ Error generating answer: {e}")
        return "Sorry, there was an error processing your request."

    vector_store = st.session_state.get("vector_store")
    if not vector_store:
        return "No documents available. Please upload a PDF first."
    
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant documents found for this question."

        # ✅ better chain type (map_reduce is usually more reliable than stuff)
        chain = load_qa_chain(llm=llm_chat, chain_type="map_reduce")

        # Add logging for debugging
        logger.info(f"🔍 Retrieved {len(docs)} documents for query: {query[:50]}...")
        
        response = chain.run(input_documents=docs, question=query)
        return response or "Sorry, I could not generate an answer."
    
    except Exception as e:
        logger.error(f"❌ Error generating answer: {e}")
        return "Sorry, there was an error processing your request."
# Streamlit UI
st.markdown("<h1 style='text-align:center;color:#4CAF50'>🤖 AI Q&A Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#555'>Upload your PDF and ask questions to get instant answers!</p>", unsafe_allow_html=True)

# Sidebar: upload PDF
st.sidebar.header("Upload PDF Files")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    sanitized_file_name = uploaded_file.name.lower().replace(" ", "_")
    temp_path = os.path.join(temp_dir, sanitized_file_name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    success = load_pdf_to_vector_db(temp_path)
    if success:
        st.sidebar.success(f"File '{sanitized_file_name}' successfully indexed!")

if st.sidebar.button("🔄 Reset Conversation"):
    st.session_state.history.clear()

# Chat input
user_query = st.chat_input("💬 Type your question here...")
if user_query:
    st.session_state.history.append({"role": "user", "content": user_query})
    with st.spinner("🔍 Generating response..."):
        answer = qa_function(user_query)
    st.session_state.history.append({"role": "assistant", "content": answer})

# Display chat
for message in st.session_state.history:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    else:
        st.chat_message("assistant").markdown(message["content"])
