# import os
# import streamlit as st
# import logging
# from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENVIRONMENT, PINECONE_INDEX_NAME
# from langchain_openai import ChatOpenAI # type: ignore
# from langchain_pinecone import PineconeVectorStore # type: ignore
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.chains.question_answering import load_qa_chain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
# from pinecone import Pinecone, ServerlessSpec #type: ignore
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# from pydantic import SecretStr

# st.set_page_config(
#     page_title="AI Q&A Assistant",  # Updated title
#     page_icon="🤖",  # Updated icon
#     layout="centered",  # Options: "centered" or "wide"
#     initial_sidebar_state="auto"  # Options: "auto", "expanded", "collapsed"
# )

# # Load environment variables and set OpenAI API key
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# # Initialize logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger()

# # Initialize Pinecone
# pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
# if PINECONE_INDEX_NAME not in [index.name for index in pinecone_client.list_indexes()]:
#     pinecone_client.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=1536,
#         metric='cosine',
#         spec=ServerlessSpec(
#             cloud="aws",
#             region=PINECONE_API_ENVIRONMENT
#         )
#     )

# # Connect to the Pinecone index
# index = pinecone_client.Index(PINECONE_INDEX_NAME)

# # Initialize embedding model
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


# # Initialize Pinecone VectorStore
# vector_store = PineconeVectorStore.from_existing_index(
#     index_name=PINECONE_INDEX_NAME,
#     embedding=embeddings
# )

# # Ensure the temp directory exists
# temp_dir = "temp"
# if not os.path.exists(temp_dir):
#     os.makedirs(temp_dir)

# # Function to load and index PDFs
# # Function to load and index PDFs
# def load_pdf_to_vector_db(pdf_path):
#     global vector_store
#     try:
#         loader = PyPDFLoader(pdf_path)
#         documents = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         split_docs = text_splitter.split_documents(documents)

#         # Index documents into Pinecone
#         vector_store = PineconeVectorStore.from_documents(
#             documents=split_docs,
#             embedding=embeddings,
#             index_name=PINECONE_INDEX_NAME
#         )
#         return vector_store
#     except Exception as e:
#         logger.error(f"Error loading PDF: {e}")
#         return None

# # Initialize conversation history
# if "history" not in st.session_state:
#     st.session_state.history = []

# # LLM setup
# llm_chat = ChatOpenAI(
#     temperature=0.9, 
#     model='gpt-4o-mini', 
#     api_key=SecretStr(OPENAI_API_KEY)
# )

# def qa_function(query):
#     if vector_store is None:
#         return "No documents available. Please upload a PDF first.", []

#     qa_prompt = PromptTemplate.from_template("""
#     You are an assistant for answering questions.

#     Context:
#     {context}

#     Question: {input}
#     """)

#     qa_chain = create_stuff_documents_chain(llm_chat, qa_prompt)
#     retriever = vector_store.as_retriever(search_kwargs={"k": 2})
#     retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=qa_chain)

#     result = retrieval_chain.invoke({"input": query})
#     return result["answer"], result.get("context", [])


# # Function to generate responses
# def response_generator(query):
#     try:
#         if vector_store is None:
#             logger.error("Vector store is not initialized. Please upload a PDF first.")
#             return None, None
#         docs = vector_store.similarity_search(query, k=2)
#         chain = load_qa_chain(llm=llm_chat, chain_type="stuff")
#         response = chain.run(input_documents=docs, question=query)
#         return response, docs
#     except Exception as e:
#         logger.error(f"Error generating response for query '{query}': {e}")
#         return None, None

# # Function to display chat history
# def display_chat():
#     for message in st.session_state.history:
#         role = message["role"]
#         if role == "user":
#             st.chat_message("user").markdown(message["content"])
#         elif role == "assistant":
#             st.chat_message("assistant").markdown(message["content"])

# # Main Interface
# st.markdown(
#     """
#     <style>
#     .main-title {
#         color: #4CAF50;
#         font-size: 36px;
#         font-weight: bold;
#         text-align: center;
#         margin-bottom: 20px;
#     }
#     .sub-header {
#         font-size: 18px;
#         text-align: center;
#         margin-bottom: 30px;
#         color: #555555;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# st.markdown('<div class="main-title">🤖 AI Q&A Assistant</div>', unsafe_allow_html=True)
# st.markdown('<div class="sub-header">Upload your documents and get instant answers to your questions!</div>', unsafe_allow_html=True)


# #Sidebar for file upload
# st.sidebar.header("Upload PDF Files")
# uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
# if uploaded_file:
#     sanitized_file_name = uploaded_file.name.lower().replace(" ", "_")
#     temp_path = os.path.join(temp_dir, sanitized_file_name)
#     with open(temp_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # Load and store in session state
#     st.session_state['vector_store'] = load_pdf_to_vector_db(temp_path)
#     st.sidebar.success(f"File '{sanitized_file_name}' successfully uploaded and indexed!")

# if st.sidebar.button("🔄 Reset Conversation"):
#     st.session_state.history.clear()

# # Display chat


# # Chat Input Handler
# user_query = st.chat_input("💬 Type your question here...")

# if user_query:
#     # Append the user's input to the history
#     st.session_state.history.append({"role": "user", "content": user_query})

#     # Generate response
#     with st.spinner("🔍 Generating response..."):
#         response, docs = qa_function(query=user_query)

#     # Fallback handling
#     if response:
#         st.session_state.history.append({"role": "assistant", "content": response})
#     else:
#         fallback_message = "Sorry, I couldn't understand your question. Could you please rephrase or ask something else?"
#         logger.warning(f"Misinterpreted input: '{user_query}'")
#         st.session_state.history.append({"role": "assistant", "content": fallback_message})
# display_chat()


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
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from pydantic import SecretStr

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
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)
        
        # Index into Pinecone
        st.session_state.vector_store = PineconeVectorStore.from_documents(
            documents=split_docs,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME
        )
        logger.info(f"Indexed {len(split_docs)} chunks from PDF.")
        return True
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        return False

# QA function
def qa_function(query):
    vector_store = st.session_state.get("vector_store")
    if not vector_store:
        return "No documents available. Please upload a PDF first."
    
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant documents found for this question."
        chain = load_qa_chain(llm=llm_chat, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        return response
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
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
