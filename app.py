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
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

st.set_page_config(
    page_title="AI Q&A Assistant",  # Updated title
    page_icon="🤖",  # Updated icon
    layout="centered",  # Options: "centered" or "wide"
    initial_sidebar_state="auto"  # Options: "auto", "expanded", "collapsed"
)

# Load environment variables and set OpenAI API key
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
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
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
        print("Loading PDF document...")
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
llm_chat = ChatOpenAI(temperature=0.9, max_tokens=150, model='gpt-4o-mini', api_key=OPENAI_API_KEY)

def qa_function(query):
    qa_prompt_template = """
    You are an assistant for answering questions.

    If context is provided below, use it to answer the question concisely.

    If no context is provided, answer the question to the best of your ability using your general knowledge.

    Context:
    {context}

    Question: {input}
    """

    
    qa_prompt = PromptTemplate.from_template(qa_prompt_template)

    # Create a "stuff" documents chain within the function
    qa_chain = create_stuff_documents_chain(llm_chat, qa_prompt)

    # Set up a retriever chain within the function
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=qa_chain
    )

    # Use the retrieval chain to get the answer
    result = retrieval_chain.invoke({"input": query})

    # Return answer and relevant documents
    return result["answer"], result.get("context", [])


# Function to generate responses
def response_generator(query):
    try:
        docs = vector_store.similarity_search(query, k=2)
        chain = load_qa_chain(llm=llm_chat, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        return response, docs
    except Exception as e:
        logger.error(f"Error generating response for query '{query}': {e}")
        return None, None

# Function to display chat history
def display_chat():
    for message in st.session_state.history:
        role = message["role"]
        if role == "user":
            st.chat_message("user").markdown(message["content"])
        elif role == "assistant":
            st.chat_message("assistant").markdown(message["content"])

# Main Interface
st.markdown(
    """
    <style>
    .main-title {
        color: #4CAF50;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 18px;
        text-align: center;
        margin-bottom: 30px;
        color: #555555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🤖 AI Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your documents and get instant answers to your questions!</div>', unsafe_allow_html=True)

if st.sidebar.button("🔄 Reset Conversation"):
    st.session_state.history.clear()

# Display chat
display_chat()

# Chat Input Handler
user_query = st.chat_input("💬 Type your question here...")

if user_query:
    # Append the user's input to the history
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


