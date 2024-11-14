
import os
import streamlit as st
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENVIRONMENT,PINECONE_INDEX_NAME
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables and set OpenAI API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize Pinecone using the new Pinecone class
pinecone_client = Pinecone(
    api_key=PINECONE_API_KEY,
    spec=ServerlessSpec(
        cloud="aws",
        region=PINECONE_API_ENVIRONMENT
    )
)

# Check if the index exists and create it if it doesn't
if PINECONE_INDEX_NAME not in [index.name for index in pinecone_client.list_indexes()]:
    pinecone_client.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # Ensure this matches the embedding dimension of your model
        metric='cosine'
    )

# Connect to the index
index = pinecone_client.Index(PINECONE_INDEX_NAME)

# Setting up Streamlit
st.title('Document Answering with Langchain and Pinecone')
usr_input = st.text_input('What is your question?')

# Set OpenAI LLM and embeddings
llm_chat = ChatOpenAI(temperature=0.9, max_tokens=150, model='gpt-4o-mini')
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.environ.get("OPENAI_API_KEY"))

# Connect LangChain's Pinecone vector store to the existing index
docsearch = PineconeVectorStore(index=index, embedding=embeddings)

# Create the QA chain
chain = load_qa_chain(llm_chat)

# Check Streamlit input
if usr_input:
    # Generate LLM response
    try:
        search = docsearch.similarity_search(usr_input)
        response = chain.run(input_documents=search, question=usr_input)
        st.write(response)
    except Exception as e:
        st.write('It looks like you entered an invalid prompt. Please try again.')
        print(e)

    with st.expander('Document Similarity Search'):
        # Display search results
        search = docsearch.similarity_search(usr_input)
        st.write(search)
