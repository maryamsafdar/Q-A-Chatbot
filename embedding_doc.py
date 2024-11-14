import os
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENVIRONMENT
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Initialize Pinecone
from pinecone import Pinecone, ServerlessSpec

# Set your API keys and environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_ENV"] = PINECONE_API_ENVIRONMENT  # or the region of your Pinecone environment

# Initialize Pinecone instance
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Check if index exists, and create it if it doesn't
index_name = "langchain-chatbot"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Assuming 1536 dimensions for OpenAI embeddings
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_API_ENVIRONMENT
        )
    )

# Load PDF document
pdf_loader = UnstructuredPDFLoader("E:\Web Developement\document-answer-langchain-pinecone-openai\PES Highlights 2021-22 New.pdf")
documents = pdf_loader.load()

# Split document into smaller chunks for embeddings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
docs = text_splitter.split_documents(documents)

# Embed documents with OpenAI embeddings
embedding = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.environ.get("OPENAI_API_KEY"))
doc_embeddings = [embedding.embed_text(doc.page_content) for doc in docs]

# Store embeddings in Pinecone index
pinecone_index = pc.Index(index_name)
for doc, vector in zip(docs, doc_embeddings):
    pinecone_index.upsert(
        items=[
            {
                "id": doc.metadata.get("id", "doc"),
                "values": vector,
                "metadata": doc.metadata
            }
        ]
    )

print("Documents embedded and stored in Pinecone successfully!")
