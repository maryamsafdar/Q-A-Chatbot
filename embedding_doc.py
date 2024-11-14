import os
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENVIRONMENT
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# Set API keys and environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_ENV"] = PINECONE_API_ENVIRONMENT

# Initialize Pinecone instance
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Define index name
index_name = "langchain-chatbot12"

# Check if the index exists, and create it if it doesn't
if index_name not in pc.list_indexes().names():
    print("Index does not exist. Creating index...")
    try:
        pc.create_index(
    name=index_name,
    dimension=1536,  # Update dimension to match embeddings
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region=PINECONE_API_ENVIRONMENT
    )
)

        print(f"Index '{index_name}' created successfully.")
    except Exception as e:
        print(f"Failed to create index '{index_name}': {e}")
else:
    print(f"Index '{index_name}' already exists.")

# Verify by listing indexes again
print("Current indexes in Pinecone:", pc.list_indexes().names())

# Load and embed document
pdf_loader = PyPDFLoader("PES Highlights 2021-22 New.pdf")
documents = pdf_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Embed documents
embedding = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.environ.get("OPENAI_API_KEY"))

# Embed documents using embed_documents instead of embed_text
doc_embeddings = embedding.embed_documents([doc.page_content for doc in docs])

# Store embeddings in Pinecone index
pinecone_index = pc.Index(index_name)
for doc, vector in zip(docs, doc_embeddings):
    pinecone_index.upsert(
    vectors=[
        {
            "id": doc.metadata.get("id", f"doc-{i}"),  # Add a unique ID for each document
            "values": vector,
            "metadata": doc.metadata
        }
        for i, (doc, vector) in enumerate(zip(docs, doc_embeddings))
    ]
)


print("Documents embedded and stored in Pinecone successfully!")
