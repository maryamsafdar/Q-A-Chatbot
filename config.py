from dotenv import load_dotenv
import os

load_dotenv()   

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENVIRONMENT: str = os.getenv("PINECONE_API_ENVIRONMENT")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME")





