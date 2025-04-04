import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "medical-chatbot")
NAMESPACE = os.getenv("NAMESPACE", "medical-knowledge")

GENERATOR_MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"