from model_loader import load_models
from query_handler import RAGHandler
from utils.logger import setup_logger

logger = setup_logger("RAGChatApp")

def main():
    logger.info("Starting Medical RAG Chatbot...")
    generator, embedder = load_models()
    rag = RAGHandler(embedder, generator)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting chatbot.")
            break
        response = rag.generate_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()


# from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# from pinecone import Pinecone
# from dotenv import load_dotenv
# import os
# import logging

# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# model_name: str = "EleutherAI/gpt-neo-1.3B"
# embedding_model_name: str='sentence-transformers/all-MiniLM-L6-v2'
# generator = pipeline("text-generation", model=model_name)

# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)

# def load_model(model_name: str=model_name, embedding_model_name: str=embedding_model_name) -> (pipeline, SentenceTransformer):# type: ignore
#     """Load a model from the Huggingface Hub."""
#     try:
#         generator = pipeline("text-generation", model=model_name)
#         embedding_model = SentenceTransformer(embedding_model_name)
#         logging.info("Model successfully loaded")
#         return generator, embedding_model
#     except Exception as e:
#         logging.error(f"Error loading model: {e}")
#         print(e)
#         print("Model does not found on hugging face hub")
#         return None