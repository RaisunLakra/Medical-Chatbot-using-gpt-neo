from transformers import pipeline
from sentence_transformers import SentenceTransformer
from utils.logger import setup_logger
from config import GENERATOR_MODEL_NAME, EMBEDDING_MODEL_NAME

logger = setup_logger(__name__)

def load_models():
    try:
        generator = pipeline("text-generation", model=GENERATOR_MODEL_NAME)
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Models loaded successfully.")
        return generator, embedder
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise