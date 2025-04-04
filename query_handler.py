from pinecone import Pinecone, Index
from config import PINECONE_API_KEY, INDEX_NAME, NAMESPACE
from utils.logger import setup_logger

logger = setup_logger(__name__)

class RAGHandler:
    def __init__(self, embedder, generator):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index: Index = self.pc.Index(INDEX_NAME)
        self.embedder = embedder
        self.generator = generator

    def fetch_context(self, query: str, top_k=3):
        logger.info(f"Fetching context for query: {query}")
        try:
            embedding = self.embedder.encode(query, convert_to_numpy=True).tolist()
            result = self.index.query(
                namespace=NAMESPACE,
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )
            matches = result.get("matches", [])
            if not matches:
                logger.warning("No relevant matches found.")
                return ""
            # formatted_context = "\n".join(
            #     [f"- {match['metadata'].get('text', '')} (Score: {match.get('score', 0):.2f})" for match in matches]
            # )
            formatted_context = "\n".join(
                    [f"- {match['metadata']['text']} (Score: {match['score']:.2f})" for match in matches]
                ) if matches else "No relevant data found."
            logger.info("Context fetched successfully.")
            return formatted_context
        except Exception as e:
            logger.error(f"Error during Pinecone query: {e}")
            return ""

    def generate_response(self, query: str) -> str:
        logger.info(f"Generating response for query: {query}")

        context = self.fetch_context(query)
        if not context:
            return "Sorry, I couldn't find any relevant information to answer your question."

        prompt = f"""You are a responsible, safety-first AI assistant designed to provide general information on medical knowledge.

User Query: {query}

Relevant Information:
{context}

Now, provide a clear and informative response:
"""

        try:
            output = self.generator(
                prompt,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=50256,
                eos_token_id=None,
                return_full_text=False
            )
            response = output[0]["generated_text"]
            logger.info("Response generated successfully.")
            return response.strip()
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return "Sorry, I couldn't generate a response."
