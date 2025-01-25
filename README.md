# Medical-LLM-Chatbot-using-aaditya-Llama3-OpenBioLLM-70B

## Flow Explanation:

### Backend (Preprocessing and Knowledge Base Creation):

i. Extract PDF using pPDF library
ii. create chunks using langchain.text_splitter
iii. create embeddings using sentence-transformers
iv. create vector space using pincone or qdrant
v. create knowledge base

### Frontend (User Interaction) :

i. query
ii. query embeddings
iii. mix with knowledge base (perform similarity search in vector DB.)

Model role:
context and response
