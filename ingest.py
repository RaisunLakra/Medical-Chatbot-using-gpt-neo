# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import DirectoryLoader, PyPDFLoader
# from langchain.vectorstores import Qdrant
# from langchain.embeddings import SentenceTransformerEmbeddings

# embeddings = SentenceTransformerEmbeddings(model_name = 'NeuML/pubmedbert-base-embeddings')

# print(embeddings)

# loader = DirectoryLoader('Data/', glob='**/*.pdf', show_progress = True, loader_cls=PyPDFLoader)


# documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, line_length=130, break_on_whitespace=False, chunk_overlap=100)

# texts = text_splitter.split_documents(documents)

from dotenv import load_dotenv

load_dotenv()
