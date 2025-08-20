from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Step 1:Load raw pdf
DATA_PATH = "data/"
def load_pdf(data):
    loader = DirectoryLoader(data, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf(data = DATA_PATH)
print(f"Loaded {len(documents)} documents.")

# Step 2: Split documents into chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
print(f"Created {len(text_chunks)} text chunks.")

# Step 3: Create vector embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )
    return embedding_model
embedding_model= get_embedding_model()

# Step 4: Store embeddings in FAISS vector store
DB_FAISS_PATH = "vectorstores/db_faiss"
db = FAISS.from_documents(documents=text_chunks, embedding=embedding_model)
db.save_local(DB_FAISS_PATH)