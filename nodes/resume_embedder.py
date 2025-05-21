import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from resume_parser import main as parse_resumes  


load_dotenv()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL")

COLLECTION_NAME = "GYANSYS-ResumeMatcher"
FOLDER_PATH = "/Users/toothless/practice/interview-assignments/gyansys/test_data/"


def qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def delete_collection_if_exists(client, collection_name):
    collections = client.get_collections().collections
    if any(c.name == collection_name for c in collections):
        client.delete_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")
    else:
        print(f"Collection '{collection_name}' does not exist. Creating new collection.")


def upload_documents_to_qdrant(docs, client):
    embedding_model = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    QdrantVectorStore.from_documents(
        docs,
        embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        distance=Distance.COSINE
    )
    print(f"Documents uploaded to collection '{COLLECTION_NAME}' successfully.")


def run_resume_ingestion_pipeline():
    print("Parsing resumes...")
    documents = parse_resumes(FOLDER_PATH)
    
    print("Connecting to Qdrant...")
    client = qdrant_client()
    
    print("Managing collection...")
    delete_collection_if_exists(client, COLLECTION_NAME)
    
    print("Uploading documents...")
    upload_documents_to_qdrant(documents, client)
