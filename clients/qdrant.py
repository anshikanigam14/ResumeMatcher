"""
Creates a vector db if the collection does not exists and ingests formatted resume.
"""


from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient  # Needed for collection ops
from dotenv import load_dotenv
from clients.openai_resume_parser import get_formatted_resume
import os
from qdrant_client.models import Distance


# Load environment variables
load_dotenv()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL")

embedding_model = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

def get_client(url, api_key):
    client = QdrantClient(url=url, api_key=api_key)
    return client


def get_collection_name():
    return "GYANSYS-ResumeMatcher"


def get_embedding_model():
    OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL")
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)


def ingest_qdrant_db(folder_path):
    all_parsed_documents = get_formatted_resume(folder_path)

    # Check and delete collection if exists
    client = get_client(QDRANT_URL, QDRANT_API_KEY)
    embedding_model = get_embedding_model()
    collections = client.get_collections().collections
    collection_name = get_collection_name()
    
    if any(c.name == collection_name for c in collections):
        client.delete_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")
    else:
        print(f"Collection '{collection_name}' does not exist.")
        print("Creating the new collection.")


    doc_store = QdrantVectorStore.from_documents(
        all_parsed_documents,
        embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name,
        distance=Distance.COSINE
    )

    print(f"Documents uploaded to collection '{collection_name}' successfully.") 

