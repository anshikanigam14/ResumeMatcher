from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient  # Needed for collection ops
from dotenv import load_dotenv
from resume_parser import main
import os
from qdrant_client.models import Distance



# Load environment variables
load_dotenv()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL")
collection_name = "GYANSYS-ResumeMatcher"
embedding_model = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)


folder_path = "/Users/toothless/practice/interview-assignments/gyansys/test_data/"
all_parsed_documents = main(folder_path)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Check and delete collection if exists
collections = client.get_collections().collections
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


##########################################################################################

    
    

