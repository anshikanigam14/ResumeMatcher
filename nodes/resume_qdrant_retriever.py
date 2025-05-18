
from qdrant_client.models import FieldCondition, Filter, Range, MatchValue
from qdrant_client.http.models import MatchText, Match
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient 
from dotenv import load_dotenv
import os

# from qdrant_client import QdrantClient, models

# client = QdrantClient(url="http://localhost:6333")

# Load environment variables
load_dotenv()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL")
embedding_model = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
collection_name = "GYANSYS-ResumeMatcher"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

filter_condition = Filter(
    should=[
        FieldCondition(
            key="metadata.role",
            match=MatchText(text="sap sd consultant")
        ),
        FieldCondition(
            key="metadata.total_experience_in_years",
            range=Range(gte=2)
        )
    ]
)

doc_store = QdrantVectorStore(client=client, collection_name=collection_name,embedding=embedding_model)
results = doc_store.similarity_search(
    query="Looking for a candidate with over 2 years of experience as an SAP SD Consultant, ideally with 3-6 years of hands-on SAP SD experience, including at least one full-cycle implementation or rollout project. The candidate should have expertise in configuring and supporting SAP SD processes such as Order-to-Cash (OTC), Pricing, Billing, Shipping, and Delivery, and be skilled in integrating SAP SD with other modules like FI, MM, EWM, TM, and CPI. Experience with API/Webservices, CPI-based integrations, and data migration tools like LSMW is essential. The candidate should be proficient in developing functional specifications for custom development, including reports, interfaces, enhancements, and data migration. They should have strong problem-solving skills and experience with Vistex, Amazon, Mandix, or BTP projects is a plus. Certification in SAP SD or equivalent work experience is required. The role is based in Bangalore and requires immediate to 15 days notice period.",
    k=2,
    filter=filter_condition
)

print("RESULTS:\n")
for doc in results:
    print(doc.metadata.get("resume_id"))