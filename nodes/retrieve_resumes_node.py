from clients.qdrant import (QDRANT_API_KEY, QDRANT_URL, get_client, get_collection_name, get_embedding_model)
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import MatchText
from qdrant_client.models import FieldCondition, Filter, Range
from state import GraphState


def retrieve_resumes(state: GraphState):
    print(f"Running {__name__}...")
    client = get_client(QDRANT_URL, QDRANT_API_KEY)
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

    collection_name = get_collection_name()
    embedding_model = get_embedding_model()
    
    # get rephrased jd from state
    query = state["rephrased_jd"]
    
    doc_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embedding_model)
    retrieved_resumes = doc_store.similarity_search(query=query, k=3, filter=filter_condition)

    # print("RESULTS:\n")
    # for doc in results:
    #     print(doc.metadata.get("resume_id"))
        
    return {"retrieved_resumes" : retrieved_resumes}