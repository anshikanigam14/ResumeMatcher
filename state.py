from typing import Optional, TypedDict, List, Dict, Any
from langchain.schema import Document


class GraphState(TypedDict):
    """The state of our graph."""
    raw_job_description: str
    user_query: str
    rephrased_jd: str
    user_feedback: str
    retrieved_resumes: Optional[List[Document]]
    ranked_resumes: Optional[List[Dict[str, Any]]]
    ranked_resumes_explained: Optional[str]
    evaluation_result : str