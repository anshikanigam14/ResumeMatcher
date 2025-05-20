from typing import Optional, TypedDict, List, Dict, Any
from langchain.schema import Document


class GraphState(TypedDict):
    """The state of our graph."""
    raw_job_description: str
    user_query: str
    rephrased_jd: str
    
    rephrased_query: Optional[str]
    retrieved_resumes: Optional[List[Document]]
    ranked_resumes: Optional[List[Dict[str, Any]]]
    recommendation: Optional[str]
    final_output: Optional[str]