from nodes.rephrase_job_description_node import rephrase_job_description
from nodes.rank_resumes_node import get_ranked_resumes
from nodes.explain_rankings_node import explain_rankings
from state import GraphState
from nodes.retrieve_resumes_node import retrieve_resumes
from langgraph.graph import StateGraph, START, END
    
    
def build_resume_matcher_graph():
    """Build the complete resume matching graph."""
    # Define the workflow graph
    workflow = StateGraph(GraphState)
    
    # Add nodes to the graph
    workflow.add_node("rephrase_job_description_node", rephrase_job_description)
    workflow.add_node("retrieve_resumes_node", retrieve_resumes)
    workflow.add_node("rank_resumes_node", get_ranked_resumes)
    workflow.add_node("explain_rankings_node", explain_rankings)
    
    # Define the edges
    workflow.add_edge(START, "rephrase_job_description_node")
    workflow.add_edge("rephrase_job_description_node", "retrieve_resumes_node")
    workflow.add_edge("retrieve_resumes_node", "rank_resumes_node")
    workflow.add_edge("rank_resumes_node", "explain_rankings_node")
    workflow.add_edge("explain_rankings_node", END)
    
    
    # Set the entrypoint
    workflow.set_entry_point("rephrase_job_description_node")
    
    # Compile the graph
    return workflow.compile()
    

def get_relevant_candidates(raw_jd_text, user_query):
    """Process a job description and find matching resumes."""
    # Create the graph
    graph = build_resume_matcher_graph()
    
    # Initialize the state
    initial_state = {
        "raw_job_description": raw_jd_text,
        "user_query": user_query if user_query else "Find the top 5 best candidates for this Job Description.",
        "rephrased_jd": None,
        "retrieved_resumes": None,
        "ranked_resumes": None,
        "ranked_resumes_explained": None
    }
    
    # Execute the graph
    result = graph.invoke(initial_state)
    
    return result["final_output"]