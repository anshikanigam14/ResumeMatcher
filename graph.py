from nodes.rephrase_job_description_node import rephrase_job_description
from state import GraphState
from nodes.retrieve_resumes_node import retrieve_resumes
from langgraph.graph import StateGraph, END
    
    
def build_resume_matcher_graph():
    """Build the complete resume matching graph."""
    # Define the workflow graph
    workflow = StateGraph(GraphState)
    
    # Add nodes to the graph
    workflow.add_node("rephrase_job_description_node", rephrase_job_description)
    workflow.add_node("retrieve_resumes_node", retrieve_resumes)
    
    
    
    """
    
    workflow.add_node("rank_resumes", rank_resumes_node)
    workflow.add_node("explain_rankings", explain_rankings_node)
    workflow.add_node("create_final_output", create_final_output_node)
    
    # Define the edges
    workflow.add_edge("parse_job_description", "rephrase_query")
    workflow.add_edge("rephrase_query", "retrieve_resumes")
    workflow.add_edge("retrieve_resumes", "rank_resumes")
    workflow.add_edge("rank_resumes", "explain_rankings")
    workflow.add_edge("explain_rankings", "create_final_output")
    workflow.add_edge("create_final_output", END)
    
    """
    # Set the entrypoint
    workflow.set_entry_point("rephrase_job_description_node")
    
    # Compile the graph
    return workflow.compile()
    


def process_job_description(raw_jd_text: str, user_query: str = "", top_k: int = 5):
    """Process a job description and find matching resumes."""
    # Create the graph
    graph = build_resume_matcher_graph()
    
    # Initialize the state
    initial_state = {
        "raw_job_description": raw_jd_text,
        "user_query": user_query if user_query else "Find the top 5 best candidates for this Job Description.",
        "rephrased_jd": None,
        "retrieved_resumes": None,
        
        # to do
        "ranked_resumes": None,
        "recommendation": None,
        "final_output": None
    }
    
    # Execute the graph
    result = graph.invoke(initial_state)
    
    return result["final_output"]