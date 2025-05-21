from langgraph.graph import StateGraph, START, END
from nodes.rephrase_job_description_node import rephrase_job_description
from nodes.human_in_loop_node import human_in_loop
from nodes.rank_resumes_node import get_ranked_resumes
from nodes.explain_rankings_node import explain_rankings
from nodes.retrieve_resumes_node import retrieve_resumes
from nodes.recommendation_evaluator_node import recommendation_evaluator
from state import GraphState
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.types import Interrupt
    
def build_resume_matcher_graph():
    """Build the complete resume matching graph."""
    
    print(f"Running {__name__}...")
    # Define the workflow graph
    workflow = StateGraph(GraphState)
    
    # Add nodes to the graph
    workflow.add_node("rephrase_job_description_node", rephrase_job_description)
    workflow.add_node("human_in_loop_node", human_in_loop)
    workflow.add_node("retrieve_resumes_node", retrieve_resumes)
    workflow.add_node("rank_resumes_node", get_ranked_resumes)
    workflow.add_node("recommendation_evaluator_node", recommendation_evaluator)
    workflow.add_node("explain_rankings_node", explain_rankings)
    
    # Define the edges
    workflow.add_edge(START, "rephrase_job_description_node")
    workflow.add_edge("rephrase_job_description_node", "human_in_loop_node")
    workflow.add_edge("human_in_loop_node", "retrieve_resumes_node")
    workflow.add_edge("retrieve_resumes_node", "rank_resumes_node")
    workflow.add_edge("rank_resumes_node", "explain_rankings_node")
    workflow.add_edge("explain_rankings_node", "recommendation_evaluator_node")
    workflow.add_edge("recommendation_evaluator_node", END)
    

    # Set the entrypoint
    workflow.set_entry_point("rephrase_job_description_node")
    
    # checkpointer = MemorySaver()
    
    # # Compile the graph
    # return workflow.compile(checkpointer=checkpointer)
    
    return workflow.compile()
    

def get_relevant_candidates(raw_jd_text, user_query):
    """Process a job description and find matching resumes."""
    # thread_config = {"configurable": {"thread_id": "1"}}
    
    # Create the graph
    graph = build_resume_matcher_graph()
    
    # Initialize the state
    initial_state = {
        "raw_job_description": raw_jd_text,
        "user_query": user_query if user_query else "Find the top 5 best candidates for this Job Description.",
        "rephrased_jd": None,
        "user_feedback": None,
        "retrieved_resumes": None,
        "ranked_resumes": None,
        "ranked_resumes_explained": None,
        "evaluation_result": None
    }
    
        
    # # Execute the graph until human feedback
    # graph.invoke(initial_state, config=thread_config)
    # updated_state = graph.get_state(thread_config)
    # result = graph.invoke(updated_state, config=thread_config)
    
    # thread_id = 1
    # while True:
    #     interrupted = False

    #     for event in graph.stream(state, config={"configurable": {"thread_id": thread_id}}, stream_mode="updates"):
    #         print("EVENT:", event)

    #         if "__interrupt__" in event:
    #             prompt = event["__interrupt__"][0].value
    #             user_response = input(f"{prompt} ").strip()

    #             # Merge the new input into the existing state!
    #             if "revise" in prompt.lower() and "y/n" in prompt.lower():
    #                 state.update({"revise_response": user_response})
    #             else:
    #                 state.update({"rephrased_jd_input": user_response})

    #             interrupted = True
    #             break

    #     if not interrupted:
    #         break  
    
    result = graph.invoke(initial_state)
        
    return result