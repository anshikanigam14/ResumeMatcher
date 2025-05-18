from langgraph.graph import StateGraph, START,  END
from typing import TypedDict

# defining state

class MyState(TypedDict):
    user_input : str
    graph_state : str
    
    
# build graph

builder = StateGraph(MyState)

# add nodes

builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)


# add edges

builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_nodes)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

graph = builder.compile()


node_1 = resume_parser
node_2 = 


# # Save the graph image to a file
# graph_image_bytes = graph.get_graph().draw_mermaid_png()

# with open("graph.png", "wb") as f:
#     f.write(graph_image_bytes)

# print("Graph saved to 'graph.png'")