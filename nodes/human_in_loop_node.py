# from langgraph.types import interrupt

# def human_in_loop(state):
#     print("---HUMAN IN THE LOOP---")
#     value = interrupt("Do you want to revise the rephrased job description? (y/n)")

#     if value == "y" :
#         interrupt_2_value = interrupt("Please input the revised job description.")
#         state["rephrased_jd"] = interrupt_2_value

#     return {"rephrased_jd": state["rephrased_jd"]}


def human_in_loop(state):
    pass