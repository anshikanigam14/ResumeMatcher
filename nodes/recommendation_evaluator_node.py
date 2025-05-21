from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from datasets import Dataset  
from state import GraphState
from ragas.llms import LangchainLLMWrapper
from clients.openai import get_llm


def recommendation_evaluator(state: GraphState):
    print(f"Running {__name__}...")
    llm = get_llm()
    evaluator_llm = LangchainLLMWrapper(llm)

    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    answer_relevancy = AnswerRelevancy(llm=evaluator_llm )

    ragas_eval_dataset = Dataset.from_dict({
        "question": [state["rephrased_jd"]],
        "contexts": [[r["raw_text"] for r in state["ranked_resumes"]]],
        "response": [state["ranked_resumes_explained"]]
    })

    result = evaluate(dataset=ragas_eval_dataset, metrics=[faithfulness_metric, answer_relevancy])
    return {"evaluation_result": str(result)}




