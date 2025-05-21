import docx2txt
from sentence_transformers import CrossEncoder
from state import GraphState

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
WEIGHT_RAW = 0.6
WEIGHT_GENERATED = 0.4


def load_model(model_name: str = MODEL_NAME):
    return CrossEncoder(model_name)


def extract_text_from_docx(docx_path):
    try:
        return docx2txt.process(docx_path)
    except Exception as e:
        print(f"Error reading {docx_path}: {e}")
        return ""


def compute_similarity_score(model, query, text):
    return float(model.predict([(query, text)]))


def process_resume(model, query, resume_key, generated_summary):
    
    raw_text = extract_text_from_docx(resume_key) if resume_key.endswith(".docx") else ""
    raw_score = compute_similarity_score(model, query, raw_text) if raw_text else 0.0
    generated_score = compute_similarity_score(model, query, generated_summary)
    weighted_score = WEIGHT_RAW * raw_score + WEIGHT_GENERATED * generated_score

    return {
        "source": resume_key,
        "raw_text": raw_text,
        "raw_score": round(raw_score, 3),
        "generated_score": round(generated_score, 3),
        "weighted_score": round(weighted_score, 3)
    }


def rank_resumes(model, query, resumes, k):
    results = []
    for resume_entry in resumes:
        resume_key = resume_entry.metadata["resume_id"]
        generated_summary = resume_entry.page_content
        result = process_resume(model, query, resume_key, generated_summary)
        results.append(result)
    
    sorted_results = sorted(results, key=lambda x: x["weighted_score"], reverse=True)

    # Assign rank
    for idx, result in enumerate(sorted_results, start=1):
        result["rank"] = idx

    return sorted_results[:k]


def get_ranked_resumes(state: GraphState):
    print(f"Running {__name__}...")
    k = 5
    rephrased_jd = state["rephrased_jd"]
    top_k_resumes = state["retrieved_resumes"]
    
    model = load_model()
    top_k_ranked_resume = rank_resumes(model, rephrased_jd, top_k_resumes, k)
    # print(top_ranked[0].keys())
    
    return {"ranked_resumes" : top_k_ranked_resume}