from clients.openai import get_llm
from guardrails import Guard
from guardrails.hub import ToxicLanguage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import GraphState

guard = Guard().use(
   ToxicLanguage(on_fail="fix")
)

ranking_explainer_prompt = """
You are an AI assistant helping recruiters understand why certain resumes were selected and ranked highly for a job requirement.

--------

Input : {rephrased_requirement}, {top_k_ranked_resumes}

--------

Input Description :
1. rephrased_requirement -  A detailed Job Description (JD) outlining the expectations for the role, required qualifications, technical and soft skills, years of experience, domain knowledge, tools, platforms, and any additional requirements.
2. top_k_ranked_resumes - A list of k resumes selected based on similarity to the JD. Each resume is a dictionary containing :
- source : Resume filename
- ParsedText : The raw text extracted from the resume
- Rank : The relevancy rank of the corresponding resume
- Weighted Score : A weighted score which defines the relevance ranking

--------

Task Description :
You are an AI assistant helping recruiters understand why certain resumes were selected and ranked highly for a job requirement. 

--------

Detailed Instruction :
Given the above inputs, generate a Markdown-formatted explanation that includes:
1. **Job Requirement**: Display the job description (JD) or `rephrased_requirement` as a heading at the top.
2. **Ranked Candidate Explanations**:
   - For each candidate (sorted by `Rank`, lowest to highest):
     - Display their `Rank` (1 to k) along with the full name of the candidate. The full name should be extracted from ParsedText, which contains the raw text extracted from the resume.
     - Display the source or filename.
     - Include a bullet list of **Key Matching Points** showing **what skills, tools, or experiences matched** the JD. Highlight *specific technologies, domains, years of experience, integration experience, certifications*, etc.
     - Add a line separator (---) after each candidate section for clarity.
4. **Comparative Justification** :
    At the end, add a section titled ###Why This Ranking? that briefly explains:
    Include a short explanation that:
        •	Validates or adjusts the ranks based on how well each candidate’s raw resume text aligns with the job description (`rephrased_requirement`).
        •	If the given rank is correct, state that clearly.
        •	If the rank appears incorrect based on your understanding, provide the corrected rank and explain why.

    Specifically:
        •	Explain why the top-ranked candidate (Rank 1) is the best fit by highlighting strong matches to the job requirements (e.g., experience range, specific tools, integrations, certifications, domain knowledge).
        •	Contrast with the lowest-ranked candidate (Rank k) by pointing out what key elements were missing, such as insufficient experience, missing technical skills, or lack of relevant project work.
        •	Keep the tone clear, professional, and evidence-based. Use facts from the raw resume and job description only.
--------

Output Format :
- Use **Markdown** with proper headings (`#`, `##`, `###`, `---` for separators).
- Ensure concise and readable formatting.
- Avoid repetition across bullet points.
- Do **not** include extra commentary or explanations outside the Markdown structure.

"""


def explain_rankings(state: GraphState):
    print(f"Running {__name__}...")
    llm = get_llm()
    rephrased_requirement = state["rephrased_jd"]
    top_k_ranked_resumes = state["ranked_resumes"]
    
    parser = StrOutputParser()
    explainer_prompt = PromptTemplate(
        input_variables=["rephrased_requirement", "top_k_ranked_resumes"],
        template=ranking_explainer_prompt
    )

    explainer_chain = explainer_prompt | llm | guard.to_runnable() | parser
    ranked_resumes_explained_md = explainer_chain.invoke({
        "rephrased_requirement": rephrased_requirement,
        "top_k_ranked_resumes": top_k_ranked_resumes
    })
    ranked_resumes_explained_clean = ranked_resumes_explained_md.strip().removeprefix("```markdown").removesuffix("```").strip()
    
    return {"ranked_resumes_explained" : ranked_resumes_explained_clean}