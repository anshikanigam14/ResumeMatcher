import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

ranking_explainer_prompt = """
You are an AI assistant helping recruiters understand why certain resumes were selected and ranked highly for a job requirement.

Input : {rephrased_requirement}, {top_5_selected_resumes}

Input Description :
1. rephrased_requirement -  A detailed Job Description (JD) outlining the expectations for the role, required qualifications, technical and soft skills, years of experience, domain knowledge, tools, platforms, and any additional requirements.
2. top_5_selected_resumes - A list of 5 resumes selected based on similarity to the JD. Each resume is a dictionary containing :
- source : Resume filename
- ParsedText : The raw text extracted from the resume
- Rank : The relevancy rank of the corresponding resume

Task Description :
You are an AI assistant helping recruiters understand why certain resumes were selected and ranked highly for a job requirement.

Detailed Instruction :
Given the above inputs, generate a Markdown-formatted explanation that includes:
1. **Job Requirement**: Display the job description (JD) or `rephrased_requirement` as a heading at the top.
2. **Ranked Candidate Explanations**:
   - For each candidate (sorted by `Rank`, lowest to highest):
     - Display their `Rank` (1 to 5) and `source` (filename).
     - Include a bullet list of **Key Matching Points** showing **what skills, tools, or experiences matched** the JD. Highlight *specific technologies, domains, years of experience, integration experience, certifications*, etc.
     - Provide a short, clear **Summary** that explains why the candidate is a good fit.
     - Add a line separator (`---`) after each candidate section for clarity.

Output Format :
- Use **Markdown** with proper headings (`#`, `##`, `###`, `---` for separators).
- Ensure concise and readable formatting.
- Avoid repetition across bullet points.
- Do **not** include extra commentary or explanations outside the Markdown structure.

## Example Output Structure:
```markdown
# Job Requirement
<Insert rephrased_requirement here>

---

## Rank 1 — Candidate 2.docx

**Key Matching Points:**
- 6+ years of SAP SD experience
- Experience in full-cycle implementation and rollout
- Strong integration with MM, FI, CPI, TM
- Expertise in OTC, pricing, delivery, and billing
- Proficient in LSMW, IRPA, and API integration
- SAP SD Certified

**Summary:**  
Highly experienced consultant matching core requirements with technical depth and relevant integrations.

---

## Rank 2 — Candidate 1.docx

**Key Matching Points:**
- 3 years of SAP SD experience
- Completed one end-to-end rollout project
- Experience with OTC cycle and pricing
- Familiarity with SAP SD-MM, SD-TM integration
- Worked on automation tools like IRPA and APIs
- Experience with LSMW and reporting

**Summary:**  
Solid match for a mid-level SAP SD role with proven hands-on rollout and tool experience.

---

(Repeat for Rank 3–5 if available)
```
"""



def llm_ranking_explainer(rephrased_requirement, top_5_selected_resumes):
    parser = StrOutputParser()
    explainer_prompt = PromptTemplate(
        input_variables=["rephrased_requirement", "top_5_selected_resumes"],
        template=ranking_explainer_prompt
    )

    explainer_chain = explainer_prompt | llm | parser
    explained_md = explainer_chain.invoke({
        "rephrased_requirement": rephrased_requirement,
        "top_5_selected_resumes": top_5_selected_resumes
    })
    
    return explained_md





explained_md = llm_ranking_explainer(rephrased_requirement, top_5_selected_resumes)
print(explained_md)