import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from resume_reranker import get_ranked_results

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

ranking_explainer_prompt = """
You are an AI assistant helping recruiters understand why certain resumes were selected and ranked highly for a job requirement.

Input : {rephrased_requirement}, {top_k_ranked_resumes}

Input Description :
1. rephrased_requirement -  A detailed Job Description (JD) outlining the expectations for the role, required qualifications, technical and soft skills, years of experience, domain knowledge, tools, platforms, and any additional requirements.
2. top_k_ranked_resumes - A list of k resumes selected based on similarity to the JD. Each resume is a dictionary containing :
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
     - Display their `Rank` (1 to k) along with the full name of the candidate. The full name should be extracted from ParsedText, which contains the raw text extracted from the resume.
     - Include a bullet list of **Key Matching Points** showing **what skills, tools, or experiences matched** the JD. Highlight *specific technologies, domains, years of experience, integration experience, certifications*, etc.
     - Add a line separator (---) after each candidate section for clarity.
3. **Comparative Justification** :
    At the end, add a section titled ###Why This Ranking? that briefly explains:
    Include a short explanation that:
        •	Validates or adjusts the ranks based on how well each candidate’s raw resume text aligns with the job description (`rephrased_requirement`).
        •	If the given rank is correct, state that clearly.
        •	If the rank appears incorrect based on your understanding, provide the corrected rank and explain why.

    Specifically:
        •	Explain why the top-ranked candidate (Rank 1) is the best fit by highlighting strong matches to the job requirements (e.g., experience range, specific tools, integrations, certifications, domain knowledge).
        •	Contrast with the lowest-ranked candidate (Rank k) by pointing out what key elements were missing, such as insufficient experience, missing technical skills, or lack of relevant project work.
        •	Keep the tone clear, professional, and evidence-based. Use facts from the raw resume and job description only.
z

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

## Rank 1 — <Name of the Candidate-1>

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

## Rank 2 — <Name of the Candidate-2>

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

(Repeat for Rank 1-k if available)
```
"""



def llm_ranking_explainer(rephrased_requirement, top_k_ranked_resumes):
    parser = StrOutputParser()
    explainer_prompt = PromptTemplate(
        input_variables=["rephrased_requirement", "top_k_ranked_resumes"],
        template=ranking_explainer_prompt
    )

    explainer_chain = explainer_prompt | llm | parser
    explained_md = explainer_chain.invoke({
        "rephrased_requirement": rephrased_requirement,
        "top_k_ranked_resumes": top_k_ranked_resumes
    })
    
    return explained_md




k = 5
top_k_resumes = [
{"/Users/toothless/practice/interview-assignments/gyansys/test_data/Candidate 1.docx" : "Certified SAP SD Consultant having 3 years of experience. Completed one end to end Rollout Project for India Specific Business and Support Project on client location. Good knowledge and understanding of organizational structure. Definition and assignment of Enterprise structure for SD module. Master Data- Customer Master Data, Customer material info record, Material Master Data, and Condition master data, Configuration in Order to Cash business process, Rush Order and Cash Sale, Return Process etc. Third party sales, Consignment Process. Knowledge of various determinations in Sales and Distribution module, Output Determination. Shipping point Determination, Item category, Pricing determination, Plant determination. SAP SD Module Expertise - Sales order processing, Pricing, Billing, and Delivery. Integration with Other SAP Modules. Master Data Management. Pricing and Billing Configuration. API/Webservices Integration with CPI. Credit Management scenario as per head office and branch related config. Output Determination. Reporting and Analytics – Create and Manage. User Exits and Enhancements. Data Migration – LSMW. Status Profiles. Availability Check (ATP). Third-Party Order Processing. SAP SD-FI Integration. SD-MM Integration. SD-TM Integration. SD-EWM Integration. Worked on-site in Mumbai, Delhi and Bangalore locations in support and consulting roles. Been a part of several new project initiatives as an SAP SD SPOC in SAP EWM, SAP TM, Distributor Management System and SAP BTP integration projects in gathering business requirements and creating business flow. SAP SD Configurations - Sales order processing, Pricing, Billing, and Delivery. Setup credit management configuration. Several Integration projects involving API, Webservices and CPI. Returns Automation. Third party sales. Developed multiple reports, Uploaders, Background jobs, Auto mailers & Interface Monitoring programs. Led SAP-IRPA Automation for SO & OBD. Led SAP-DMS Integration for SD OTC Cycle. Led SAP-Amazon Portal Integration for SO and Invoice posting. Led SAP-HMD Integration project for SD OTC Cycle. Involved in SAP-Vistex deals integration by making changes to existing pricing procedures. Setup Status profiles for multiple sales doc types. Documentation – Functional specifications and SOP documentation."},
{"/Users/toothless/practice/interview-assignments/gyansys/test_data/Candidate 2.docx" : "CANDIDATE 2 is an SAP SD Consultant with over 6 years of career experience, including 2.11 years of hands-on experience in SAP Support, Process-Implementation, SAP-Roll out, System Configuration, and Testing. The candidate has 3+ years of domain experience in the Manufacturing Industry. They possess good knowledge and understanding of organizational structure, definition and assignment of Enterprise structure for SD module, and Master Data Management. They have expertise in SAP SD Module, including Sales order processing, Pricing, Billing, and Delivery, and integration with other SAP Modules. The candidate has worked on-site in Mumbai locations in support and consulting roles, being a part of several new project initiatives as an SAP SD SPOC in SAP EWM, SAP TM, Distributor Management System, and SAP BTP integration projects. They have led various automation and integration projects, developed multiple reports, uploaders, background jobs, auto mailers, and interface monitoring programs. The candidate holds a B.E. in Automobile from ADCET, Ashta, Shivaji University Kolhapur, and a Diploma in Automobile from P.V.P.I.T. Budhgaon, MSBTE. They are also an SAP Certified Application Associate - Sales and Distribution with SAP ERP 6.0 EhP7."}
]
rephrased_requirement = "Looking for a candidate with over 2 years of experience as an SAP SD Consultant, ideally with 3-6 years of hands-on SAP SD experience, including at least one full-cycle implementation or rollout project. The candidate should have expertise in configuring and supporting SAP SD processes such as Order-to-Cash (OTC), Pricing, Billing, Shipping, and Delivery, and be skilled in integrating SAP SD with other modules like FI, MM, EWM, TM, and CPI. Experience with API/Webservices, CPI-based integrations, and data migration tools like LSMW is essential. The candidate should be proficient in developing functional specifications for custom development, including reports, interfaces, enhancements, and data migration. They should have strong problem-solving skills and experience with Vistex, Amazon, Mandix, or BTP projects is a plus. Certification in SAP SD or equivalent work experience is required. The role is based in Bangalore and requires immediate to 15 days notice period."

top_k_ranked_resumes = get_ranked_results(k, top_k_resumes, rephrased_requirement)
# print(top_ranked[0])

explained_md = llm_ranking_explainer(rephrased_requirement, top_k_ranked_resumes)
explained_md_clean = explained_md.strip().removeprefix("```markdown").removesuffix("```").strip()

print(explained_md_clean)