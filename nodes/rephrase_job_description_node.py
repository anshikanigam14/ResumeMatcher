from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import GraphState
from clients.openai import get_llm


query_rephraser_prompt = """
Input : 
1. user_query: {user_query}
2. raw_job_description: {raw_job_description}

Input Description :
1. user_query -  A search query entered by the recruiter specifying the desired candidate profile (e.g., skills, role, experience).
2. raw_job_description - A detailed Job Description (JD) outlining the expectations for the role, required qualifications, technical and soft skills, years of experience, domain knowledge, tools, platforms, and any additional requirements.


Objective :
Your task is to generate a single rephrased query that combines and preserves **all important information** from both the `user_query` and the `raw_job_description`. This rephrased query will be used to retrieve the most relevant candidate resumes from a semantic search engine.

Guidelines :
• Carefully extract and retain all critical details from both the recruiter's original query and the job description.
• Pay special attention to:
    – Job role or title
    – Required technical and soft skills
    – Years of experience
    – Domains, industries, or preferred clients
    – Tools, platforms, or certifications
    – Preferred qualifications
• Do not omit or generalize important qualifications or criteria.
• Write the final query in natural, concise language suitable for a semantic search.
• The final query must be under 5000 words.

Output Format :
Return only the rephrased query string.
"""


def rephrase_job_description(state: GraphState):
    print(f"Running {__name__}...")
    llm = get_llm()
    parser = StrOutputParser()
    rephraser_prompt = PromptTemplate(
        input_variables=["user_query", "raw_job_description"],
        template=query_rephraser_prompt
    )

    map_chain = rephraser_prompt | llm | parser
    rephrased_jd = map_chain.invoke({
        "user_query": state["user_query"],
        "raw_job_description": state["raw_job_description"]
    })
    
    return {"rephrased_jd": rephrased_jd}