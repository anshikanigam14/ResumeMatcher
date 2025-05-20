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

query_rephraser_prompt = """
Input : {original_query}, {input_jd}

Input Description :
1. original_query -  A search query entered by the recruiter specifying the desired candidate profile (e.g., skills, role, experience).
2. input_jd - A detailed Job Description (JD) outlining the expectations for the role, required qualifications, technical and soft skills, years of experience, domain knowledge, tools, platforms, and any additional requirements.


Objective :
Your task is to generate a single rephrased query that combines and preserves **all important information** from both the `original_query` and the `input_jd`. This rephrased query will be used to retrieve the most relevant candidate resumes from a semantic search engine.

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



def llm_query_rephraser(original_query, input_jd):
    parser = StrOutputParser()
    rephraser_prompt = PromptTemplate(
        input_variables=["original_query", "input_jd"],
        template=query_rephraser_prompt
    )

    map_chain = rephraser_prompt | llm | parser
    rephrased_query = map_chain.invoke({
        "original_query": original_query,
        "input_jd": input_jd
    })
    
    return rephrased_query


original_query = """
Give me most relevant resume for a candidate working as 'SAP SD CONSULTANT' for more than 2 years.
"""
input_jd = """
About the job

Role: SAP SD Consultant
Location: Bangalore (Onsite)
Experience: 3–6 years
Notice Period: Immediate - 15 days

⸻

Key Responsibilities:
	•	Configure and support SAP SD processes including Order-to-Cash (OTC), Pricing, Billing, Shipping, and Delivery.
	•	Handle integration with other SAP modules such as FI, MM, EWM, TM, and CPI.
	•	Develop and support functional specifications for custom development including reports, interfaces, enhancements, and data migration (LSMW).
	•	Work closely with business users to gather and document requirements and translate them into SAP solutions.
	•	Implement and support third-party sales, returns automation, and credit management.
	•	Coordinate with technical teams for API/Webservices integration and CPI middleware projects.
	•	Manage status profiles, user exits, output determination, and ATP configurations.
	•	Lead small to mid-sized SAP SD rollout and automation projects.

⸻

Required Skills and Qualifications:
	•	3–6 years of hands-on SAP SD experience with at least one full-cycle implementation or rollout project.
	•	Strong understanding of organizational structures and enterprise structure assignments in SD.
	•	Solid knowledge of SD module integration with EWM, TM, FI, MM, and CPI.
	•	Experience with API/Webservices, CPI-based integrations.
	•	Working knowledge of data migration tools such as LSMW.
	•	Proficiency in business documentation including Functional Specifications and SOPs.
	•	Certification in SAP SD or equivalent work experience.
	•	Good problem-solving skills with exposure to Vistex, Amazon, Mandix, or BTP projects is a plus.

⸻

If you are interested, please share your updated resume to Venkatesan.Selvam@gyansys.com

⸻

About the company

GyanSys Inc.
IT Services and IT Consulting
1,001–5,000 employees | 1,809 on LinkedIn

Founded in 2005, GyanSys is a leading global system integrator company supporting enterprise customers worldwide. We specialize in solutions implementations, managed services, and data analytics spanning SAP, Salesforce, Microsoft, and other prime enterprise platforms. Using a mature blended delivery model with over 3,000 consultants, we support over 350 enterprise customers across the Americas, Europe, and APAC.
"""

rephrased_query = llm_query_rephraser(original_query, input_jd)
print(rephrased_query)