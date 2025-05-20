import os

import docx2txt
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

feature_extractor_prompt = """
        Task Description

        You are provided with a parsed resume in dictionary format as input: {parsed_resume}.

        Input Format

        The input dictionary contains the following keys:
            •	resume_id: A unique identifier for the resume.
            •	resume_text: The complete parsed text content of the resume.

        ⸻

        Objective

        Your task is to extract the following structured information from the resume_text and return the results in a clean JSON format:
            1.	resume_id
            2.	role
            3.	total_experience_in_years
            4.	overall_summary
            5.	skills
            6.	domains
            7.	associated_clients
            8.	certifications
            9.	experiences
            10.	education

        Use the precise definitions provided below for each feature. If a feature is not explicitly mentioned in the resume_text, return "NA" for that feature. Do not infer or hallucinate information.

        ⸻

        Feature Definitions
            1.	resume_id: The same unique resume ID provided in the input.
            2.	role: Extract the candidate’s primary job role or title (e.g., “SAP SD Consultant”, “Python Developer”) from the top section of the resume. If not clearly mentioned, return “NA”. Make sure the role is written entirely in lowercase letters.
            3.	total_experience_in_years: Total number of years of professional work experience. Use a numeric format such as 4, 4.5, 6, etc. This value would be in float.
            4.	overall_summary: A comprehensive generated summary (up to 5000 words) that captures all important information from the resume. Include all mentioned skills, total professional experience, domains, clients, and especially all key details from professional experience, such as roles, responsibilities, tools used, education, and achievements. Do not skip or generalize important content.
            5.	skills: A list of technical and non-technical skills explicitly mentioned in the resume.
            6.	domains: A list of industry or functional domains the candidate has worked in (e.g., finance, healthcare, retail, etc.).
            7.	associated_clients: A list of organization or client names the candidate has directly worked with, as mentioned in their resume.
            8.	certifications: A list of certifications (with names) mentioned in the resume.
            9.	experiences: A structured list of work experiences. Each item should be labeled as experience_summary_1, experience_summary_2, and so on. Include relevant details like job role, duration, skills used, and key responsibilities.
            10.	education: Extract details about academic background, including degree, institution name, and year (if available).

        ⸻

        Output Constraints
            •	Only return information that is explicitly stated in the resume text.
            •	Do not fabricate or assume any data.
            •	If a feature is missing, return "NA" for that field.
            •	Ensure the final output is in valid JSON format.
            
        ⸻
         
        Output Format Instructions:
            1. Do not include any unnecessary elements such as literal marks, code fences, extraneous characters, or irrelevant strings in the output.
            2. The output must strictly adhere to the following JSON format:
            ```
            {{
                "resume_id": "",
                "role": "",
                "total_experience_in_years": ,
                "overall_summary": "",
                "skills": [],
                "domains": [],
                "associated_clients" : [],
                "certifications": [],
                "experiences": [
                    {{
                    "experience_summary_1": ""
                    }},
                    {{
                    "experience_summary_2": ""
                    }},
                    {{
                    "experience_summary_3": ""
                    }},
                    ...
                ],
                "education": {{
                    "degree": "",
                    "college": "",
                    "graduation_year": ""
                }}
            }}
            ```
"""


def docx_parser(folder_path):
    all_resume_parsed = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            doc_text = docx2txt.process(file_path)
            all_resume_parsed.append({
                    "resume_id": file_path,
                    "resume_text": doc_text.strip()
                })
    return all_resume_parsed


def llm_feature_extractor(all_resume_parsed):
    all_formatted_resumes = []
    parser = JsonOutputParser()
    feature_prompt = PromptTemplate(
        input_variables=["parsed_resume"],
        template=feature_extractor_prompt,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    map_chain = feature_prompt | llm | parser
    
    for resume_dict in all_resume_parsed :
        llm_parsed_resume = map_chain.invoke(str(resume_dict))
        # Separate summary and metadata
        overall_summary = llm_parsed_resume["overall_summary"]
        metadata_keys = ["role", "resume_id", "total_experience_in_years"]
        metadata = {key: llm_parsed_resume[key] for key in metadata_keys if key in llm_parsed_resume}
        # Create LangChain Document
        formatted_resume = Document(
            page_content=overall_summary,
            metadata=metadata
        )
        all_formatted_resumes.append(formatted_resume)
    
    return all_formatted_resumes


def main(folder_path):
    # folder_path = "/Users/toothless/practice/interview-assignments/gyansys/test_data/"
    all_doc_texts_output = docx_parser(folder_path)
    all_formatted_resumes = llm_feature_extractor(all_doc_texts_output)
    # print(all_doc_texts_output[0])
    # print(all_formatted_resumes[0])
    return all_formatted_resumes