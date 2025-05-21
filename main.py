"""
Entry Point
"""

import warnings

from clients.qdrant import ingest_qdrant_db
from example_jd_text import jd_text_1
from graph import get_relevant_candidates
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

if __name__ == "__main__":
    folder_path = "/Users/toothless/practice/interview-assignments/gyansys/test_data" 
    
    # User or Recruiter Query
    query = "Give me most relevant resume for a candidate working as 'SAP SD CONSULTANT' for more than 2 years."
    
    # initialise qdrant db and ingest formatted resume
    print("Ingesting given corpus of resumes...")
    ingest_qdrant_db(folder_path)
    
    # Process Job Description and user query --> rephrase query --> retrieve results --> rank results --> explain results
    print("Running Graph...")
    result = get_relevant_candidates(jd_text_1, query)
    
    
    # Print result)
    print("\nFinal Recommendation with Justification :")
    print(result["ranked_resumes_explained"])
    print("\n")
    print("---")
    print("Evaluation Results of Recommendation :")
    print(result["evaluation_result"])