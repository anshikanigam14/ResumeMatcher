"""
Entry Point
"""

from example_jd_text import jd_text_1
from clients.qdrant import ingest_qdrant_db

if __name__ == "__main__":
    folder_path = "/Users/toothless/practice/interview-assignments/gyansys/test_data" 
    
    # User or Recruiter Query
    query = "Give me most relevant resume for a candidate working as 'SAP SD CONSULTANT' for more than 2 years."
    
    # initialise qdrant db and ingest formatted resume
    ingest_qdrant_db(folder_path)
    
    

    

    
    # Process job description
    result = process_job_description(jd_text_1, query, top_k=3)
    
    # Print result
    print("\nFinal Recommendation:")
    print(result)