from sentence_transformers import CrossEncoder
from docx import Document
from typing import List, Dict, Tuple

# --- CONFIG ---
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
WEIGHT_RAW = 0.6
WEIGHT_GENERATED = 0.4

# --- Load CrossEncoder Model ---
def load_model(model_name: str = MODEL_NAME):
    return CrossEncoder(model_name)

# --- Extract text from a .docx file ---
def extract_text_from_docx(docx_path: str) -> str:
    try:
        doc = Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading {docx_path}: {e}")
        return ""

# --- Compute similarity score ---
def compute_similarity_score(model, query: str, text: str) -> float:
    return float(model.predict([(query, text)]))

# --- Process a single resume ---
def process_resume(
    model,
    query: str,
    resume_key: str,
    generated_summary: str
) -> Dict:
    raw_text = extract_text_from_docx(resume_key) if resume_key.endswith(".docx") else ""
    raw_score = compute_similarity_score(model, query, raw_text) if raw_text else 0.0
    generated_score = compute_similarity_score(model, query, generated_summary)
    weighted_score = WEIGHT_RAW * raw_score + WEIGHT_GENERATED * generated_score

    return {
        "source": resume_key,
        "raw_score": round(raw_score, 3),
        "generated_score": round(generated_score, 3),
        "weighted_score": round(weighted_score, 3)
    }

# --- Main ranking function ---
def rank_resumes(
    model,
    query: str,
    resumes: List[Dict[str, str]],
    top_k: int = 5
) -> List[Dict]:
    results = []
    for resume_entry in resumes:
        for resume_key, generated_summary in resume_entry.items():
            result = process_resume(model, query, resume_key, generated_summary)
            results.append(result)
    
    sorted_results = sorted(results, key=lambda x: x["weighted_score"], reverse=True)
    return sorted_results[:top_k]

# --- Example Execution ---
if __name__ == "__main__":
    top_k_resumes = [
    {"/Users/toothless/practice/interview-assignments/gyansys/test_data/Candidate 1.docx" : "Certified SAP SD Consultant having 3 years of experience. Completed one end to end Rollout Project for India Specific Business and Support Project on client location. Good knowledge and understanding of organizational structure. Definition and assignment of Enterprise structure for SD module. Master Data- Customer Master Data, Customer material info record, Material Master Data, and Condition master data, Configuration in Order to Cash business process, Rush Order and Cash Sale, Return Process etc. Third party sales, Consignment Process. Knowledge of various determinations in Sales and Distribution module, Output Determination. Shipping point Determination, Item category, Pricing determination, Plant determination. SAP SD Module Expertise - Sales order processing, Pricing, Billing, and Delivery. Integration with Other SAP Modules. Master Data Management. Pricing and Billing Configuration. API/Webservices Integration with CPI. Credit Management scenario as per head office and branch related config. Output Determination. Reporting and Analytics – Create and Manage. User Exits and Enhancements. Data Migration – LSMW. Status Profiles. Availability Check (ATP). Third-Party Order Processing. SAP SD-FI Integration. SD-MM Integration. SD-TM Integration. SD-EWM Integration. Worked on-site in Mumbai, Delhi and Bangalore locations in support and consulting roles. Been a part of several new project initiatives as an SAP SD SPOC in SAP EWM, SAP TM, Distributor Management System and SAP BTP integration projects in gathering business requirements and creating business flow. SAP SD Configurations - Sales order processing, Pricing, Billing, and Delivery. Setup credit management configuration. Several Integration projects involving API, Webservices and CPI. Returns Automation. Third party sales. Developed multiple reports, Uploaders, Background jobs, Auto mailers & Interface Monitoring programs. Led SAP-IRPA Automation for SO & OBD. Led SAP-DMS Integration for SD OTC Cycle. Led SAP-Amazon Portal Integration for SO and Invoice posting. Led SAP-HMD Integration project for SD OTC Cycle. Involved in SAP-Vistex deals integration by making changes to existing pricing procedures. Setup Status profiles for multiple sales doc types. Documentation – Functional specifications and SOP documentation."},
    {"CANDIDATE 2 is an SAP SD Consultant with over 6 years of career experience, including 2.11 years of hands-on experience in SAP Support, Process-Implementation, SAP-Roll out, System Configuration, and Testing. The candidate has 3+ years of domain experience in the Manufacturing Industry. They possess good knowledge and understanding of organizational structure, definition and assignment of Enterprise structure for SD module, and Master Data Management. They have expertise in SAP SD Module, including Sales order processing, Pricing, Billing, and Delivery, and integration with other SAP Modules. The candidate has worked on-site in Mumbai locations in support and consulting roles, being a part of several new project initiatives as an SAP SD SPOC in SAP EWM, SAP TM, Distributor Management System, and SAP BTP integration projects. They have led various automation and integration projects, developed multiple reports, uploaders, background jobs, auto mailers, and interface monitoring programs. The candidate holds a B.E. in Automobile from ADCET, Ashta, Shivaji University Kolhapur, and a Diploma in Automobile from P.V.P.I.T. Budhgaon, MSBTE. They are also an SAP Certified Application Associate - Sales and Distribution with SAP ERP 6.0 EhP7."}
    ]

    rephrased_query = "Looking for a candidate with over 2 years of experience as an SAP SD Consultant, ideally with 3-6 years of hands-on SAP SD experience, including at least one full-cycle implementation or rollout project. The candidate should have expertise in configuring and supporting SAP SD processes such as Order-to-Cash (OTC), Pricing, Billing, Shipping, and Delivery, and be skilled in integrating SAP SD with other modules like FI, MM, EWM, TM, and CPI. Experience with API/Webservices, CPI-based integrations, and data migration tools like LSMW is essential. The candidate should be proficient in developing functional specifications for custom development, including reports, interfaces, enhancements, and data migration. They should have strong problem-solving skills and experience with Vistex, Amazon, Mandix, or BTP projects is a plus. Certification in SAP SD or equivalent work experience is required. The role is based in Bangalore and requires immediate to 15 days notice period."

    model = load_model()
    top_ranked = rank_resumes(model, rephrased_query, top_k_resumes)

    # Display results
    for i, resume in enumerate(top_ranked, 1):
        print(f"{i}. {resume['source']}")
        print(f"   Raw Score: {resume['raw_score']}")
        print(f"   Generated Score: {resume['generated_score']}")
        print(f"   Weighted Score: {resume['weighted_score']}")
        print()