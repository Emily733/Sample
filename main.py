from fastapi import FastAPI, UploadFile, Form, HTTPException
import spacy
from spacy import displacy
import numpy as np
from PyPDF2 import PdfReader

# Initialize FastAPI app
app = FastAPI()

# Load the trained SpaCy models
trained_resume = spacy.load("assets/model_cv/model-best")
trained_job = spacy.load("assets/model_cv/model-last")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

# Function to extract unique entities of all labels
def get_unique_entities(doc):
    unique_entities = set()
    for ent in doc.ents:
        unique_entities.add(ent.text.lower())  # Use lowercase for consistency
    return list(unique_entities)

# Use these entities for cosine similarity, focusing on all entities for broader matching
def compute_entity_similarity(job_ents, resume_ents, nlp):
    if not job_ents or not resume_ents:
        return 0.0, [], []  # Return 0 similarity if no entities

    # Get vectors for all entities
    job_ent_vecs = [nlp(ent).vector for ent in job_ents]
    resume_ent_vecs = [nlp(ent).vector for ent in resume_ents]

    # Compute average vectors for all entities
    job_avg_vec = np.mean(job_ent_vecs, axis=0)
    resume_avg_vec = np.mean(resume_ent_vecs, axis=0)

    # Calculate cosine similarity
    if np.linalg.norm(job_avg_vec) == 0 or np.linalg.norm(resume_avg_vec) == 0:
        return 0.0, job_ents, resume_ents  # Handle zero vectors
    cos_sim_score = np.dot(resume_avg_vec, job_avg_vec) / (
        np.linalg.norm(resume_avg_vec) * np.linalg.norm(job_avg_vec)
    )

    return float(cos_sim_score), job_ents, resume_ents  # Convert numpy.float32 to native float

# API endpoint to calculate similarity
@app.post("/")
async def calculate_similarity(resume: UploadFile, job_description: str = Form(...)):
    # Validate file type
    if not resume.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Extract text from PDF
    resume_text = extract_text_from_pdf(resume.file)

    # Process with NER models
    resume_doc = trained_resume(resume_text)
    job_doc = trained_job(job_description)

    # Extract unique entities from job and resume (all entity types)
    job_entities = get_unique_entities(job_doc)
    resume_entities = get_unique_entities(resume_doc)

    # Compute cosine similarity for all entities
    similarity_score, unique_job_ents, unique_resume_ents = compute_entity_similarity(job_entities, resume_entities, trained_resume)

    # Print for debugging (optional)
    print("\nUnique Job Entities (All Types):", unique_job_ents)
    print("Unique Resume Entities (All Types):", unique_resume_ents)
    print(f"\nCosine Similarity between all entities: {similarity_score}")
    print(similarity_score)
    if similarity_score > 0.5:
        print("Semantic Match: High similarity detected for entities.")
    else:
        print("Semantic Mismatch: Low similarity between entities.")

    # Return JSON response with similarity score as a native float
    return {
        "similarity_score": similarity_score * 100  # Convert to percentage for clarity
    }
