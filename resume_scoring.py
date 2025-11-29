# resume_scoring.py

import io
import re
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load embedding model once
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def extract_text_from_file(uploaded_file):
    """
    Supports: PDF, TXT.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    # For .txt or unknown types – try decoding as text
    try:
        content = uploaded_file.read()
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from a PDF file uploaded via Streamlit.
    """
    pdf_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # reset pointer so file can be re-read if needed

    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text


def _clean_words(text: str):
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    return words


def extract_skill_set(text: str):
    return set(_clean_words(text))


def score_resume_against_jd(jd_text: str, resume_text: str) -> dict:
    """
    Uses SentenceTransformer embeddings to compute semantic similarity
    between job description and resume.
    """

    print("🧠 Using Hugging Face scoring")  # debug line

    if not jd_text.strip() or not resume_text.strip():
        return {
            "score": 0,
            "strengths": "Insufficient text in resume or job description.",
            "gaps": "Provide a full JD and a readable resume.",
            "summary": "Could not compute similarity because input text was empty.",
        }

    try:
        jd_vec = model.encode([jd_text], convert_to_numpy=True)
        resume_vec = model.encode([resume_text], convert_to_numpy=True)

        sim = cosine_similarity(jd_vec, resume_vec)[0][0]
        sim = float(max(0.0, min(1.0, sim)))
        score = int(round(sim * 100))

        jd_skills = extract_skill_set(jd_text)
        resume_skills = extract_skill_set(resume_text)

        matched = sorted(list(jd_skills.intersection(resume_skills)))
        missing = sorted(list(jd_skills - resume_skills))

        matched_display = matched[:8] if matched else []
        missing_display = missing[:8] if missing else []

        strengths_text = (
            "• " + "\n• ".join(matched_display)
            if matched_display
            else "No major overlapping keywords detected (may be formatting issue)."
        )

        gaps_text = (
            "• " + "\n• ".join(missing_display)
            if missing_display
            else "Most JD keywords are present in the resume."
        )

        summary = (
            f"The resume matches the job description with a semantic similarity score of {score}/100. "
            f"It shares about {len(matched)} overlapping keywords with the JD and is missing around "
            f"{len(missing)} JD keywords."
        )

        return {
            "score": score,
            "strengths": strengths_text,
            "gaps": gaps_text,
            "summary": summary,
        }

    except Exception as e:
        return {
            "score": 0,
            "strengths": "Error while computing embeddings.",
            "gaps": str(e),
            "summary": "The local AI model failed to compute a score.",
        }
