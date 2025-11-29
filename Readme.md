An AI-powered agent that automatically ranks multiple resumes against a given Job Description (JD), highlights strengths and gaps for each candidate, and provides a structured CSV export HR teams can use directly.

This project is built for the AI Agent Development Challenge as a Resume Screening Agent under the People & HR category.

---

OVERVIEW

Recruiters often receive hundreds of resumes for a single role and manually screening them is slow and inconsistent. This agent:

- Accepts a Job Description and multiple resumes
- Extracts text from each resume 
- Uses a semantic embedding model to measure how closely each resume matches the JD
- Computes a 0-100 composite fit score for each candidate
- Shows a ranked table with summaries, strengths and gaps
- Allows CSV download of the ranked list for easy sharing

The agent runs entirely on local/open-source models(Hugging Face)–no paid APIs or external LLM calls are required.

FEATURES

- Job Description Input – paste any JD with required skills, role and responsibilities  
- Multi-Resume Upload – upload several candidate resumes at once (PDF/TXT)  
- Robust Text Extraction
  - `pdfplumber` for regular text PDFs  
  - optional OCR fallback using `pdf2image + pytesseract` for scanned/image PDFs  
- Semantic matching with Hugging Face
  - Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings  
  - Computes semantic similarity between JD and resume  
- Weighted Scoring
  - 60% semantic similarity  
  - 25% JD vs resume skill keyword overlap  
  - 15% education / experience heuristic (years and degree)  
- Ranked Results Dashboard
  - Total candidates, best score, average score  
  - Ranked table with score, candidate, and summary  
  - Detailed per-candidate view with strengths & gaps  
  - Color-coded score badges (green / orange / red)  
- CSV Export
  - Download all ranked results as `resume_screening_results.csv`

---

TECH STACK

Language and Framework

- Python
- Streamlit (web UI)

AI & NLP

- `sentence-transformers` – `all-MiniLM-L6-v2` embedding model
- `scikit-learn` – cosine similarity
- Basic regex + heuristics for skills, experience, and degree detection

PDF & OCR

- `pdfplumber` – extract text from PDFs
- `pdf2image` – convert PDF pages to images (for OCR)
- `pytesseract` – Tesseract OCR engine
- `Pillow` – image handling

Other

- `pandas` – result tables & CSV export

> Note: OCR is optional. If Tesseract and Poppler are not installed on the system, the app still works for normal text-based PDFs.

---

ARCHITECTURE

High-Level flow

1. User Interaction (Streamlit UI - 'app.py')
   - User pastes the Job Description.
   - Uploads one or more resumes (PDF/TXT).
   - Clicks "Rank Resumes".

2. Text Extraction (`resume_scoring.py`)
   - For each uploaded file:
     - If it is a PDF:
       - Try to extract text using `pdfplumber`.
       - If extracted text is very short and OCR is available:
         - Convert PDF pages to images (`pdf2image`).
         - Run Tesseract OCR (`pytesseract`) to extract text.
     - If it is a TXT file:
       - Decode raw text.
   - Returns cleaned resume text.

3. AI Scoring Engine (`score_resume_against_jd`)
   - Embeds JD and resume text using `SentenceTransformer("all-MiniLM-L6-v2")`.
   - Computes semantic similarity using cosine similarity.
   - Extracts sets of keywords from JD and resume; calculates skill overlap ratio.
   - Heuristically extracts years of experience and checks for degrees in text.
   - Combines these signals into a weighted composite score (0–100).
   - Generates:
     - Strengths (semantic fit, skills, experience, degree)
     - Gaps (missing skills, low similarity, missing degree/experience)
     - Short summary.

4. Presentation Layer (Streamlit UI)
   - Ranks candidates by score (descending).
   - Displays:
     - Metrics (total candidates, best score, average score)
     - Ranked table (Score, Candidate, Summary)
     - Expandable per-candidate details (Strengths, Gaps).
   - Provides CSV download button.

DIAGRAM

See `assets/architecture_diagram.png` for a visual architecture diagram.

---

KEY FUTURE ENHANCEMENTS

 1. Advanced Skill Extraction (NER-based)
  Use NLP models to detect technical & soft skills more accurately instead of simple keyword matching.

 2️. Customizable Scoring Weights
  Allow recruiters to decide what matters more: skills, experience, education, domain, etc.

 3️. ATS Integration (Industry Ready)
  Export screened candidates directly to HR systems like Naukri ATS, Lever, or Google Sheets.

 4️. Bias & Fairness Detection
  Ensure screening remains unbiased toward gender, age, or region by anonymous evaluation options.

 5️. Login + Screening History
  Store and revisit multiple screening sessions, making the app useful for ongoing hiring pipelines.

SETUP & RUN INSTRUCTIONS

 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/AI-ResumeScreeningAgent.git
cd AI-ResumeScreeningAgent
