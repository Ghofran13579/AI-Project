"""
Resume Screener — 100% Local, No API Key Required
Uses sentence-transformers (semantic similarity) + spaCy (NLP keyword extraction)

Install dependencies:
    pip install sentence-transformers spacy pdfplumber python-docx scikit-learn
    python -m spacy download en_core_web_md

Usage:
    python resume_screener_local.py --resume path/to/resume.pdf --position "Data Scientist"
    python resume_screener_local.py --resume path/to/resume.docx --position "Software Engineer"
"""

import argparse
import sys
import re
from pathlib import Path

# ─────────────────────────────────────────────
# LAZY IMPORTS (with helpful error messages)
# ─────────────────────────────────────────────

def import_dependencies():
    missing = []
    try:
        import pdfplumber
    except ImportError:
        missing.append("pdfplumber")
    try:
        from docx import Document
    except ImportError:
        missing.append("python-docx")
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        missing.append("sentence-transformers scikit-learn")
    try:
        import spacy
    except ImportError:
        missing.append("spacy")

    if missing:
        print("\n[ERROR] Missing packages. Run this command first:\n")
        print(f"  pip install {' '.join(missing)}")
        if "spacy" in missing:
            print("  python -m spacy download en_core_web_md")
        sys.exit(1)


# ─────────────────────────────────────────────
# 1. FILE READERS
# ─────────────────────────────────────────────

def read_pdf(path: str) -> str:
    import pdfplumber
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text.append(content)
    if not text:
        raise ValueError("Could not extract text from PDF. It may be scanned/image-based.")
    return "\n".join(text)


def read_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        raise ValueError("Could not extract text from Word document.")
    return "\n".join(paragraphs)


def load_resume(path: str) -> str:
    path_lower = path.lower()
    if path_lower.endswith(".pdf"):
        print(f"[INFO] Reading PDF: {path}")
        return read_pdf(path)
    elif path_lower.endswith(".docx"):
        print(f"[INFO] Reading Word document: {path}")
        return read_docx(path)
    else:
        raise ValueError("Unsupported file. Please provide .pdf or .docx")


# ─────────────────────────────────────────────
# 2. SKILLS DATABASE (used for gap detection)
# ─────────────────────────────────────────────

SKILLS_DB = {
    "data scientist": [
        "python", "machine learning", "deep learning", "statistics", "sql",
        "data visualization", "pandas", "numpy", "scikit-learn", "tensorflow",
        "pytorch", "r", "tableau", "power bi", "big data", "nlp", "a/b testing",
        "feature engineering", "model deployment", "data wrangling"
    ],
    "software engineer": [
        "python", "java", "c++", "algorithms", "data structures", "git",
        "rest api", "microservices", "docker", "kubernetes", "agile",
        "unit testing", "sql", "cloud", "ci/cd", "object oriented programming"
    ],
    "machine learning engineer": [
        "python", "machine learning", "deep learning", "tensorflow", "pytorch",
        "mlops", "model deployment", "docker", "kubernetes", "feature engineering",
        "data pipelines", "gpu", "scikit-learn", "api", "cloud"
    ],
    "data analyst": [
        "sql", "excel", "python", "r", "tableau", "power bi", "data visualization",
        "statistics", "reporting", "dashboard", "data cleaning", "pivot tables",
        "business intelligence", "kpi", "google analytics"
    ],
    "ai engineer": [
        "python", "machine learning", "deep learning", "nlp", "computer vision",
        "tensorflow", "pytorch", "llm", "prompt engineering", "api integration",
        "model fine-tuning", "rag", "transformers", "langchain"
    ],
    "web developer": [
        "html", "css", "javascript", "react", "node.js", "rest api", "sql",
        "git", "responsive design", "typescript", "vue", "angular", "docker"
    ],
    "project manager": [
        "project planning", "agile", "scrum", "risk management", "stakeholder management",
        "budget", "communication", "leadership", "ms project", "jira", "pmp"
    ],
    "marketing manager": [
        "digital marketing", "seo", "social media", "content strategy",
        "google analytics", "campaign management", "branding", "crm",
        "email marketing", "market research", "copywriting"
    ],
    "nurse": [
        "patient care", "clinical skills", "medication administration", "emr",
        "vital signs", "wound care", "critical thinking", "communication",
        "teamwork", "medical terminology", "bls", "acls"
    ],
    "financial analyst": [
        "financial modeling", "excel", "valuation", "accounting", "sql",
        "bloomberg", "forecasting", "budget", "reporting", "python",
        "power bi", "tableau", "investment analysis"
    ],
}

def get_position_skills(position_title: str) -> list:
    """Return known skills for a position, or generic professional skills."""
    position_lower = position_title.lower()
    # Try exact match first
    if position_lower in SKILLS_DB:
        return SKILLS_DB[position_lower]
    # Try partial match
    for key in SKILLS_DB:
        if any(word in position_lower for word in key.split()):
            return SKILLS_DB[key]
    # Generic fallback
    return [
        "communication", "teamwork", "problem solving", "leadership",
        "time management", "analytical thinking", "microsoft office",
        "project management", "attention to detail", "adaptability"
    ]


# ─────────────────────────────────────────────
# 3. NLP ANALYSIS
# ─────────────────────────────────────────────

def extract_keywords(text: str) -> list:
    """Extract meaningful keywords from text using spaCy."""
    import spacy
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        print("[ERROR] spaCy model not found. Run: python -m spacy download en_core_web_md")
        sys.exit(1)

    doc = nlp(text[:10000])  # limit for speed

    keywords = set()

    # Named entities (skills, tools, organizations, etc.)
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT", "GPE", "PERSON", "WORK_OF_ART"):
            keywords.add(ent.text.lower().strip())

    # Noun phrases (key concepts)
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        if 2 <= len(phrase.split()) <= 4:
            keywords.add(phrase)

    # Single important nouns and proper nouns
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and len(token.text) > 2:
            keywords.add(token.lemma_.lower().strip())

    return list(keywords)


def compute_similarity(resume_text: str, position_title: str) -> float:
    """Compute semantic similarity between resume and position using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    print("[INFO] Loading NLP model (first run may take a moment)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Split resume into chunks for better coverage
    sentences = [s.strip() for s in re.split(r'[\n.•]', resume_text) if len(s.strip()) > 20]
    sentences = sentences[:60]  # limit for speed

    if not sentences:
        return 0.0

   # Encode position title and resume sentences
    position_embedding = model.encode([position_title], convert_to_numpy=True)
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)

    # Compute similarity of each sentence with position title
    similarities = cosine_similarity(position_embedding, sentence_embeddings)[0]

    # Score = average of top 10 most relevant sentences
    top_scores = sorted(similarities, reverse=True)[:10]
    score = float(np.mean(top_scores))

    return score


def find_strengths(resume_text: str, position_skills: list) -> list:
    """Find skills from position requirements that appear in the resume."""
    resume_lower = resume_text.lower()
    found = []
    for skill in position_skills:
        if skill.lower() in resume_lower:
            found.append(skill)
    return found


def find_gaps(resume_text: str, position_skills: list) -> list:
    """Find expected skills that are NOT in the resume."""
    resume_lower = resume_text.lower()
    missing = []
    for skill in position_skills:
        if skill.lower() not in resume_lower:
            missing.append(skill)
    return missing[:6]  # top 6 gaps max


# ─────────────────────────────────────────────
# 4. SCREEN RESUME
# ─────────────────────────────────────────────

def screen_resume(resume_text: str, position_title: str) -> dict:
    print(f"[INFO] Analyzing resume for position: {position_title}")

    # Semantic similarity score (0 to 1)
    raw_score = compute_similarity(resume_text, position_title)

    # Scale to 0-100 with calibration
    # raw cosine similarity of 0.3+ is already quite good
    fit_percentage = min(100, int(raw_score * 200))

    # Clamp between 5 and 95
    fit_percentage = max(5, min(95, fit_percentage))

    # Skills matching
    position_skills = get_position_skills(position_title)
    strengths_found = find_strengths(resume_text, position_skills)
    gaps_found = find_gaps(resume_text, position_skills)

    # Boost score based on skill matches
    skill_boost = min(20, len(strengths_found) * 2)
    fit_percentage = min(98, fit_percentage + skill_boost)

    fit = fit_percentage >= 60

    return {
        "fit": fit,
        "fit_percentage": fit_percentage,
        "strengths": strengths_found if strengths_found else ["General qualifications detected in resume"],
        "gaps": gaps_found if gaps_found else ["No major gaps detected"],
        "verdict": f"{'Strong' if fit_percentage >= 75 else 'Moderate' if fit_percentage >= 60 else 'Weak'} match for {position_title}",
        "recommendation": (
            f"This candidate {'is a good fit' if fit else 'may not be the best fit'} for the {position_title} role "
            f"with a {fit_percentage}% match score. "
            f"{'Consider moving forward with an interview.' if fit else 'Consider other candidates or assess transferable skills.'}"
        )
    }


# ─────────────────────────────────────────────
# 5. DISPLAY RESULTS
# ─────────────────────────────────────────────

def display_results(result: dict, position: str):
    fit = result["fit"]
    percentage = result["fit_percentage"]
    strengths = result["strengths"]
    gaps = result["gaps"]
    verdict = result["verdict"]
    recommendation = result["recommendation"]

    fit_label = "FIT" if fit else "NOT FIT"
    fit_icon = "YES" if fit else "NO"
    bar_filled = int(percentage / 5)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)

    print("\n" + "=" * 60)
    print("  RESUME SCREENING REPORT")
    print(f"  Position : {position}")
    print("=" * 60)
    print(f"\n  Result     :  [{fit_icon}]  {fit_label}")
    print(f"  Fit Score  :  {percentage}%  [{bar}]")
    print(f"\n  Verdict    :  {verdict}")

    print("\n  STRENGTHS (matching skills found in resume):")
    for i, s in enumerate(strengths, 1):
        print(f"    {i}. {s}")

    print("\n  GAPS (expected skills not found):")
    for i, g in enumerate(gaps, 1):
        print(f"    {i}. {g}")

    print(f"\n  RECOMMENDATION:")
    print(f"    {recommendation}")
    print("\n" + "=" * 60)


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    import_dependencies()

    parser = argparse.ArgumentParser(
        description="Screen a resume against a position title — 100% local, no API key needed."
    )
    parser.add_argument("--resume", "-r", required=True, help="Path to resume (.pdf or .docx)")
    parser.add_argument("--position", "-p", required=True, help='Position title e.g. "Data Scientist"')
    args = parser.parse_args()

    try:
        resume_text = load_resume(args.resume)
        result = screen_resume(resume_text, args.position)
        display_results(result, args.position)

    except FileNotFoundError:
        print(f"[ERROR] File not found: {args.resume}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
