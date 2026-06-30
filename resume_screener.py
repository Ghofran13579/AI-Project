"""
Resume Screener — PDF & Word support
Uses Anthropic Claude API to evaluate any resume against any position title.

Requirements:
    pip install anthropic pdfplumber python-docx

Usage:
    python resume_screener.py --resume path/to/resume.pdf --position "Data Scientist"
    python resume_screener.py --resume path/to/resume.docx --position "AI Engineer"
"""

import argparse
import sys
import json
import anthropic
import pdfplumber
from docx import Document


# ─────────────────────────────────────────────
# 1. FILE READERS
# ─────────────────────────────────────────────

def read_pdf(path: str) -> str:
    """Extract all text from a PDF file."""
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text.append(content)
    if not text:
        raise ValueError("Could not extract any text from the PDF. It may be scanned/image-based.")
    return "\n".join(text)


def read_docx(path: str) -> str:
    """Extract all text from a Word (.docx) file."""
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        raise ValueError("Could not extract any text from the Word document.")
    return "\n".join(paragraphs)


def load_resume(path: str) -> str:
    """Detect file type and extract resume text."""
    path_lower = path.lower()
    if path_lower.endswith(".pdf"):
        print(f"[INFO] Reading PDF: {path}")
        return read_pdf(path)
    elif path_lower.endswith(".docx"):
        print(f"[INFO] Reading Word document: {path}")
        return read_docx(path)
    else:
        raise ValueError("Unsupported file type. Please provide a .pdf or .docx file.")


# ─────────────────────────────────────────────
# 2. AI SCREENING
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert HR recruiter and talent acquisition specialist with 20+ years of experience screening candidates across all industries and roles.

Your task is to evaluate a resume against a given position title and return a structured JSON response.

IMPORTANT: Return ONLY valid JSON — no markdown, no explanation outside the JSON.

JSON format:
{
  "fit": true or false,
  "fit_percentage": integer between 0 and 100,
  "verdict": "one sentence summary of the overall fit",
  "strengths": [
    "specific strength 1 from the resume that supports this role",
    "specific strength 2",
    "specific strength 3"
  ],
  "gaps": [
    "missing skill or experience 1 for this role",
    "missing skill or experience 2"
  ],
  "recommendation": "A 2-3 sentence recommendation for the hiring manager"
}

Rules:
- fit is true if fit_percentage >= 60
- Be specific — reference actual content from the resume
- Be objective and fair regardless of gender, nationality, or background
- If the resume is empty or unreadable, set fit_percentage to 0 and explain in verdict
"""


def screen_resume(resume_text: str, position_title: str) -> dict:
    """Send resume + position to Claude and return structured result."""
    client = anthropic.Anthropic()

    user_message = f"""
Position Title: {position_title}

Resume Content:
{resume_text}

Please evaluate this resume for the position above and return the JSON result.
"""

    print(f"[INFO] Sending to Claude for screening...")

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    raw = response.content[0].text.strip()

    # Clean up in case model wraps in markdown
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"Claude returned invalid JSON:\n{raw}")

    return result


# ─────────────────────────────────────────────
# 3. DISPLAY RESULTS
# ─────────────────────────────────────────────

def display_results(result: dict, position: str):
    """Print a clean, readable screening report."""
    fit = result.get("fit", False)
    percentage = result.get("fit_percentage", 0)
    verdict = result.get("verdict", "")
    strengths = result.get("strengths", [])
    gaps = result.get("gaps", [])
    recommendation = result.get("recommendation", "")

    fit_label = "✅  FIT" if fit else "❌  NOT FIT"
    bar_filled = int(percentage / 5)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)

    print("\n" + "=" * 60)
    print(f"  RESUME SCREENING REPORT")
    print(f"  Position: {position}")
    print("=" * 60)
    print(f"\n  Result     :  {fit_label}")
    print(f"  Fit Score  :  {percentage}%  [{bar}]")
    print(f"\n  Verdict    :  {verdict}")

    print("\n  STRENGTHS (what makes them a good fit):")
    for i, s in enumerate(strengths, 1):
        print(f"    {i}. {s}")

    if gaps:
        print("\n  GAPS (what is missing for this role):")
        for i, g in enumerate(gaps, 1):
            print(f"    {i}. {g}")
    else:
        print("\n  GAPS: None identified.")

    print(f"\n  RECOMMENDATION:")
    print(f"    {recommendation}")
    print("\n" + "=" * 60)


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Screen a resume (PDF or DOCX) against a position title using AI."
    )
    parser.add_argument(
        "--resume", "-r",
        required=True,
        help="Path to the resume file (.pdf or .docx)"
    )
    parser.add_argument(
        "--position", "-p",
        required=True,
        help='Position title to screen against, e.g. "Data Scientist"'
    )
    args = parser.parse_args()

    try:
        # Step 1: Load resume
        resume_text = load_resume(args.resume)

        # Step 2: Screen with AI
        result = screen_resume(resume_text, args.position)

        # Step 3: Display
        display_results(result, args.position)

    except FileNotFoundError:
        print(f"[ERROR] File not found: {args.resume}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except anthropic.AuthenticationError:
        print("[ERROR] Invalid Anthropic API key. Set it as environment variable: ANTHROPIC_API_KEY=sk-...")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
