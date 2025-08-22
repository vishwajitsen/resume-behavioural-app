import os
import nltk
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import fitz  # PyMuPDF

# ---------------- FastAPI Init ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Resume Analyzer API is running"}

# ---------------- NLTK Setup ----------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")

# ---------------- Trait Definitions ----------------
TRAIT_PROTOTYPES = {
    "Technical Expertise": ["Strong background in AI, machine learning, data science, and statistical modeling."],
    "Leadership": ["Proven ability to lead teams, mentor colleagues, and drive AI projects to success."],
    "Innovation": ["Creative thinker, able to design and build new AI products, solutions, and features."],
    "Problem Solving": ["Analytical and structured in approaching complex problems with data-driven methods."],
    "Communication": ["Clear communicator of complex technical ideas to non-technical stakeholders."],
    "Business Acumen": ["Understands business needs, product-market fit, and the value of AI for growth."],
    "Collaboration": ["Works effectively with cross-functional teams, engineers, product managers, and executives."],
    "Research Orientation": ["Keeps up with cutting-edge AI research and applies it in practical scenarios."],
    "Execution & Delivery": ["Track record of delivering AI solutions on time, at scale, with measurable impact."],
    "Strategic Vision": ["Ability to envision the future of AI and data products, aligning with company strategy."],
    # HR-Expanded Traits
    "Adaptability": ["Able to adjust quickly to changing priorities, technologies, and work environments."],
    "Emotional Intelligence": ["Demonstrates empathy, self-awareness, and effective interpersonal skills."],
    "Work Ethic": ["Proactive, dedicated, and committed to achieving professional excellence."],
    "Learning Agility": ["Quick learner, open to feedback, continuously developing new skills."],
    "Cultural Fit": ["Shares company values, collaborative spirit, and alignment with organizational culture."]
}

TRAIT_WEIGHTS = {
    "Technical Expertise": 0.12,
    "Leadership": 0.12,
    "Innovation": 0.08,
    "Problem Solving": 0.08,
    "Communication": 0.07,
    "Business Acumen": 0.07,
    "Collaboration": 0.07,
    "Research Orientation": 0.05,
    "Execution & Delivery": 0.07,
    "Strategic Vision": 0.07,
    "Adaptability": 0.05,
    "Emotional Intelligence": 0.05,
    "Work Ethic": 0.04,
    "Learning Agility": 0.04,
    "Cultural Fit": 0.02
}

# ---------------- Load NLP Model ----------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- Helper Functions ----------------
def extract_text_from_pdf_bytes(file_bytes: bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = " ".join([page.get_text("text") for page in doc])
    return text

def extract_candidate_name(text: str):
    lines = text.split("\n")
    if lines:
        first_line = lines[0].strip()
        words = first_line.split()
        return " ".join(words[:3]) if len(words) >= 3 else " ".join(words[:2])
    return "Candidate"

def hr_comment(score: float, trait: str):
    """Generate HR-style comments based on score ranges"""
    if score >= 90:
        return f"Outstanding strength in {trait}, a clear differentiator."
    elif score >= 85:
        return f"Strong capability in {trait}, reliable and consistent."
    elif score >= 80:
        return f"Moderate competency in {trait}, potential for growth."
    else:
        return f"Limited evidence of {trait}, may need development."

def compute_trait_scores(text: str):
    trait_scores = {}
    evidence = {}
    comments = {}
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return {}, {}, {}

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    for trait, prototypes in TRAIT_PROTOTYPES.items():
        prototype_embedding = model.encode(prototypes, convert_to_tensor=True)
        cos_scores = util.cos_sim(sentence_embeddings, prototype_embedding)
        max_idx = int(np.argmax(cos_scores))
        best_sentence = sentences[max_idx]
        similarity = float(cos_scores[max_idx])

        score = 75 + (similarity / 0.6) * 20
        score = min(max(score, 78), 98)

        trait_scores[trait] = round(score, 2)
        evidence[trait] = best_sentence
        comments[trait] = hr_comment(score, trait)
    return trait_scores, evidence, comments

def compute_weighted_score(trait_scores: dict):
    weighted_sum = sum(trait_scores[t] * TRAIT_WEIGHTS[t] for t in TRAIT_WEIGHTS if t in trait_scores)
    final_score = weighted_sum
    if final_score < 85:
        final_score += 5
    return round(min(final_score, 97), 2)

def generate_charts(trait_scores, candidate_name):
    # Bar Chart
    plt.figure(figsize=(8, 5))
    plt.barh(list(trait_scores.keys()), list(trait_scores.values()))
    plt.title(f"{candidate_name} - Trait Scores")
    plt.xlabel("Score")
    bar_path = f"{candidate_name}_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()

    # Radar Chart
    traits = list(trait_scores.keys())
    values = list(trait_scores.values())
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(traits), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, "o-", linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), traits)
    plt.title("Candidate Trait Radar")
    radar_path = f"{candidate_name}_radar.png"
    plt.savefig(radar_path)
    plt.close()

    return bar_path, radar_path

def generate_pdf_report(candidate_name, trait_scores, weighted_score, evidence, comments, output_path="resume_report.pdf"):
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>Resume Analysis Report</b>", styles["Title"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"<b>Candidate:</b> {candidate_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Overall Fit Score:</b> {weighted_score}", styles["Normal"]))
    story.append(Spacer(1, 20))

    # Trait Scores
    story.append(Paragraph("<b>Trait Analysis</b>", styles["Heading2"]))
    for trait, score in trait_scores.items():
        story.append(Paragraph(f"<b>{trait}:</b> {score}", styles["Normal"]))
        story.append(Paragraph(f"<i>Evidence:</i> {evidence[trait]}", styles["Normal"]))
        story.append(Paragraph(f"<i>HR Comment:</i> {comments[trait]}", styles["Normal"]))
        story.append(Spacer(1, 12))

    # Add charts
    bar_path, radar_path = generate_charts(trait_scores, candidate_name)
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Visual Insights</b>", styles["Heading2"]))
    story.append(Spacer(1, 12))
    story.append(Image(bar_path, width=400, height=250))
    story.append(Spacer(1, 20))
    story.append(Image(radar_path, width=400, height=400))

    # Save PDF
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    doc.build(story)
    return output_path

# ---------------- API Endpoints ----------------
@app.post("/extract_resume")
async def extract_resume(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()

        # Extract text
        resume_text = extract_text_from_pdf_bytes(file_bytes)
        preview = resume_text[:500] + "..." if len(resume_text) > 500 else resume_text

        # Candidate + Scores
        candidate_name = extract_candidate_name(resume_text)
        trait_scores, evidence, comments = compute_trait_scores(resume_text)
        weighted_score = compute_weighted_score(trait_scores)

        # Generate PDF report
        report_path = f"{candidate_name.replace(' ', '_')}_report.pdf"
        generate_pdf_report(candidate_name, trait_scores, weighted_score, evidence, comments, report_path)

        return {
            "filename": file.filename,
            "size": len(file_bytes),
            "candidate": candidate_name,
            "preview": preview,
            "trait_scores": trait_scores,
            "weighted_score": weighted_score,
            "evidence": evidence,
            "comments": comments,
            "report_url": f"/download_report/{candidate_name.replace(' ', '_')}_report.pdf"
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/download_report/{filename}")
async def download_report(filename: str):
    file_path = filename
    if not os.path.exists(file_path):
        return JSONResponse({"error": "Report not found"}, status_code=404)
    return FileResponse(path=file_path, filename=filename, media_type="application/pdf")
