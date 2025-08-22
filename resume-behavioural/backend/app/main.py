# backend/app/main.py
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from analyzer import analyze_resume

app = FastAPI(title="Resume Behavioral Analysis API")

# allow requests from mobile app during dev; lock down in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your client URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
REPORT_DIR = "reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # basic validation
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    uid = uuid.uuid4().hex
    save_path = os.path.join(UPLOAD_DIR, f"{uid}.pdf")
    with open(save_path, "wb") as f:
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # limit to 10MB
            raise HTTPException(status_code=400, detail="File too large")
        f.write(content)

    # call analyzer
    try:
        result = analyze_resume(save_path, output_dir=REPORT_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # return JSON with link to download pdf
    pdf_filename = os.path.basename(result["pdf_path"])
    result["report_url"] = f"/reports/{pdf_filename}"
    return result

@app.get("/reports/{pdf_name}")
def get_report(pdf_name: str):
    path = os.path.join(REPORT_DIR, pdf_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, filename=pdf_name, media_type="application/pdf")
