from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE = Path(__file__).parent
app = FastAPI(title="Fake Job Posting Detector")
model = joblib.load(BASE / "model.pkl")

# Notebook pipeline has a 'preprocessor' step (text + categorical + numeric);
# a text-only pipeline (TF-IDF -> classifier) just takes the combined text.
USES_TABULAR = "preprocessor" in getattr(model, "named_steps", {})


class JobPosting(BaseModel):
    title: str = ""
    company_profile: str = ""
    description: str = ""
    requirements: str = ""
    employment_type: str = "Unknown"
    required_experience: str = "Unknown"
    required_education: str = "Unknown"
    telecommuting: int = 0
    has_company_logo: int = 0
    has_questions: int = 0


@app.get("/")
def home():
    return FileResponse(BASE / "index.html")


@app.post("/predict")
def predict(job: JobPosting):
    text = " ".join([job.title, job.company_profile, job.description, job.requirements]).strip()
    if USES_TABULAR:
        X = pd.DataFrame([{
            "text": text,
            "telecommuting": job.telecommuting,
            "has_company_logo": job.has_company_logo,
            "has_questions": job.has_questions,
            "employment_type": job.employment_type,
            "required_experience": job.required_experience,
            "required_education": job.required_education,
        }])
    else:
        X = [text]
    p = float(model.predict_proba(X)[0][1])
    return {"fraudulent": p >= 0.5, "fraud_probability": round(p, 4), "legit_probability": round(1 - p, 4)}