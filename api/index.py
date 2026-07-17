import json
import os
import pickle

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
INDEX_HTML_PATH = os.path.join(BASE_DIR, "..", "index.html")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()

# ---------- Load models once, at cold start ----------
with open(os.path.join(MODELS_DIR, "dropout_model.pkl"), "rb") as f:
    dropout_model = pickle.load(f)

gpa_model = joblib.load(os.path.join(MODELS_DIR, "gpa_model.pkl"))

with open(os.path.join(MODELS_DIR, "fee_default_model.pkl"), "rb") as f:
    fee_bundle = pickle.load(f)
fee_model = fee_bundle["model"]
fee_scaler = fee_bundle["scaler"]
fee_encoders = fee_bundle["encoders"]
fee_feature_cols = fee_bundle["feature_cols"]

with open(os.path.join(BASE_DIR, "admission_forecast_data.json"), "r") as f:
    admission_forecast_data = json.load(f)

_exam_df = pd.read_excel(os.path.join(MODELS_DIR, "university_dataset.xlsx"), sheet_name="ExamRecords")
_exam_df.columns = _exam_df.columns.str.strip()
_exam_df["StudentID"] = _exam_df["StudentID"].astype(str).str.strip().str.upper()

with open(os.path.join(MODELS_DIR, "student_risk_fallback_model.pkl"), "rb") as f:
    student_risk_fallback_model = pickle.load(f)


def risk_level(proba_pct: float) -> str:
    if proba_pct < 30:
        return "Low"
    if proba_pct < 70:
        return "Medium"
    return "High"


def encode_fee(col: str, value: str):
    mapping = fee_encoders[col]
    if value in mapping:
        return mapping[value]
    return list(mapping.values())[0]


# ---------- Request schemas ----------
class DropoutRequest(BaseModel):
    attendance: float
    gpa: float
    fees_paid: int
    lms_activity: float


class GpaRequest(BaseModel):
    Attendance: float
    Assignments: float
    QuizScores: float
    PreviousGPA: float
    ExamResults: float


class FeeDefaultRequest(BaseModel):
    Gender: str
    Department: str
    YearLevel: str
    Status: str
    City: str
    Scholarship: str
    CGPA: float


class StudentRiskRequest(BaseModel):
    student_id: str = "UNKNOWN"
    attendance_percentage: float
    current_gpa: float
    assignments_submitted: float
    assignments_total: float
    backlogs: float


# ---------- Routes ----------
@app.post("/api/dropout")
def predict_dropout(body: DropoutRequest):
    try:
        row = pd.DataFrame({
            "attendance": [body.attendance],
            "gpa": [body.gpa],
            "fees_paid": [body.fees_paid],
            "lms_activity": [body.lms_activity],
        })
        pred = int(dropout_model.predict(row)[0])
        proba = float(dropout_model.predict_proba(row)[0][1]) * 100
        return {
            "dropout_prediction": pred,
            "dropout_probability": round(proba, 2),
            "risk_level": risk_level(proba),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/gpa")
def predict_gpa(body: GpaRequest):
    try:
        row = pd.DataFrame({
            "Attendance": [body.Attendance],
            "Assignments": [body.Assignments],
            "QuizScores": [body.QuizScores],
            "PreviousGPA": [body.PreviousGPA],
            "ExamResults": [body.ExamResults],
        })
        pred = float(gpa_model.predict(row)[0])
        pred = max(0.0, min(4.0, pred))
        return {"predicted_gpa": round(pred, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/fee_default")
def fee_default_options():
    options = {col: list(vals.keys()) for col, vals in fee_encoders.items()}
    return {"options": options}


@app.post("/api/fee_default")
def predict_fee_default(body: FeeDefaultRequest):
    try:
        row = pd.DataFrame({
            "Gender": [encode_fee("Gender", body.Gender)],
            "Department": [encode_fee("Department", body.Department)],
            "YearLevel": [encode_fee("YearLevel", body.YearLevel)],
            "Status": [encode_fee("Status", body.Status)],
            "City": [encode_fee("City", body.City)],
            "Scholarship": [encode_fee("Scholarship", body.Scholarship)],
            "CGPA": [body.CGPA],
        })[fee_feature_cols]
        row_scaled = fee_scaler.transform(row)
        pred = int(fee_model.predict(row_scaled)[0])
        proba = float(fee_model.predict_proba(row_scaled)[0][1]) * 100
        return {
            "default_prediction": "Default" if pred == 1 else "Paid",
            "default_probability": round(proba, 2),
            "risk_level": risk_level(proba),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/admission_forecast")
def get_admission_forecast():
    return admission_forecast_data


@app.get("/api/recommend/{student_id}")
def recommend(student_id: str):
    search_id = student_id.strip().upper()
    student_records = _exam_df[_exam_df["StudentID"] == search_id]

    if student_records.empty:
        raise HTTPException(status_code=404, detail=f"Student ID '{student_id}' not found.")

    low_performance = student_records[student_records["Percentage"] < 50]
    response = {
        "student_id": student_id,
        "status": "Good Standing" if low_performance.empty else "Needs Attention",
        "weak_areas": [],
    }
    for _, row in low_performance.iterrows():
        response["weak_areas"].append({
            "course_id": str(row.get("CourseID", "Unknown")),
            "percentage": float(row.get("Percentage", 0)),
            "grade": str(row.get("Grade", "N/A")),
        })
    return response


@app.post("/api/student_risk_fallback")
def predict_student_risk_fallback(body: StudentRiskRequest):
    try:
        total = body.assignments_total if body.assignments_total > 0 else 1
        completion_rate = min(1.0, body.assignments_submitted / total)

        row = pd.DataFrame({
            "attendance_percentage": [body.attendance_percentage],
            "current_gpa": [body.current_gpa],
            "completion_rate": [completion_rate],
            "backlogs": [body.backlogs],
        })
        proba = float(student_risk_fallback_model.predict_proba(row)[0][1])
        may_fail = bool(proba > 0.5)
        level = risk_level(proba * 100)

        alerts = []
        if may_fail:
            alerts.append({
                "type": "may_fail",
                "severity": "high" if proba > 0.7 else "medium",
                "message": f"Academic risk score is {round(proba, 2)}. Student may fail.",
            })
        if body.attendance_percentage < 75:
            alerts.append({
                "type": "low_attendance",
                "severity": "high" if body.attendance_percentage < 50 else "medium",
                "message": f"Attendance is {body.attendance_percentage}%, below the required 75%.",
            })

        return {
            "student_id": body.student_id,
            "risk_score": round(proba, 3),
            "risk_level": level.lower(),
            "may_fail": may_fail,
            "alerts": alerts,
            "source": "local_fallback",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
