# Technify University ERP - Predictive Analytics Dashboard

This is a predictive analytics system I built for the Technify University ERP project. It has six prediction modules, all connected to one dashboard, and it's fully deployed and working.

**Live link:** https://technify-five.vercel.app

## What it does

It's one dashboard page where you can run six different predictions:

1. **Student Risk Prediction** - tells you if a student is at risk of failing, based on attendance, GPA, assignments, and backlogs. Also gives early-warning alerts.
2. **Dropout Prediction** - predicts the chance a student drops out, based on attendance, GPA, fee status, and LMS activity.
3. **Fee Default Prediction** - predicts the chance a student defaults on fee payment, based on department, year, city, scholarship, and CGPA.
4. **GPA Prediction** - predicts next semester's GPA from attendance, assignments, quizzes, previous GPA, and exam results.
5. **Enrollment Forecasting** - forecasts how many students will enroll in the next 5 years, based on past enrollment data.
6. **Recommendation Engine** - looks at a student's exam records and tells you which courses they're weak in.

All six are real - they call actual trained models or read actual data, nothing is hardcoded for show.

## How it's built

The dashboard (`index.html`) is a single page with one section per module. It sends requests to a Python backend (`api/index.py`), which is one FastAPI app doing all the work - loading the models, running predictions, and sending back the results. Everything is deployed on Vercel.

The Student Risk module is a bit different from the rest. It first tries to call a live API that a teammate deployed separately on Hugging Face. If that API doesn't respond (it sleeps when idle, or it can go down), the dashboard automatically switches to a small backup model I built locally, so the demo doesn't break. When that happens, it clearly shows a note saying the result came from the backup model, not the live one.

**Tech used:**
- Frontend: plain HTML/CSS/JavaScript, no framework
- Backend: Python + FastAPI
- ML: scikit-learn (Logistic Regression, Random Forest)
- Forecasting: Prophet
- Data: Pandas, NumPy
- Hosting: Vercel

## Project files
project/
├── index.html                        # the dashboard
├── api/
│   ├── index.py                      # the whole backend, one FastAPI app
│   └── admission_forecast_data.json  # enrollment forecast, precomputed
├── models/
│   ├── dropout_model.pkl
│   ├── fee_default_model.pkl
│   ├── gpa_model.pkl
│   ├── student_risk_fallback_model.pkl
│   ├── admission_forecast.json
│   └── university_dataset.xlsx       # real student/exam data
├── requirements.txt
├── vercel.json
└── .gitignore

Everything the backend needs is already inside the repo - the models, the dataset, the forecast numbers. No database, no API keys, no setup needed beyond deploying it.

## API routes

| Method | Route | What it does |
|---|---|---|
| GET | `/` | loads the dashboard |
| POST | `/api/dropout` | dropout prediction |
| POST | `/api/gpa` | GPA prediction |
| GET | `/api/fee_default` | gets dropdown options for the fee default form |
| POST | `/api/fee_default` | fee default prediction |
| GET | `/api/admission_forecast` | enrollment numbers, historical + forecast |
| GET | `/api/recommend/{student_id}` | weak courses for a student ID |
| POST | `/api/student_risk_fallback` | backup student risk model |

Example, dropout prediction:
```bash
curl -X POST https://technify-five.vercel.app/api/dropout \
  -H "Content-Type: application/json" \
  -d '{"attendance": 88, "gpa": 3.5, "fees_paid": 1, "lms_activity": 45}'
```
Returns:
```json
{"dropout_prediction": 0, "dropout_probability": 0.0, "risk_level": "Low"}
```

Example, recommendation engine:
```bash
curl https://technify-five.vercel.app/api/recommend/STU0064
```
Returns:
```json
{
  "student_id": "STU0064",
  "status": "Needs Attention",
  "weak_areas": [
    {"course_id": "CRS078", "percentage": 48.9, "grade": "D"},
    {"course_id": "CRS043", "percentage": 37.5, "grade": "F"}
  ]
}
```

## Things I'm being upfront about

- **Dropout model** runs on synthetic data (no real dropout history exists yet), but it's built with realistic weighting so the results make sense across different inputs. Should be retrained once real data is available.
- **Fee Default model** also runs on a synthetic label for the same reason - no real fee-payment history exists yet. The whole pipeline is ready to retrain the moment real data comes in.
- **Enrollment Forecast** is precomputed instead of calculated live, since it doesn't depend on any input from the user, and it keeps a heavy dependency (Prophet) out of the live backend.
- **Student Risk fallback model** is only a backup, not meant to replace the real external model - it's there so the dashboard still works if that service is down.

None of this is hidden - the dashboard tells you directly when something is a live prediction versus a placeholder waiting on real data.

## Running it yourself

```bash
npm i -g vercel
git clone <this-repo>
cd <this-repo>
vercel dev
```
Then open `http://localhost:3000`.

## Deploying your own copy

```bash
npm i -g vercel
vercel
vercel --prod
```
No environment variables needed, no extra setup. Vercel picks up the Python backend automatically.

## Credits

Built, debugged, integrated, and deployed by **Asadullah Chandio**, Team Leader, Data Science Alpha Team.

The prediction modules started as early drafts from the Data Science Alpha Team members. I went through each one, fixed the issues, rebuilt what needed rebuilding, and combined everything into this one working system.
