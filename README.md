# Technify University — Predictive Analytics Dashboard

Single-page dashboard + Python serverless API, ready to deploy on Vercel.

## What's included
- `index.html` — the dashboard (static, served from project root)
- `api/dropout.py` — dropout prediction (Logistic Regression)
- `api/fee_default.py` — fee default prediction (Random Forest)
- `api/gpa.py` — GPA prediction (Random Forest, reuses the model your teammate built)
- `api/admission_forecast.py` — enrollment forecast (Prophet, precomputed)
- `models/` — the trained model files loaded by the API functions
- `requirements.txt` — Python deps for the serverless functions (versions pinned to match how the models were saved)

## Notes on the models
- **Dropout**: the submitted `dropout_model.pkl` was a 0-byte empty file, so it was retrained
  from the same notebook logic against `students.csv` (the only labeled dropout data available at
  the time). That first retrain used only 4 training rows, which caused the model to ignore GPA
  and fee status almost entirely and swing to 90%+ risk for decent students with moderate
  attendance/LMS activity. It has since been retrained again on 3,000 synthetic but realistically
  distributed rows where all four inputs (attendance, GPA, fee status, LMS activity) genuinely
  influence the result. It's still a synthetic-data model pending real historical dropout records,
  but its behavior across realistic inputs is now sensible and demo-safe.
- **Fee Default**: no fee-payment data exists yet in the university dataset, so (matching the
  original notebook's own fallback behavior) the target label is synthetic/random. Predictions
  are for demo purposes only until real fee-payment history is available.
- **GPA**: reuses the actual model + logic your teammate built (`GPA_prediction_module.zip`),
  unchanged.
- **Enrollment Forecast**: real Prophet forecast on `university_dataset.xlsx`, computed once and
  served as static JSON — this avoids bundling Prophet/cmdstan (large, slow cold starts) in a
  serverless function for a forecast that doesn't depend on per-request input anyway.
- **Student Risk** (Module 1): tries Abdullah &amp; Rashid's live Hugging Face API first. If that
  service doesn't respond with valid JSON (e.g. it's asleep, still building, or has crashed), the
  dashboard automatically retries once, and if it still fails, falls back to a small local
  RandomForestClassifier (`models/student_risk_fallback_model.pkl`) trained on the same four inputs
  (attendance, GPA, assignment completion, backlogs). Results from the fallback are clearly labeled
  in the UI so nobody mistakes them for the real deployed service.
- **Recommendation Engine** (Module 6): the submitted notebook's logic (read `ExamRecords`, flag
  any course below 50%) is now a real endpoint — `api/recommend/{student_id}` — running against
  the actual `ExamRecords` sheet from `university_dataset.xlsx` (bundled in `models/`). Try
  `STU0064` or `STU0675` for a student with weak courses, or any other `STU####` ID from the
  dataset.

## Deploy to Vercel

Option A — Vercel CLI (fastest):
```bash
npm i -g vercel
cd this-project-folder
vercel        # first deploy (follow prompts, link/create a project)
vercel --prod # promote to production
```

Option B — GitHub:
1. Push this folder to a GitHub repo.
2. Go to vercel.com → New Project → import the repo.
3. Leave build settings as default (no framework) → Deploy.

No environment variables or extra config needed — `vercel.json` and `requirements.txt` are
already set up. Vercel auto-detects the Python functions in `/api`.

## Local testing
```bash
npm i -g vercel
vercel dev
```
Then open http://localhost:3000
