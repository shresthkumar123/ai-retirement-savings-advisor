# 🏦 AI Retirement Savings Advisor
### B.Tech Final Year / Mini Project | AI + FinTech Domain

---

## 📌 Project Overview
**AI Retirement Savings Advisor** is a full-stack intelligent web application that helps Indian users plan their retirement using machine learning predictions, financial calculations, and personalized AI recommendations — all in a clean, interactive dashboard.

---

## ✨ Key Features
| Feature | Technology |
|---|---|
| Retirement Corpus Estimation | Financial Math (FV, SIP, Compound Interest) |
| Retirement Readiness Score | Gradient Boosting Regressor |
| On-Track / At-Risk Classification | Random Forest Classifier |
| Risk Profile Detection | Random Forest Classifier |
| Personalized Recommendations | Rule-based AI Engine |
| Interactive Dashboard | Chart.js (Line, Bar, Doughnut) |
| Data Persistence | SQLite Database |
| REST API Backend | Python Flask |

---

## 🏗️ Project Structure
```
ai_retirement_advisor/
├── app.py                  # Flask backend + API routes
├── requirements.txt
├── data/
│   └── retirement_dataset.csv    # 2000-row synthetic dataset
├── ml/
│   ├── train_model.py            # ML pipeline (train + save)
│   └── models/                   # Saved .pkl model files
│       ├── scaler.pkl
│       ├── clf_track.pkl
│       ├── clf_risk.pkl
│       ├── reg_score.pkl
│       ├── risk_enc.pkl
│       ├── risk_label_enc.pkl
│       └── feature_names.pkl
├── templates/
│   ├── index.html                # Multi-step input form
│   ├── dashboard.html            # Results + charts
│   └── history.html              # Past analyses
└── instance/
    └── advisor.db                # SQLite database (auto-created)
```

---

## 🚀 Setup Instructions

### 1. Clone / Download the project
```bash
cd ai_retirement_advisor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train ML models (run once)
```bash
python ml/train_model.py
```

### 4. Run the Flask app
```bash
python app.py
```

### 5. Open browser
```
http://localhost:5000
```

---

## 🧠 ML Models Summary

| Model | Algorithm | Task | Performance |
|---|---|---|---|
| On-Track Classifier | Random Forest | Binary Classification | ~97% Accuracy |
| Risk Profile Classifier | Random Forest | Multi-class Classification | ~100% Accuracy |
| Readiness Score Regressor | Gradient Boosting | Regression | MAE ≈ 2.86, R² ≈ 0.937 |

### Features Used (19 total)
- Age, Retirement Age, Monthly Income/Expense
- Current Savings, Existing Investments
- Inflation Rate, Expected Return
- Dependents, Risk Tolerance
- Investment Instruments (EPF, PPF, NPS, MF, FD)
- Engineered: Savings Rate, Investment Coverage, Years to Retire

---

## 💹 Financial Formulas

### Future Value of Lump Sum
```
FV = PV × (1 + r)^n
```

### Future Value of SIP (Monthly)
```
FV = P × [((1 + r)^n - 1) / r]
```

### Corpus Required
```
Corpus = Monthly_Expense_Future × 12 × Years_in_Retirement / Return_Rate
```

### Monthly Saving Needed
```
PMT = Shortfall × r / ((1 + r)^n - 1)
```

---

## 📊 Dataset
- **Rows:** 2,000 synthetic records
- **Features:** 23 columns
- **Target Variables:** `readiness_score`, `on_track`, `risk_profile`
- **Generated using:** NumPy random sampling with financial logic
- **File:** `data/retirement_dataset.csv`

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Main input form |
| GET | `/dashboard` | Results dashboard |
| GET | `/history` | Past analyses |
| POST | `/api/analyze` | Run AI analysis |
| GET | `/api/stats` | Platform statistics |

---

## 👨‍💻 Tech Stack
- **Backend:** Python 3.10+, Flask 3.0
- **ML:** Scikit-learn, Pandas, NumPy
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Charts:** Chart.js 4.4
- **Database:** SQLite (via Flask-SQLite)
- **Fonts:** Sora, JetBrains Mono (Google Fonts)

---

## 🔮 Future Scope
1. Real-time stock/MF data via NSE/BSE API
2. User login with JWT authentication
3. SEBI-registered fund recommendations
4. Tax calculation module (80C, 80CCD)
5. Mobile app (React Native / Flutter)
6. Integration with Zerodha / Groww APIs

---

## 📄 License
MIT License — Free for academic use

---

*Built with ❤️ for Indian retirement planning | B.Tech Final Year Project*
