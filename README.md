# 🏦 AI Retirement Savings Advisor | AI + FinTech

## 📌 Project Overview
AI Retirement Savings Advisor is an intelligent full-stack web application designed to help Indian users plan for retirement through machine learning predictions, financial calculations, and personalized recommendations. It analyzes users' financial information to estimate retirement savings, evaluate retirement readiness, identify risk profiles, and suggest strategies for achieving their retirement goals.

## ✨ Key Features
- **Retirement Corpus Estimation:** Calculates the estimated retirement fund using SIP, compound interest, and future value formulas.
- **Retirement Readiness Score:** Uses Gradient Boosting Regression to estimate financial preparedness for retirement.
- **On-Track Prediction:** Uses a Random Forest Classifier to determine whether users are on track to meet their retirement goals.
- **Risk Profile Detection:** Classifies users into investment risk categories using machine learning.
- **Personalized Recommendations:** Provides rule-based suggestions based on users' financial situations.
- **Interactive Dashboard:** Visualizes retirement projections and financial insights using Chart.js.
- **Analysis History:** Stores previous analyses using an SQLite database.
- **REST API:** Provides backend endpoints for financial analysis and dashboard statistics.

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Backend:** Flask
- **Machine Learning:** Scikit-learn
- **Data Analysis:** Pandas, NumPy
- **Frontend:** HTML5, CSS3, JavaScript
- **Data Visualization:** Chart.js
- **Database:** SQLite
- **Machine Learning Algorithms:** Random Forest, Gradient Boosting

## 🧠 Machine Learning
The project includes three primary predictive models:
- Random Forest Classifier for retirement tracking.
- Random Forest Classifier for risk profile classification.
- Gradient Boosting Regressor for retirement readiness scoring.

The models use financial features such as age, income, expenses, current savings, investments, inflation, expected returns, dependents, and risk tolerance.

## 📊 Dataset
- **Records:** 2,000 synthetic financial records.
- **Features:** 23 columns.
- **Target Variables:** `readiness_score`, `on_track`, and `risk_profile`.
- **Data Processing:** Synthetic data generation and feature engineering.

## 🚀 Project Goals
- Make retirement planning more accessible through technology.
- Demonstrate the application of machine learning in personal finance.
- Provide interactive financial insights through a web-based dashboard.
- Help users understand potential savings gaps and retirement preparedness.

## 🔮 Future Enhancements
- Real-time stock and mutual fund data integration.
- User authentication and personalized accounts.
- Tax planning and tax-saving investment calculations.
- Integration with financial platforms.
- Mobile application development.

## ⚠️ Disclaimer
This project is intended for educational and informational purposes only. Its predictions and recommendations are based on synthetic data and should not be treated as professional financial advice.

---
Developed as an AI + FinTech project to explore machine learning, financial modeling, and full-stack web development.

#ArtificialIntelligence #MachineLearning #FinTech #Python #Flask #ScikitLearn #FullStackDevelopment
