"""
AI Retirement Savings Advisor — Flask Backend
"""
import os, pickle, sqlite3, json, math
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, g

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, 'ml', 'models')
DB_PATH   = os.path.join(BASE, 'instance', 'advisor.db')

app = Flask(__name__)
app.secret_key = 'retirement_ai_secret_2024'

# ── Load ML artifacts ──────────────────────────────────────────────────────────
def load_models():
    names = ['scaler','clf_track','clf_risk','reg_score',
             'risk_enc','risk_label_enc','feature_names']
    return {n: pickle.load(open(f'{MODEL_DIR}/{n}.pkl','rb')) for n in names}

MODELS = load_models()
FEATURES = MODELS['feature_names']

# ── SQLite helpers ─────────────────────────────────────────────────────────────
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db: db.close()

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, age INTEGER, created_at TEXT,
            input_data TEXT, result_data TEXT
        )
    """)
    db.commit(); db.close()

# ── Financial Calculation Engine ───────────────────────────────────────────────
def calculate_retirement(d):
    age            = int(d['age'])
    ret_age        = int(d['retirement_age'])
    monthly_income = float(d['monthly_income'])
    monthly_expense= float(d['monthly_expense'])
    current_savings= float(d['current_savings'])
    existing_inv   = float(d['existing_investments'])
    monthly_saving = monthly_income - monthly_expense
    inflation      = float(d['inflation_rate']) / 100
    ret_return     = float(d['expected_return']) / 100
    years_to_ret   = ret_age - age
    years_in_ret   = int(d.get('years_in_retirement', 25))

    r_m = ret_return / 12

    # Future value calculations
    fv_savings  = current_savings  * ((1 + r_m) ** (years_to_ret * 12))
    fv_sip      = max(monthly_saving, 0) * (((1 + r_m)**(years_to_ret*12) - 1) / r_m) if r_m else 0
    fv_existing = existing_inv * ((1 + r_m)**(years_to_ret * 12))
    projected_corpus = fv_savings + fv_sip + fv_existing

    # Corpus required
    future_monthly_exp = monthly_expense * ((1 + inflation) ** years_to_ret)
    corpus_needed = future_monthly_exp * 12 * years_in_ret / ret_return if ret_return else 0

    # Monthly saving required to hit target
    shortfall = max(corpus_needed - projected_corpus, 0)
    if r_m and years_to_ret > 0:
        months = years_to_ret * 12
        monthly_needed = shortfall * r_m / (((1 + r_m)**months) - 1) if shortfall > 0 else 0
    else:
        monthly_needed = 0

    savings_rate = monthly_saving / monthly_income if monthly_income else 0
    investment_coverage = (current_savings + existing_inv) / (monthly_expense * 12 * years_in_ret) if monthly_expense else 0

    return {
        'projected_corpus':     round(projected_corpus),
        'corpus_needed':        round(corpus_needed),
        'monthly_saving':       round(monthly_saving),
        'monthly_saving_needed':round(monthly_needed),
        'shortfall':            round(shortfall),
        'surplus':              round(max(projected_corpus - corpus_needed, 0)),
        'years_to_retire':      years_to_ret,
        'future_monthly_expense': round(future_monthly_exp),
        'savings_rate':         round(savings_rate * 100, 1),
        'investment_coverage':  round(investment_coverage, 3),
    }

def predict_ml(d, fin):
    risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
    risk_enc_val = risk_map.get(d.get('risk_tolerance', 'Medium'), 1)

    row = {
        'age':                  int(d['age']),
        'retirement_age':       int(d['retirement_age']),
        'monthly_income':       float(d['monthly_income']),
        'monthly_expense':      float(d['monthly_expense']),
        'monthly_saving':       fin['monthly_saving'],
        'current_savings':      float(d['current_savings']),
        'existing_investments': float(d['existing_investments']),
        'inflation_rate':       float(d['inflation_rate']) / 100,
        'expected_return':      float(d['expected_return']) / 100,
        'dependents':           int(d.get('dependents', 0)),
        'has_epf':              int(d.get('has_epf', 0)),
        'has_ppf':              int(d.get('has_ppf', 0)),
        'has_nps':              int(d.get('has_nps', 0)),
        'has_mutual_fund':      int(d.get('has_mutual_fund', 0)),
        'has_fd':               int(d.get('has_fd', 0)),
        'years_to_retire':      fin['years_to_retire'],
        'savings_rate':         fin['savings_rate'] / 100,
        'investment_coverage':  fin['investment_coverage'],
        'risk_tolerance_enc':   risk_enc_val,
    }
    X = np.array([[row[f] for f in FEATURES]])
    X_scaled = MODELS['scaler'].transform(X)

    on_track_pred  = int(MODELS['clf_track'].predict(X_scaled)[0])
    on_track_proba = float(MODELS['clf_track'].predict_proba(X_scaled)[0][1])
    risk_enc_pred  = MODELS['clf_risk'].predict(X_scaled)[0]
    risk_profile   = MODELS['risk_label_enc'].inverse_transform([risk_enc_pred])[0]
    readiness_score= float(MODELS['reg_score'].predict(X_scaled)[0])
    readiness_score= round(min(max(readiness_score, 0), 100), 1)

    return {
        'on_track':       on_track_pred,
        'on_track_proba': round(on_track_proba * 100, 1),
        'risk_profile':   risk_profile,
        'readiness_score':readiness_score,
    }

def generate_recommendations(d, fin, ml):
    recs = []
    score = ml['readiness_score']
    profile = ml['risk_profile']
    savings_rate = fin['savings_rate']
    shortfall = fin['shortfall']

    # Savings gap
    if fin['monthly_saving_needed'] > fin['monthly_saving']:
        gap = round(fin['monthly_saving_needed'] - fin['monthly_saving'])
        recs.append({
            'type': 'warning',
            'icon': '📈',
            'title': 'Increase Monthly SIP',
            'detail': f"Increase your monthly SIP by ₹{gap:,} to bridge the retirement gap."
        })
    else:
        recs.append({
            'type': 'success',
            'icon': '✅',
            'title': 'On Target — Keep Investing',
            'detail': "Your current savings rate is sufficient. Stay consistent!"
        })

    # Savings rate nudge

    
    if savings_rate < 20:
        recs.append({
            'type': 'warning',
            'icon': '💡',
            'title': 'Boost Your Savings Rate',
            'detail': f"Your savings rate is {savings_rate}%. Aim for at least 20–30% of income."
        })

    # EPF / PPF / NPS nudges
    if not int(d.get('has_epf', 0)):
        recs.append({'type':'info','icon':'🏛️','title':'Join EPF',
            'detail':'Enroll in Employee Provident Fund for tax-free, guaranteed 8.25% returns.'})
    if not int(d.get('has_ppf', 0)):
        recs.append({'type':'info','icon':'🏦','title':'Open a PPF Account',
            'detail':'PPF gives 7.1% tax-free returns with ₹1.5L/year deduction under 80C.'})
    if not int(d.get('has_nps', 0)):
        recs.append({'type':'info','icon':'📋','title':'Invest in NPS',
            'detail':'NPS gives an extra ₹50,000 tax deduction under 80CCD(1B) plus market returns.'})
    if not int(d.get('has_mutual_fund', 0)):
        recs.append({'type':'info','icon':'📊','title':'Start Mutual Fund SIP',
            'detail':'Equity mutual funds have historically delivered 12–15% CAGR over 10+ years.'})

    # Risk-based allocation
    if profile == 'Aggressive':
        recs.append({'type':'primary','icon':'🚀','title':'Equity-Heavy Allocation',
            'detail':'Consider 70% Equity (Large + Mid Cap), 20% Debt, 10% Gold for long-term growth.'})
    elif profile == 'Moderate':
        recs.append({'type':'primary','icon':'⚖️','title':'Balanced Portfolio',
            'detail':'Maintain 50% Equity, 30% Debt (Bonds/PPF), 20% Gold/REITs for stability.'})
    else:
        recs.append({'type':'primary','icon':'🛡️','title':'Capital Protection Focus',
            'detail':'Prefer 30% Equity, 50% Debt (FD/PPF/Bonds), 20% Gold for safety.'})

    # Emergency fund
    recs.append({'type':'info','icon':'🆘','title':'Maintain 6-Month Emergency Fund',
        'detail':f"Keep ₹{round(float(d['monthly_expense'])*6):,} liquid in a high-yield savings account or liquid fund."})

    # Term Insurance
    recs.append({'type':'primary','icon':'🔐','title':'Secure Term Insurance',
        'detail':f"Take a term plan of ₹{round(float(d['monthly_income'])*12*10/100000)*100000:,} to protect your family from financial risk."})

    # Score-based final message
    if score >= 80:
        recs.append({'type':'success','icon':'🏆','title':'Excellent Retirement Readiness!',
            'detail':'You are well-positioned. Review your portfolio annually and rebalance.'})
    elif score >= 50:
        recs.append({'type':'warning','icon':'⚠️','title':'Moderate Readiness — Action Needed',
            'detail':'Make incremental changes now; small steps compound greatly over time.'})
    else:
        recs.append({'type':'danger','icon':'🔴','title':'Urgent: Retirement Gap Detected',
            'detail':'Start a disciplined savings plan immediately. Consider delaying retirement by 2–3 years.'})

    return recs

def build_chart_data(d, fin):
    years = fin['years_to_retire']
    ret_return = float(d['expected_return']) / 100 / 12
    current_savings = float(d['current_savings'])
    existing_inv = float(d['existing_investments'])
    monthly_saving = max(fin['monthly_saving'], 0)

    corpus_growth, sip_growth, labels = [], [], []
    for y in range(0, years + 1, max(1, years // 10)):
        months = y * 12
        fv_s = current_savings * ((1 + ret_return) ** months)
        fv_sip = monthly_saving * (((1+ret_return)**months - 1)/ret_return) if ret_return else 0
        fv_e = existing_inv * ((1+ret_return)**months)
        corpus_growth.append(round((fv_s + fv_sip + fv_e) / 100000, 2))
        sip_growth.append(round((monthly_saving * months) / 100000, 2))
        labels.append(f"Yr {y}")

    # Allocation pie
    profile = 'Aggressive'
    rt = d.get('risk_tolerance','Medium')
    if rt == 'Low': equity, debt, gold = 30, 50, 20
    elif rt == 'High': equity, debt, gold = 70, 20, 10
    else: equity, debt, gold = 50, 35, 15

    return {
        'corpus_labels':   labels,
        'corpus_values':   corpus_growth,
        'sip_values':      sip_growth,
        'allocation':      {'equity': equity, 'debt': debt, 'gold': gold},
        'corpus_needed_L': round(fin['corpus_needed'] / 100000, 2),
        'corpus_proj_L':   round(fin['projected_corpus'] / 100000, 2),
    }

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/history')
def history():
    db   = get_db()
    rows = db.execute("SELECT id,name,age,created_at,result_data FROM analyses ORDER BY id DESC LIMIT 50").fetchall()
    items = []
    for r in rows:
        res = json.loads(r['result_data'])
        items.append({
            'id': r['id'], 'name': r['name'], 'age': r['age'],
            'created_at': r['created_at'],
            'readiness_score': res['ml']['readiness_score'],
            'risk_profile': res['ml']['risk_profile'],
            'on_track': res['ml']['on_track'],
        })
    return render_template('history.html', items=items)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        d = request.get_json()
        fin  = calculate_retirement(d)
        ml   = predict_ml(d, fin)
        recs = generate_recommendations(d, fin, ml)
        charts = build_chart_data(d, fin)

        result = {'financial': fin, 'ml': ml, 'recommendations': recs, 'charts': charts}

        db = get_db()
        db.execute(
            "INSERT INTO analyses (name,age,created_at,input_data,result_data) VALUES (?,?,?,?,?)",
            (d.get('name','Anonymous'), d['age'],
             datetime.now().strftime('%Y-%m-%d %H:%M'),
             json.dumps(d), json.dumps(result))
        )
        db.commit()
        return jsonify({'status':'ok', 'result': result})
    except Exception as e:
        return jsonify({'status':'error', 'message': str(e)}), 400

@app.route('/api/stats')
def stats():
    db = get_db()
    total = db.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    avg_score = db.execute("SELECT AVG(json_extract(result_data,'$.ml.readiness_score')) FROM analyses").fetchone()[0]
    on_track_count = db.execute(
        "SELECT COUNT(*) FROM analyses WHERE json_extract(result_data,'$.ml.on_track')=1"
    ).fetchone()[0]
    return jsonify({
        'total_analyses': total or 0,
        'avg_readiness':  round(avg_score or 0, 1),
        'on_track_pct':   round((on_track_count/(total or 1))*100, 1),
    })

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
