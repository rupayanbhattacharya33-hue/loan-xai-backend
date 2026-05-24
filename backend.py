
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')
 
app = Flask(__name__)
CORS(app)
 
# ─────────────────────────────────────────────
# Feature names
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    'annual_income',
    'loan_amount',
    'loan_term_months',
    'credit_score',
    'debt_to_income',
    'employment_years',
    'num_credit_lines',
    'num_derogatory_marks'
]
 
FEATURE_DISPLAY = {
    'annual_income':        'Annual Income',
    'loan_amount':          'Loan Amount',
    'loan_term_months':     'Loan Term',
    'credit_score':         'Credit Score',
    'debt_to_income':       'Debt-to-Income Ratio',
    'employment_years':     'Employment Years',
    'num_credit_lines':     'Credit Lines Open',
    'num_derogatory_marks': 'Derogatory Marks'
}
 
# ─────────────────────────────────────────────
# Synthetic training data generator
# ─────────────────────────────────────────────
def generate_training_data(n=2000):
    np.random.seed(42)
    income    = np.random.lognormal(10.8, 0.5, n)
    loan_amt  = np.random.lognormal(10.1, 0.6, n)
    loan_term = np.random.choice([12, 24, 36, 48, 60], n)
    credit    = np.random.normal(680, 80, n).clip(300, 850)
    dti       = np.random.beta(2, 5, n) * 60
    emp_yrs   = np.random.exponential(5, n).clip(0, 40)
    cr_lines  = np.random.poisson(6, n).clip(0, 20)
    derog     = np.random.poisson(0.5, n).clip(0, 10)
 
    X = np.column_stack([
        income, loan_amt, loan_term, credit,
        dti, emp_yrs, cr_lines, derog
    ])
 
    # Approval scoring formula (mirrors real lending criteria)
    score = (
        (credit - 300) / 550 * 0.35 +
        (income / loan_amt).clip(0, 5) / 5 * 0.25 +
        (1 - dti / 60) * 0.20 +
        emp_yrs / 40 * 0.12 +
        (1 - derog / 10) * 0.08
    )
    noise = np.random.normal(0, 0.05, n)
    prob  = 1 / (1 + np.exp(-10 * (score + noise - 0.55)))
    y     = (prob > 0.5).astype(int)
    return X, y
 
# ─────────────────────────────────────────────
# Train model at startup
# ─────────────────────────────────────────────
print("Training model... (takes ~10 seconds)")
 
X_train, y_train = generate_training_data(2000)
 
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
 
model = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=4,
    random_state=42
)
model.fit(X_scaled, y_train)
print("Model trained!")
 
# ─────────────────────────────────────────────
# Build explainers
# ─────────────────────────────────────────────
shap_explainer = shap.TreeExplainer(model)
print("SHAP explainer ready!")
 
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=FEATURE_NAMES,
    class_names=['Denied', 'Approved'],
    mode='classification',
    random_state=42
)
print("LIME explainer ready! Server starting...")
 
# ─────────────────────────────────────────────
# /predict endpoint
# ─────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
 
    features = np.array([[
        data['annual_income'],
        data['loan_amount'],
        data['loan_term_months'],
        data['credit_score'],
        data['debt_to_income'],
        data['employment_years'],
        data['num_credit_lines'],
        data['num_derogatory_marks']
    ]])
 
    features_scaled = scaler.transform(features)
 
    proba = model.predict_proba(features_scaled)[0]
 
    # LIME
    def predict_fn(x):
        return model.predict_proba(scaler.transform(x))
 
    lime_result = lime_explainer.explain_instance(
        features[0],
        predict_fn,
        num_features=8,
        num_samples=1000
    )
    lime_output = [
        {'feature': f, 'weight': float(w)}
        for f, w in lime_result.as_list()
    ]
 
    # SHAP
    shap_values = shap_explainer.shap_values(features_scaled)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]
 
    shap_output = sorted([
        {
            'feature': FEATURE_NAMES[i],
            'display': FEATURE_DISPLAY[FEATURE_NAMES[i]],
            'value':   float(sv[i])
        }
        for i in range(8)
    ], key=lambda x: abs(x['value']), reverse=True)
 
    ev       = shap_explainer.expected_value
    base_val = float(
        ev[1] if hasattr(ev, '__len__') and len(ev) > 1
        else ev[0] if hasattr(ev, '__len__')
        else ev
    )
 
    return jsonify({
        'prediction':  int(proba[1] >= 0.5),
        'probability': float(proba[1]),
        'lime':        lime_output,
        'shap':        shap_output,
        'base_value':  base_val
    })
 
 
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})
 
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)