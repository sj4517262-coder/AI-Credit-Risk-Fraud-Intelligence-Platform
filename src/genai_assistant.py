import os
import sys
import google.generativeai as genai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.credit_model import predict_credit_risk
from src.fraud_model import predict_fraud
from src.explainability import explain_single

genai.configure(api_key=config.GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')


FEATURE_DESCRIPTIONS = {
    'cb_person_default_on_file': 'history of previous loan defaults',
    'cb_person_cred_hist_length': 'length of credit history in years',
}


def describe_feature(name):
    return FEATURE_DESCRIPTIONS.get(name, name.replace('_', ' '))


def build_context(prediction, shap_result, kind='credit'):
    risk_factors = [describe_feature(f['feature']) for f in shap_result['risk_factors']]
    safe_factors = [describe_feature(f['feature']) for f in shap_result['safe_factors']]

    if kind == 'credit':
        return (
            f"Loan application risk assessment.\n"
            f"Risk category: {prediction['risk_label']}\n"
            f"Risk score: {prediction['risk_score']} out of 1000\n"
            f"Factors increasing risk: {', '.join(risk_factors) if risk_factors else 'none'}\n"
            f"Factors reducing risk: {', '.join(safe_factors) if safe_factors else 'none'}\n"
            f"Recommended action: {prediction['action']}"
        )
    else:
        return (
            f"Transaction fraud assessment.\n"
            f"Status: {prediction['status']}\n"
            f"Fraud probability: {prediction['fraud_probability']} percent\n"
            f"Factors increasing suspicion: {', '.join(risk_factors) if risk_factors else 'none'}\n"
            f"Factors suggesting normal behavior: {', '.join(safe_factors) if safe_factors else 'none'}\n"
            f"Recommended action: {prediction['action']}"
        )


def generate_credit_explanation(input_data):
    prediction = predict_credit_risk(input_data)
    shap_result = explain_single(input_data, model_type='credit')
    context = build_context(prediction, shap_result, kind='credit')

    prompt = (
        "You are a risk analyst assistant at a bank. Based on the following "
        "model output, write a short 3 to 4 sentence explanation in plain English "
        "for a bank manager, explaining the decision and why it was made.\n\n"
        f"{context}"
    )

    response = model.generate_content(prompt)

    return {
        'explanation': response.text,
        'prediction': prediction,
        'shap_result': shap_result,
        'context': context
    }


def generate_fraud_explanation(input_data):
    prediction = predict_fraud(input_data)
    shap_result = explain_single(input_data, model_type='fraud')
    context = build_context(prediction, shap_result, kind='fraud')

    prompt = (
        "You are a fraud analyst assistant at a bank. Based on the following "
        "model output, write a short 3 to 4 sentence explanation in plain English "
        "for a fraud investigation team, explaining the decision and why it was made.\n\n"
        f"{context}"
    )

    response = model.generate_content(prompt)

    return {
        'explanation': response.text,
        'prediction': prediction,
        'shap_result': shap_result,
        'context': context
    }


def answer_question(question, explanation_data):
    prompt = (
        "You are a bank risk assistant. Here is the assessment data for a case:\n\n"
        f"{explanation_data['context']}\n\n"
        f"A bank employee asks: \"{question}\"\n\n"
        "Answer their question in 1 to 2 sentences using only the information above. "
        "Do not make up numbers that are not given."
    )

    response = model.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    sample = {
        'person_age': 22,
        'person_income': 9600,
        'person_home_ownership': 0,
        'person_emp_length': 0,
        'loan_intent': 0,
        'loan_grade': 3,
        'loan_amnt': 35000,
        'loan_int_rate': 16.02,
        'loan_percent_income': 0.59,
        'cb_person_default_on_file': 0,
        'cb_person_cred_hist_length': 3
    }

    result = generate_credit_explanation(sample)
    print(result['explanation'])

    print("\n--- asking follow up questions ---")
    print(answer_question("Why is this risky?", result))
    print(answer_question("What action should I take?", result))