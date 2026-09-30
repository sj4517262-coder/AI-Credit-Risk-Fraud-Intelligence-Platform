import os
import sys
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.credit_model import predict_credit_risk
from src.explainability import explain_single
from src.database import log_credit_prediction

st.set_page_config(page_title="Credit Risk", page_icon="💳", layout="wide")
st.title("💳 Credit Risk Assessment")

with st.form("credit_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Applicant Age", 18, 100, 30)
        income = st.number_input("Annual Income (₹)", 1000, 10000000, 500000)
        home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        emp_length = st.number_input("Years Employed", 0, 50, 5)

    with c2:
        intent = st.selectbox("Loan Purpose", [
            "EDUCATION", "MEDICAL", "VENTURE",
            "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"
        ])
        grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_amnt = st.number_input("Loan Amount (₹)", 500, 5000000, 200000)
        int_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 11.5)

    with c3:
        default_before = st.selectbox("Previous Default?", ["No", "Yes"])
        cred_hist = st.number_input("Credit History (years)", 0, 40, 5)

    submitted = st.form_submit_button("Assess Risk", type="primary")

if submitted:
    home_map = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
    intent_map = {"EDUCATION": 0, "MEDICAL": 1, "VENTURE": 2,
                  "PERSONAL": 3, "DEBTCONSOLIDATION": 4, "HOMEIMPROVEMENT": 5}
    grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}

    input_data = {
        'person_age': age,
        'person_income': income,
        'person_home_ownership': home_map[home],
        'person_emp_length': emp_length,
        'loan_intent': intent_map[intent],
        'loan_grade': grade_map[grade],
        'loan_amnt': loan_amnt,
        'loan_int_rate': int_rate,
        'loan_percent_income': round(loan_amnt / income, 2),
        'cb_person_default_on_file': 1 if default_before == "Yes" else 0,
        'cb_person_cred_hist_length': cred_hist
    }

    result = predict_credit_risk(input_data)
    shap_result = explain_single(input_data, model_type='credit')
    log_credit_prediction(input_data, result)

    st.divider()
    col1, col2 = st.columns([1, 2])

    with col1:
        score = result['risk_score']
        if score >= 700:
            bar_color = "#c62828"
        elif score >= 500:
            bar_color = "#ef6c00"
        elif score >= 300:
            bar_color = "#f9a825"
        else:
            bar_color = "#2e7d32"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Risk Score", 'font': {'size': 20}},
            number={'font': {'size': 44, 'color': bar_color}},
            gauge={
                'axis': {
                    'range': [0, 1000],
                    'tickmode': 'array',
                    'tickvals': [0, 250, 500, 750, 1000],
                    'tickcolor': "gray"
                },
                'bar': {'color': bar_color, 'thickness': 0.3},
                'steps': [
                    {'range': [0, 300], 'color': "#66bb6a"},
                    {'range': [300, 500], 'color': "#ffee58"},
                    {'range': [500, 700], 'color': "#ffa726"},
                    {'range': [700, 1000], 'color': "#ef5350"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'value': score
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        if result['risk_label'] in ["High Risk", "Very High Risk"]:
            st.error(f"**{result['risk_label']}**")
        else:
            st.success(f"**{result['risk_label']}**")
        st.write(f"**Recommended action:** {result['action']}")

        st.markdown("##### Applicant Summary")
        summary = {
            "Age": age,
            "Income": f"₹{income:,}",
            "Loan Amount": f"₹{loan_amnt:,}",
            "Loan/Income Ratio": f"{round(loan_amnt/income*100, 1)}%",
            "Interest Rate": f"{int_rate}%",
            "Model Used": result['model']
        }
        st.table({"Field": list(summary.keys()), "Value": list(summary.values())})

    with col2:
        st.markdown("#### Feature Impact (SHAP)")

        contribs = shap_result['contributions'][:7]
        features = [c['feature'].replace('_', ' ').title() for c in contribs]
        values = [c['shap_value'] for c in contribs]
        colors = ['#ef5350' if v > 0 else '#42a5f5' for v in values]

        fig2 = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{'+' if v > 0 else ''}{v}" for v in values],
            textposition='outside'
        ))
        fig2.update_layout(
            height=340,
            margin=dict(l=10, r=30, t=10, b=30),
            xaxis_title="← reduces risk   |   increases risk →",
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.session_state['last_credit_input'] = input_data
    st.session_state['last_credit_result'] = result
    st.session_state['last_credit_shap'] = shap_result
    st.success("Assessment saved. Go to AI Assistant page to ask questions about this result.")