import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.credit_model import predict_credit_batch

st.set_page_config(page_title="Batch Scoring", page_icon="📂", layout="wide")
st.title("📂 Batch Credit Risk Scoring")
st.caption("Upload a CSV of applicants and score all of them at once.")

home_map = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
intent_map = {"EDUCATION": 0, "MEDICAL": 1, "VENTURE": 2,
              "PERSONAL": 3, "DEBTCONSOLIDATION": 4, "HOMEIMPROVEMENT": 5}
grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
default_map = {"Y": 1, "N": 0}

required_cols = [
    'person_age', 'person_income', 'person_home_ownership',
    'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
    'loan_int_rate', 'loan_percent_income',
    'cb_person_default_on_file', 'cb_person_cred_hist_length'
]

guide = pd.DataFrame({
    "Column": required_cols,
    "Meaning": [
        "Applicant's age (18-100)",
        "Annual income in ₹",
        "RENT, OWN, MORTGAGE or OTHER",
        "Years employed",
        "EDUCATION, MEDICAL, VENTURE, PERSONAL, DEBTCONSOLIDATION or HOMEIMPROVEMENT",
        "Loan grade: A (best) to G (worst)",
        "Loan amount requested in ₹",
        "Interest rate (e.g. 11.5)",
        "Loan amount ÷ income (e.g. 0.3)",
        "Y if defaulted before, else N",
        "Years of credit history"
    ],
    "Example": [
        "25", "45000", "RENT", "2", "PERSONAL", "C",
        "15000", "13.5", "0.33", "N", "3"
    ]
})

with st.expander("📖 How to prepare your file (click to read)", expanded=True):
    st.write("Your CSV must contain these columns. Here's what each one means:")
    st.dataframe(guide, use_container_width=True, hide_index=True)

    sample = pd.DataFrame({
        'person_age': [25, 40],
        'person_income': [45000, 90000],
        'person_home_ownership': ['RENT', 'OWN'],
        'person_emp_length': [2, 15],
        'loan_intent': ['PERSONAL', 'EDUCATION'],
        'loan_grade': ['C', 'A'],
        'loan_amnt': [15000, 8000],
        'loan_int_rate': [13.5, 7.2],
        'loan_percent_income': [0.33, 0.09],
        'cb_person_default_on_file': ['N', 'N'],
        'cb_person_cred_hist_length': [3, 20]
    })
    template_csv = sample.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇  Download blank template (fill it and re-upload)",
        template_csv, "applicant_template.csv", "text/csv"
    )

st.divider()

uploaded = st.file_uploader("Upload your applicant CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.write(f"Uploaded {len(df)} applicants.")
    st.dataframe(df.head(), use_container_width=True, hide_index=True)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Your file is missing these columns: {', '.join(missing)}")
        st.info("Download the template above to get the correct format.")
        st.stop()

    if st.button("Score All Applicants", type="primary"):
        # convert all text columns to numbers at once
        work = df.copy()
        work['person_home_ownership'] = work['person_home_ownership'].astype(str).str.upper().map(home_map)
        work['loan_intent'] = work['loan_intent'].astype(str).str.upper().map(intent_map)
        work['loan_grade'] = work['loan_grade'].astype(str).str.upper().map(grade_map)
        work['cb_person_default_on_file'] = work['cb_person_default_on_file'].astype(str).str.upper().map(default_map)

        with st.spinner(f"Scoring {len(work)} applicants..."):
            preds = predict_credit_batch(work)

        results = []
        for i in range(len(df)):
            results.append({
                'Age': df.iloc[i]['person_age'],
                'Income': df.iloc[i]['person_income'],
                'Loan Amount': df.iloc[i]['loan_amnt'],
                'Risk Level': preds[i]['risk_label'],
                'Risk Score': preds[i]['risk_score'],
                'Risk %': preds[i]['risk_probability'],
                'Recommendation': preds[i]['action']
            })

        result_df = pd.DataFrame(results)
        st.success(f"Scored all {len(result_df)} applicants!")

        c1, c2, c3 = st.columns(3)
        high = len(result_df[result_df['Risk Level'].isin(['High Risk', 'Very High Risk'])])
        c1.metric("Total Scored", len(result_df))
        c2.metric("High Risk", high)
        c3.metric("Approval Rate", f"{round((len(result_df)-high)/len(result_df)*100)}%")

        st.markdown("##### Risk Distribution")
        counts = result_df['Risk Level'].value_counts().reset_index()
        counts.columns = ['Risk Level', 'Count']
        fig = px.bar(counts, x='Risk Level', y='Count', color='Risk Level',
                     color_discrete_map={
                         'Low Risk': '#2e7d32',
                         'Medium Risk': '#f9a825',
                         'High Risk': '#ef6c00',
                         'Very High Risk': '#c62828'
                     })
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Full Results")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇  Download Results (CSV)", csv,
                           "scored_applicants.csv", "text/csv", type="primary")