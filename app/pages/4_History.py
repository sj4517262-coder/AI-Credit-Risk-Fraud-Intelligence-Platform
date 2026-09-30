import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.database import get_credit_predictions, get_fraud_predictions

st.set_page_config(page_title="History", page_icon="📜", layout="wide")

st.title("📜 Prediction History")
st.caption("A complete audit trail of every assessment made on this platform")

credit = get_credit_predictions()
fraud = get_fraud_predictions()

st.write("")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Credit Assessments", len(credit))
high_risk = len([r for r in credit if r['risk_label'] in ['High Risk', 'Very High Risk']])
c2.metric("High Risk Found", high_risk)
c3.metric("Fraud Checks", len(fraud))
frauds = len([r for r in fraud if r['status'] in ['Confirmed Fraud', 'Suspected Fraud']])
c4.metric("Fraud Flagged", frauds)

st.divider()

tab1, tab2 = st.tabs(["💳  Credit Risk", "🛡️  Fraud Detection"])


def clean_number(value):
    # convert bytes or any weird type into a plain float safely
    try:
        if isinstance(value, bytes):
            import struct
            return round(struct.unpack('<f', value)[0], 1)
        return round(float(value), 1)
    except Exception:
        return 0.0


with tab1:
    if credit:
        rows = []
        for r in credit:
            rows.append({
                'Time': r['timestamp'],
                'Age': r['age'],
                'Income (₹)': f"₹{int(r['income']):,}",
                'Loan (₹)': f"₹{int(r['loan_amount']):,}",
                'Risk Level': r['risk_label'],
                'Score': r['risk_score'],
                'Risk %': f"{clean_number(r['risk_probability'])}%",
                'Model': r['model']
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇  Download as CSV", csv, "credit_history.csv", "text/csv")
    else:
        st.info("No credit assessments recorded yet.")

with tab2:
    if fraud:
        rows = []
        for r in fraud:
            rows.append({
                'Time': r['timestamp'],
                'Amount (₹)': f"₹{int(r['amount']):,}",
                'Hour': r['trans_hour'],
                'Status': r['status'],
                'Fraud %': f"{clean_number(r['fraud_probability'])}%",
                'Model': r['model']
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇  Download as CSV", csv, "fraud_history.csv", "text/csv")
    else:
        st.info("No fraud checks recorded yet.")