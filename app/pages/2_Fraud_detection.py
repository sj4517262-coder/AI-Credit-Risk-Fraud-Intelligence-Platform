import os
import sys
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.fraud_model import predict_fraud
from src.explainability import explain_single
from src.database import log_fraud_prediction

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Fraud Detection")

with st.form("fraud_form"):
    c1, c2 = st.columns(2)

    with c1:
        amt = st.number_input("Transaction Amount (₹)", 1, 500000, 5000)
        hour = st.slider("Transaction Hour (0-23)", 0, 23, 14)
        day = st.selectbox("Day of Week", [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ])

    with c2:
        age = st.number_input("Card Holder Age", 18, 100, 35)
        merch_lat = st.number_input("Merchant Latitude", -90.0, 90.0, 28.6)
        merch_long = st.number_input("Merchant Longitude", -180.0, 180.0, 77.2)

    submitted = st.form_submit_button("Check Transaction", type="primary")

if submitted:
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
               "Friday": 4, "Saturday": 5, "Sunday": 6}

    input_data = {
        'amt': amt,
        'trans_hour': hour,
        'trans_day': day_map[day],
        'age': age,
        'merch_lat': merch_lat,
        'merch_long': merch_long
    }

    result = predict_fraud(input_data)
    shap_result = explain_single(input_data, model_type='fraud')
    log_fraud_prediction(input_data, result)

    st.divider()
    col1, col2 = st.columns([1, 2])

    with col1:
        fraud_prob = result['fraud_probability']
        if fraud_prob >= 80:
            bar_color = "#c62828"
        elif fraud_prob >= 50:
            bar_color = "#ef6c00"
        elif fraud_prob >= 30:
            bar_color = "#f9a825"
        else:
            bar_color = "#2e7d32"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob,
            title={'text': "Fraud Probability (%)", 'font': {'size': 20}},
            number={'font': {'size': 44, 'color': bar_color}, 'suffix': "%"},
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickmode': 'array',
                    'tickvals': [0, 25, 50, 75, 100],
                    'tickcolor': "gray"
                },
                'bar': {'color': bar_color, 'thickness': 0.3},
                'steps': [
                    {'range': [0, 30], 'color': "#66bb6a"},
                    {'range': [30, 50], 'color': "#ffee58"},
                    {'range': [50, 80], 'color': "#ffa726"},
                    {'range': [80, 100], 'color': "#ef5350"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'value': fraud_prob
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        if result['status'] in ["Confirmed Fraud", "Suspected Fraud"]:
            st.error(f"**{result['status']}**")
        else:
            st.success(f"**{result['status']}**")
        st.write(f"**Recommended action:** {result['action']}")

        st.markdown("##### Transaction Summary")
        summary = {
            "Amount": f"₹{amt:,}",
            "Hour": f"{hour}:00",
            "Day": day,
            "Card Holder Age": age,
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
            xaxis_title="← suggests normal   |   suggests fraud →",
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.session_state['last_fraud_input'] = input_data
    st.session_state['last_fraud_result'] = result
    st.session_state['last_fraud_shap'] = shap_result
    st.success("Assessment saved. Go to AI Assistant page to ask questions about this result.")