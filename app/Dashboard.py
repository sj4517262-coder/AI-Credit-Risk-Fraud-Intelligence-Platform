import streamlit as st

st.set_page_config(
    page_title="Credit Risk & Fraud Intelligence Platform",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
    .hero {
        padding: 2.5rem 2rem;
        border-radius: 16px;
        background: #1e2229;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: white;
        font-size: 2.3rem;
        margin-bottom: 0.4rem;
    }
    .hero p {
        color: #b0b8c4;
        font-size: 1.05rem;
        margin: 0;
    }
    .feature-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.4rem;
        height: 100%;
    }
    .feature-card h3 {
        margin-top: 0;
        font-size: 1.15rem;
    }
    .feature-card p {
        color: #9aa4b2;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .step-badge {
        display: inline-block;
        background: #2d3139;
        color: white;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.8rem;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🏦 AI Credit Risk & Fraud Intelligence Platform</h1>
    <p>Explainable machine learning for credit scoring and fraud detection, with a built-in GenAI risk assistant.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("ML Models", "2", "Credit + Fraud")
c2.metric("Algorithms Compared", "4", "per model")
c3.metric("Explainability", "SHAP", "active")
c4.metric("AI Assistant", "Gemini", "connected")

st.write("")

st.subheader("What this platform does")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>💳 Credit Risk Scoring</h3>
        <p>Predicts loan default probability using an XGBoost model trained on real lending data,
        with an intuitive 0–1000 risk score and clear approve/reject recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🛡️ Fraud Detection</h3>
        <p>Flags suspicious transactions in real time based on amount, timing, location and
        cardholder behaviour, catching fraud patterns humans would miss.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>🧠 Explainable AI</h3>
        <p>Every decision comes with a SHAP-based breakdown and a plain-English explanation
        from Gemini, so no prediction is ever a black box.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

st.subheader("How it works")

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.markdown('<span class="step-badge">Step 1</span>', unsafe_allow_html=True)
    st.markdown("**Enter details**")
    st.caption("Fill in applicant or transaction information on the relevant page.")

with s2:
    st.markdown('<span class="step-badge">Step 2</span>', unsafe_allow_html=True)
    st.markdown("**Get a prediction**")
    st.caption("The model returns a risk score and a clear recommendation instantly.")

with s3:
    st.markdown('<span class="step-badge">Step 3</span>', unsafe_allow_html=True)
    st.markdown("**See why**")
    st.caption("SHAP shows exactly which factors drove the decision.")

with s4:
    st.markdown('<span class="step-badge">Step 4</span>', unsafe_allow_html=True)
    st.markdown("**Ask the assistant**")
    st.caption("Chat with the GenAI assistant for a plain-English explanation.")

st.write("")
st.divider()

st.info("👈 Use the sidebar to navigate between Credit Risk, Fraud Detection, the AI Assistant, and History.")

st.caption("Built as a production-style fintech prototype — machine learning, explainability, GenAI, and persistence in one platform.")