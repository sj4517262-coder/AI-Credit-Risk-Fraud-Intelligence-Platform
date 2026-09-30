import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")

CREDIT_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "credit_risk_model.pkl"
)

FRAUD_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "fraud_model.pkl"
)

CREDIT_SCALER_PATH = os.path.join(
    MODELS_DIR,
    "credit_scaler.pkl"
)

FRAUD_SCALER_PATH = os.path.join(
    MODELS_DIR,
    "fraud_scaler.pkl"
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR}/predictions.db"
)

APP_TITLE = "AI Credit Risk & Fraud Intelligence Platform"

APP_ICON = "🏦"

DEBUG_MODE = (
    os.getenv("DEBUG_MODE", "False").lower() == "true"
)