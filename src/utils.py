import os
import sys
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_credit_data():
    df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "credit_risk_dataset.csv"))

    # remove impossible values (data quality, not leakage - safe to do here)
    df = df[df['person_age'] <= 100]
    df = df[df['person_emp_length'] <= df['person_age']]

    # encode categorical text to numbers
    df['person_home_ownership'] = df['person_home_ownership'].map(
        {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
    )
    df['loan_intent'] = df['loan_intent'].map(
        {'EDUCATION': 0, 'MEDICAL': 1, 'VENTURE': 2,
         'PERSONAL': 3, 'DEBTCONSOLIDATION': 4, 'HOMEIMPROVEMENT': 5}
    )
    df['loan_grade'] = df['loan_grade'].map(
        {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    )
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map(
        {'Y': 1, 'N': 0}
    )

    # NOTE: we do NOT fill missing values here anymore.
    # the pipeline's imputer handles that, fit only on training folds,
    # so there is no data leakage into validation/test.
    return df


def load_fraud_data():
    df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "fraud_sample.csv"))

    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    df['trans_day'] = df['trans_date_trans_time'].dt.dayofweek

    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (pd.Timestamp.now() - df['dob']).dt.days // 365

    df = df[['amt', 'trans_hour', 'trans_day', 'age',
             'merch_lat', 'merch_long', 'is_fraud']]

    return df


def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"no file at {path} — train the model first")
    return joblib.load(path)