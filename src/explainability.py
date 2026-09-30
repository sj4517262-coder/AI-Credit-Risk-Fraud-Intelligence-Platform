import os
import sys
import warnings
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import load_model

PLOTS_DIR = os.path.join(config.BASE_DIR, 'data', 'shap_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def transform_features(pipe, x):
    # run the input through every pipeline step except SMOTE and the model
    # so the data matches what the model actually sees
    data = x.copy()
    for name, step in pipe.named_steps.items():
        if name in ('model', 'smote'):
            continue
        data = step.transform(data)
    return data


def get_model_and_data(pipe, x):
    model = pipe.named_steps['model']
    transformed = transform_features(pipe, x)
    return model, transformed


def get_explainer(model, X, model_name):
    if any(m in model_name for m in ['forest', 'xgboost', 'boosting']):
        return shap.TreeExplainer(model)
    return shap.LinearExplainer(model, X)


def normalize_shap(raw):
    if isinstance(raw, list):
        raw = raw[1] if len(raw) > 1 else raw[0]
    raw = np.asarray(raw)
    if raw.ndim == 3:
        raw = raw[:, :, 1] if raw.shape[2] > 1 else raw[:, :, 0]
    if raw.ndim > 1:
        raw = raw[0]
    return np.ravel(raw)


def explain_single(input_data, model_type='credit'):
    if model_type == 'credit':
        pipe = load_model(config.CREDIT_MODEL_PATH)
        info = load_model(os.path.join(config.MODELS_DIR, 'credit_info.pkl'))
    else:
        pipe = load_model(config.FRAUD_MODEL_PATH)
        info = load_model(os.path.join(config.MODELS_DIR, 'fraud_info.pkl'))

    features = info['features']
    x = pd.DataFrame([input_data])[features]

    model, transformed = get_model_and_data(pipe, x)
    explainer = get_explainer(model, transformed, info['model_name'])
    raw = explainer.shap_values(transformed)
    vals = normalize_shap(raw)

    if len(vals) != len(features):
        raise ValueError(f"SHAP length {len(vals)} != {len(features)} features")

    contributions = sorted([
        {
            'feature': features[i],
            'shap_value': round(float(vals[i]), 4),
            'direction': 'increases risk' if float(vals[i]) > 0 else 'reduces risk',
            'importance': abs(float(vals[i]))
        }
        for i in range(len(features))
    ], key=lambda c: c['importance'], reverse=True)

    return {
        'contributions': contributions,
        'risk_factors': [c for c in contributions if c['shap_value'] > 0][:3],
        'safe_factors': [c for c in contributions if c['shap_value'] < 0][:3],
        'model': info['model_name']
    }