import os
import sys
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import load_fraud_data, save_model, load_model


def build_pipeline(model_name, model):
    # only the linear model needs feature scaling
    if model_name == 'logistic_regression':
        return ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])

    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])


def train_fraud_model():

    df = load_fraud_data()

    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    features = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # fraud is highly imbalanced so the models lean harder on the rare class
    candidates = {

        'random_forest': (
            RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [8, 12, 16],
                'model__min_samples_split': [2, 5]
            }
        ),

        'xgboost': (
            XGBClassifier(
                scale_pos_weight=10,
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            ),
            {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [4, 6, 8],
                'model__learning_rate': [0.05, 0.1, 0.2]
            }
        ),

        'gradient_boosting': (
            GradientBoostingClassifier(
                random_state=42
            ),
            {
                'model__n_estimators': [100, 150, 200],
                'model__max_depth': [3, 5],
                'model__learning_rate': [0.05, 0.1]
            }
        ),

        'logistic_regression': (
            LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ),
            {
                'model__C': [0.1, 1.0, 10.0]
            }
        )
    }

    results = {}

    for name, (model, param_grid) in candidates.items():

        pipe = build_pipeline(name, model)

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            n_iter=10,
            cv=cv,
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        results[name] = {
            'pipe': search.best_estimator_,
            'cv_auc': search.best_score_,
            'best_params': search.best_params_
        }

        print(f"{name}: cv_auc={search.best_score_:.3f}  {search.best_params_}")

    best_name = max(results, key=lambda k: results[k]['cv_auc'])
    best_pipe = results[best_name]['pipe']

    y_prob = best_pipe.predict_proba(X_test)[:, 1]
    y_pred = best_pipe.predict(X_test)

    test_auc = roc_auc_score(y_test, y_prob)
    test_f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\nBest model: {best_name}")
    print(
        f"CV ROC-AUC={results[best_name]['cv_auc']:.3f}  "
        f"Test ROC-AUC={test_auc:.3f}  "
        f"F1={test_f1:.3f}"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=['normal', 'fraud']
        )
    )

    save_model(best_pipe, config.FRAUD_MODEL_PATH)

    save_model(
        {
            'model_name': best_name,
            'features': features
        },
        os.path.join(config.MODELS_DIR, 'fraud_info.pkl')
    )

    print("Fraud model saved")

    return best_pipe, features


def predict_fraud(input_data):
    # score a single transaction

    pipe = load_model(config.FRAUD_MODEL_PATH)
    info = load_model(os.path.join(config.MODELS_DIR, 'fraud_info.pkl'))

    x = pd.DataFrame([input_data])[info['features']]

    prob = pipe.predict_proba(x)[0]
    fraud_prob = float(prob[1])

    if fraud_prob >= 0.80:
        status = "Confirmed Fraud"
        action = "Block transaction"

    elif fraud_prob >= 0.50:
        status = "Suspected Fraud"
        action = "Hold for manual review"

    elif fraud_prob >= 0.30:
        status = "Suspicious"
        action = "Flag and monitor"

    else:
        status = "Normal"
        action = "Approve transaction"

    return {
        'status': status,
        'action': action,
        'fraud_probability': float(round(fraud_prob * 100, 2)),
        'safe_probability': float(round(float(prob[0]) * 100, 2)),
        'model': str(info['model_name'])
    }


if __name__ == "__main__":
    train_fraud_model()