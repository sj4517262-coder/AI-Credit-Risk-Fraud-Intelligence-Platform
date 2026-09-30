import os
import sys
import sqlite3
from datetime import datetime

# let python find config.py in the main project folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# the database will be a single file called predictions.db in the project root
DB_PATH = os.path.join(config.BASE_DIR, 'predictions.db')


def init_db():
    # connect to the database (creates the file if it does not exist)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # create the table for credit predictions if it is not already there
    cur.execute("""
        CREATE TABLE IF NOT EXISTS credit_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age INTEGER,
            income REAL,
            loan_amount REAL,
            risk_label TEXT,
            risk_score INTEGER,
            risk_probability REAL,
            model TEXT
        )
    """)

    # create the table for fraud predictions if it is not already there
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fraud_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            amount REAL,
            trans_hour INTEGER,
            status TEXT,
            fraud_probability REAL,
            model TEXT
        )
    """)

    conn.commit()
    conn.close()


def log_credit_prediction(input_data, result):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # wrap every value in int/float/str to convert numpy types to plain python
    # types, otherwise sqlite stores them as bytes and breaks later
    cur.execute("""
        INSERT INTO credit_predictions
        (timestamp, age, income, loan_amount, risk_label,
         risk_score, risk_probability, model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        int(input_data['person_age']),
        float(input_data['person_income']),
        float(input_data['loan_amnt']),
        str(result['risk_label']),
        int(result['risk_score']),
        float(result['risk_probability']),
        str(result['model'])
    ))

    conn.commit()
    conn.close()


def log_fraud_prediction(input_data, result):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO fraud_predictions
        (timestamp, amount, trans_hour, status, fraud_probability, model)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        float(input_data['amt']),
        int(input_data['trans_hour']),
        str(result['status']),
        float(result['fraud_probability']),
        str(result['model'])
    ))

    conn.commit()
    conn.close()


def get_credit_predictions():
    # read all credit predictions back, newest first
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    cur = conn.cursor()
    cur.execute("SELECT * FROM credit_predictions ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_fraud_predictions():
    # read all fraud predictions back, newest first
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM fraud_predictions ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# make sure the tables exist as soon as this file is imported anywhere
init_db()


if __name__ == "__main__":
    init_db()
    print("database ready")
    print("credit predictions:", len(get_credit_predictions()))
    print("fraud predictions:", len(get_fraud_predictions()))