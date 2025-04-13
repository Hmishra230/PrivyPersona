from flask import Flask, request, jsonify, render_template
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

app = Flask(__name__)

# Initialize SQLite DB
def init_db():
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS behavior (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scroll_depth REAL,
                    dwell_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/track-scroll', methods=['POST'])
def track_scroll():
    data = request.json
    scroll_depth = data.get('scrollDepth')
    
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute("INSERT INTO behavior (scroll_depth) VALUES (?)", (scroll_depth,))
    conn.commit()
    conn.close()
    
    return jsonify({"status": "Scroll depth recorded"}), 200

@app.route('/track-dwell-time', methods=['POST'])
def track_dwell_time():
    data = request.json
    dwell_time = data.get('dwellTime')
    
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute("INSERT INTO behavior (dwell_time) VALUES (?)", (dwell_time,))
    conn.commit()
    conn.close()
    
    return jsonify({"status": "Dwell time recorded"}), 200

@app.route('/add-sample-data')
def add_sample_data():
    sample_data = [
        (75.0, 60.0),
        (20.0, 15.0),
        (40.0, 30.0),
        (85.0, 80.0),
        (30.0, 25.0)
    ]
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    for scroll_depth, dwell_time in sample_data:
        c.execute("INSERT INTO behavior (scroll_depth, dwell_time) VALUES (?, ?)", (scroll_depth, dwell_time))
    conn.commit()
    conn.close()
    return "Sample data added!"

def load_data():
    conn = sqlite3.connect('user_data.db')
    df = pd.read_sql_query("SELECT scroll_depth, dwell_time FROM behavior WHERE scroll_depth IS NOT NULL AND dwell_time IS NOT NULL", conn)
    conn.close()
    return df

def preprocess_data(df):
    df['scroll_depth_normalized'] = df['scroll_depth'] / df['scroll_depth'].max()
    return df

def train_model():
    df = load_data()

    if df.empty or df.shape[0] < 2:
        raise ValueError("Not enough data to train the model. Add more behavior data.")

    df = preprocess_data(df)

    df['behavior_persona'] = np.where(df['dwell_time'] > 50, 'high_intent_buyer', 'deal_seeker')

    X = df[['scroll_depth_normalized', 'dwell_time']]
    y = df['behavior_persona']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Classification Report:\n", classification_report(y_test, model.predict(X_test)))

    return model

@app.route('/personalize', methods=['POST'])
def personalize():
    try:
        data = request.json
        scroll_depth = data.get('scrollDepth', 50.0)
        dwell_time = data.get('dwellTime', 10.0)

        model = train_model()
        X_new = pd.DataFrame([[scroll_depth / 100.0, dwell_time]], columns=['scroll_depth_normalized', 'dwell_time'])

        persona = model.predict(X_new)[0]

        if persona == 'high_intent_buyer':
            content = "🔴 Showing premium products and express checkout!"
        else:
            content = "🟢 Highlighting discounts and best deals for you!"

        return jsonify({"personalized_content": content})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Unexpected error: " + str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
