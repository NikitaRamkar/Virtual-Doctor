from flask import Flask, request, redirect, url_for, render_template, send_from_directory, session, jsonify
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient
from datetime import datetime

# Initialize Flask app
app = Flask(__name__, static_folder='assets')
app.secret_key = 'your_secret_key'

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['VirtualDoctor']
login_collection = db['login_signup']
results_collection = db['results']
daily_logs_collection = db['daily_logs']

# Model and dataset paths
MODEL_FILE = "depression_model.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
DATASET_FILE = r"C:\\Users\\HP\\OneDrive\\Desktop\\Virtual Doctor\\models\\models\\depression_dataset.xlsx"

# Train and save model
def train_and_save_model():
    df = pd.read_excel(DATASET_FILE)
    if 'Total_Score' not in df.columns:
        df['Total_Score'] = df[[f"Q{i}" for i in range(1, 10)]].sum(axis=1)

    X = df[[f"Q{i}" for i in range(1, 10)]]
    Z = df["Depression_Level"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(Z)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(le, LABEL_ENCODER_FILE)

# Train if model not found
if not os.path.exists(MODEL_FILE) or not os.path.exists(LABEL_ENCODER_FILE):
    train_and_save_model()

# Load model
model = joblib.load(MODEL_FILE)
label_encoder = joblib.load(LABEL_ENCODER_FILE)

# Routes
@app.route('/')
def home():
    return send_from_directory(os.path.join(app.root_path, 'templates'), 'index.html')

@app.route('/test')
def test():
    if 'user' not in session:
        return redirect(url_for('login'))
    return send_from_directory(os.path.join(app.root_path, 'templates'), 'test.html')

@app.route('/submit_test', methods=['POST'])
def submit_test():
    try:
        email = session.get("user")
        if not email:
            return redirect(url_for('login'))

        answers = [int(request.form[f'q{i}']) for i in range(1, 10)]
        pred_encoded = model.predict([answers])[0]
        result = label_encoder.inverse_transform([pred_encoded])[0]

        results_collection.insert_one({
            "email": email,
            "answers": answers,
            "result": result,
            "timestamp": datetime.now()
        })

        return redirect(url_for('result', level=result))
    except Exception as e:
        return f"Error processing request: {str(e)}", 400

@app.route('/result')
def result():
    level = request.args.get('level', 'Unknown')
    return render_template('result.html', level=level)

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect('/login')

    email = session['user']

    # Get user's full name
    user = login_collection.find_one({"email": email})
    full_name = user['fullName'] if user and 'fullName' in user else "User"

    # Fetch test results
    test_results = list(results_collection.find(
        {"email": email},
        {"_id": 0, "result": 1, "timestamp": 1}
    ))

    for test in test_results:
        if isinstance(test['timestamp'], str):
            test['date'] = datetime.strptime(test['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d")
        else:
            test['date'] = test['timestamp'].strftime("%Y-%m-%d")

    # Fetch daily mood logs
    logs = list(daily_logs_collection.find(
        {"email": email},
        {"_id": 0}
    ))

    for log in logs:
        if isinstance(log['date'], datetime):
            log['date'] = log['date'].strftime("%Y-%m-%d")

    # Extract completed day numbers from logs
    completed_days = [log['day'] for log in logs if 'day' in log]

    # Render profile with all data
    return render_template(
        'profile.html',
        test_results=test_results,
        full_name=full_name,
        logs=logs,
        completed_days=completed_days
    )

# ✅ Updated route to support ?day=... from calendar
@app.route('/logs')
def logs():
    if 'user' not in session:
        return redirect(url_for('login'))

    day = request.args.get('day')  # Extract day from calendar
    return render_template('logs.html', day=day)

@app.route('/submit_log', methods=['POST'])
def submit_log():
    if 'user' not in session:
        return redirect(url_for('login'))

    email = session['user']
    try:
        mood = int(request.form.get('mood'))
        sleep_duration = request.form.get('sleep_duration')
        sleep_quality = int(request.form.get('sleep_quality'))
        energy = int(request.form.get('energy'))
        motivation = int(request.form.get('motivation'))
        self_care = request.form.get('self_care')
        mindfulness = request.form.get('mindfulness')
        social = request.form.get('social')
        day_rating = int(request.form.get('day_rating'))
        activities = request.form.getlist('activities')
        day = int(request.form.get('day')) if request.form.get('day') else None

        log_entry = {
            "email": email,
            "date": datetime.now(),
            "mood": mood,
            "sleep_duration": sleep_duration,
            "sleep_quality": sleep_quality,
            "energy": energy,
            "motivation": motivation,
            "activities": activities,
            "self_care": self_care,
            "mindfulness": mindfulness,
            "social": social,
            "day_rating": day_rating,
        }

        if day:
            log_entry["day"] = day

        daily_logs_collection.insert_one(log_entry)
        return redirect(url_for('profile'))

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return send_from_directory(os.path.join(app.root_path, 'templates'), 'login_signup.html')
    else:
        email = request.form['email']
        password = request.form['password']
        user = login_collection.find_one({"email": email})
        if user and user['password'] == password:
            session['user'] = email
            return redirect(url_for('home'))
        return "Invalid email or password", 401

@app.route('/signup', methods=['POST'])
def signup():
    data = request.form
    existing = login_collection.find_one({"email": data['email']})
    if existing:
        return "User already exists", 400
    login_collection.insert_one({
        "fullName": data['fullName'],
        "email": data['email'],
        "password": data['password'],
        "age": int(data['age']),
        "gender": data['gender'],
        "timestamp": datetime.now()
    })
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return send_from_directory(os.path.join(app.root_path, 'templates'), 'logout.html')

if __name__ == '__main__':
    app.run(debug=True)
