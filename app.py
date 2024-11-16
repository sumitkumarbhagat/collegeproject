from flask import Flask, render_template, request, redirect, url_for, flash, session
from sklearn import metrics
from sklearn.calibration import LabelEncoder
import numpy as np
import pickle
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

# Loading the trained model
import os
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(model_path, 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Used for session management and flash messages

# Dummy user database for demonstration (use a real DB in production)
users_db = {
    "sgtbmit@gmail.com": {"password": "123"}  # Example username/password
}

def plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']  # Get the username from form
        password = request.form['password']  # Get the password from form
        remember_me = 'remember_me' in request.form  # Check if "Remember Me" is checked

        # Check if user exists in the dummy database
        if username in users_db and users_db[username]['password'] == password:
            session['username'] = username  # Store username in session
            flash('Login successful!', 'success')  # Flash a success message
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials, please try again.', 'danger')

    return render_template('index1.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extracting the form data
        Item_Identifier = float(request.form["Item_Identifier"])
        Item_weight = float(request.form["Item_weight"])
        Item_Fat_Content = float(request.form["Item_Fat_Content"])
        Item_visibility = float(request.form["Item_visibility"])
        Item_Type = float(request.form["Item_Type"])
        Item_MPR = float(request.form["Item_MPR"])
        Outlet_identifier = float(request.form["Outlet_identifier"])
        Outlet_established_year = int(request.form["Outlet_established_year"])
        Outlet_size = float(request.form["Outlet_size"])
        Outlet_location_type = float(request.form["Outlet_location_type"])
        Outlet_type = float(request.form["Outlet_type"])

        # Predicting the sales
        features = np.array([[Item_Identifier, Item_weight, Item_Fat_Content, Item_visibility, Item_Type, 
                              Item_MPR, Outlet_identifier, Outlet_established_year, Outlet_size, 
                              Outlet_location_type, Outlet_type]], dtype=np.float32)
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=prediction)

@app.route("/performance", methods=['GET', 'POST'])
def performance():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Load the data
    data = pd.read_csv('train.csv')

    # Separate features and target
    X = data.drop(columns='Item_Outlet_Sales')
    y = data['Item_Outlet_Sales']

    # Apply label encoding to categorical columns
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    # Ensure all columns are numeric
    assert X.select_dtypes(include=['object']).empty, "Some columns are still not numeric."

    # Predict
    y_pred = model.predict(X)

    # Calculate performance metrics
    r2_score_value = metrics.r2_score(y, y_pred)
    mae = metrics.mean_absolute_error(y, y_pred)
    range_actual = np.max(y) - np.min(y)
    overall_accuracy = (1 - (mae / range_actual)) * 100

    # Plotting
    plt.figure(figsize=(14, 6))

    # Scatter plot for actual values
    plt.scatter(y, y, alpha=0.6, color='green', label='Actual Values', marker='o')

    # Scatter plot for predicted values
    plt.scatter(y, y_pred, alpha=0.6, color='blue', label='Predicted Values', marker='x')

    # Line for perfect prediction
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')

    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted Sales')
    plt.legend()

    # Convert plot to base64
    plt_image = plot_to_base64(plt)
    plt.close()

    return render_template('performance.html', r2_score=r2_score_value, mae=mae, overall_accuracy=overall_accuracy, plot_url=plt_image)

if __name__ == '__main__':
    app.run(debug=True)