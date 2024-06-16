import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template_string

# Function to prepare data for classification
def prepare_data(age, genetic_risk, alcohol_use, smoking, gender, weight, height, family_history, diet, physical_activity):
    data = {
        'Age': [age],
        'Genetic Risk': [genetic_risk],
        'Alcohol use': [alcohol_use],
        'Smoking': [smoking],
        'Gender': [gender],
        'Weight': [weight],
        'Height': [height],
        'Family History': [family_history],
        'Diet': [diet],
        'Physical Activity': [physical_activity]
    }
    df = pd.DataFrame(data)
    return df

# Function to train the model and predict
def train_model_and_predict(X):
    # Example model (Support Vector Machine)
    sv = svm.SVC(probability=True)
    
    # Mocking training data for demonstration
    mock_data = {
        'Age': [25, 35, 45, 55, 65, 30, 40, 50, 60, 20, 70, 80, 90, 100, 110],
        'Genetic Risk': [0.1, 0.4, 0.5, 0.7, 0.8, 0.3, 0.5, 0.6, 0.7, 0.2, 0.9, 0.1, 0.4, 0.5, 0.7],
        'Alcohol use': [0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        'Smoking': [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
        'Gender': [1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1],
        'Weight': [70, 80, 90, 100, 110, 75, 85, 95, 105, 60, 120, 70, 80, 90, 100],
        'Height': [170, 175, 180, 185, 190, 172, 177, 182, 187, 165, 195, 170, 175, 180, 185],
        'Family History': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],  # 1 for Yes, 0 for No
        'Diet': [1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1],  # 1 for Healthy, 2 for Unhealthy
        'Physical Activity': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0]  # 1 for Active, 0 for Inactive
    }
    mock_target = ['Low', 'Medium', 'Medium', 'High', 'High', 'Low', 'Medium', 'Medium', 'High', 'Low', 'High', 'Low', 'high', 'high', 'High']
    
    mock_df = pd.DataFrame(mock_data)
    X_train, X_test, y_train, y_test = train_test_split(mock_df, mock_target, test_size=0.2, random_state=42)
    
    # Standardize the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X)
    
    # Fit the model
    sv.fit(X_train_scaled, y_train)
    
    # Predict probabilities
    y_pred_proba = sv.predict_proba(X_scaled)
    
    # Predicted class (assuming multi-class classification)
    y_pred_class = sv.predict(X_scaled)
    
    # Calculate patient risk level
    risk_level = calculate_risk_level(y_pred_class[0])
    
    return y_pred_class, risk_level

def calculate_risk_level(predicted_class):
    # Assuming three classes: 'Low', 'Medium', 'High'
    risk_levels = {'Low': 'Low', 'Medium': 'Medium', 'High': 'High'}
    return risk_levels[predicted_class]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string('''
        <form action="/predict" method="post">
            <label for="age">Age:</label>
            <input type="number" id="age" name="age" required><br><br>
            
            <label for="genetic_risk">Genetic Risk:</label>
            <input type="number" step="0.01" id="genetic_risk" name="genetic_risk" required><br><br>
            
            <label for="alcohol_use">Alcohol use (0 or 1):</label>
            <input type="number" id="alcohol_use" name="alcohol_use" min="0" max="1" required><br><br>
            
            <label for="smoking">Smoking status (0 or 1):</label>
            <input type="number" id="smoking" name="smoking" min="0" max="1" required><br><br>
            
            <label for="gender">Gender (1 for male, 2 for female):</label>
            <input type="number" id="gender" name="gender" min="1" max="2" required><br><br>
            
            <label for="weight">Weight (kg):</label>
            <input type="number" id="weight" name="weight" required><br><br>
            
            <label for="height">Height (cm):</label>
            <input type="number" id="height" name="height" required><br><br>
            
            <label for="family_history">Family history of cancer (1 for Yes, 0 for No):</label>
            <input type="number" id="family_history" name="family_history" min="0" max="1" required><br><br>
            
            <label for="diet">Diet (1 for Healthy, 2 for Unhealthy):</label>
            <input type="number" id="diet" name="diet" min="1" max="2" required><br><br>
            
            <label for="physical_activity">Physical activity level (1 for Active, 0 for Inactive):</label>
            <input type="number" id="physical_activity" name="physical_activity" min="0" max="1" required><br><br>
            
            <input type="submit" value="Predict Risk">
        </form>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    genetic_risk = float(request.form['genetic_risk'])
    alcohol_use = float(request.form['alcohol_use'])
    smoking = float(request.form['smoking'])
    gender = int(request.form['gender'])
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    family_history = int(request.form['family_history'])
    diet = int(request.form['diet'])
    physical_activity = int(request.form['physical_activity'])
    
    # Prepare the data for prediction
    user_data = prepare_data(age, genetic_risk, alcohol_use, smoking, gender, weight, height, family_history, diet, physical_activity)
    X = user_data[['Age', 'Genetic Risk', 'Alcohol use', 'Smoking', 'Gender', 'Weight', 'Height', 'Family History', 'Diet', 'Physical Activity']]
    
    # Predict the user's risk level
    user_preds, risk_level = train_model_and_predict(X)
    
    return f'''
        <h1>Predicted Level: {user_preds[0]}</h1>
        <h2>Risk Level: {risk_level}</h2>
        <a href="/">Back to Input</a>
    '''

if __name__ == '__main__':
    app.run(debug=True)
