from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the pre-trained XGBoost model
with open('XGBoost.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [int(request.form['Gender']), int(request.form['Married']), int(request.form['Dependents']),
                int(request.form['Education']), int(request.form['Self_Employed']),
                float(request.form['ApplicantIncome']), float(request.form['CoapplicantIncome']),
                float(request.form['LoanAmount']), float(request.form['Loan_Amount_Term']),
                float(request.form['Credit_History']), int(request.form['Property_Area'])]
    
    # Convert features to numpy array and reshape for prediction
    features_array = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features_array)
    
    # Interpret prediction
    if prediction[0] == 1:
        result = "You're loan application has Approved"
    else:
        result = "You're loan application has Rejcted"
    
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
