from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)

# Load the classification model
classification_model_file_path = "rf_classifier_model.pkl"
classification_model = joblib.load(classification_model_file_path)

# Load the regression model
regression_model_file_path = "rf_regressor_model.pkl"
regression_model = joblib.load(regression_model_file_path)

# Define the selected features used for training
selected_features_classification = ['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10', 
                                    'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10', 
                                    'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10', 
                                    'Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10', 
                                    'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10', 
                                    'RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']

selected_features_regression = ['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10', 
                                'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10', 
                                'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10', 
                                'Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10', 
                                'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10', 
                                'RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']

@app.route('/')
def about():
    return render_template('home.html')
@app.route('/project')
def index():
    return render_template('project.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Get the uploaded CSV file from the request
    csv_file = request.files['csvFile']
    
    # Read the CSV file into a pandas DataFrame
    eeg_data_to_predict = pd.read_csv(csv_file)
    
    # Select features for classification
    X_classification = eeg_data_to_predict[selected_features_classification]

    # Predict classes for new data using the classification model
    predicted_classes = classification_model.predict(X_classification)

    # Convert predicted classes to strings
    predicted_labels_classification = ['Engaging' if label == 1 else 'Non-Engaging' for label in predicted_classes]
    
    # Select features for regression
    X_regression = eeg_data_to_predict[selected_features_regression]

    # Predict S_L values for new data using the regression model
    predicted_s_l = regression_model.predict(X_regression)

    # Redirect to the result page with predicted labels and S_L values as parameters
    return redirect(url_for('result', labels_classification=predicted_labels_classification, predicted_s_l=predicted_s_l))

@app.route('/result')
def result():
    predicted_labels_classification = request.args.get('labels_classification').split(',')
    predicted_s_l = request.args.get('predicted_s_l').split(',')
    return render_template('result.html', predicted_labels_classification=predicted_labels_classification, predicted_s_l=predicted_s_l)



@app.route('/about')
def team():
    return render_template('about.html')
    
@app.route('/contact')
def ex():
    return render_template('contact.html')
    

if __name__ == '__main__':
    app.run(debug=True)
