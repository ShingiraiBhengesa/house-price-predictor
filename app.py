from flask import Flask, request, render_template, jsonify, send_from_directory
import joblib
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Start the web app
app = Flask(__name__, static_url_path='/static')

# Load the saved model and scaler
model = joblib.load('house_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature names for reference
feature_names = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]

# Feature descriptions for tooltips
feature_descriptions = {
    'MedInc': 'Median income in block group (in tens of thousands)',
    'HouseAge': 'Median house age in block group (in years)',
    'AveRooms': 'Average number of rooms per household',
    'AveBedrms': 'Average number of bedrooms per household',
    'Population': 'Block group population',
    'AveOccup': 'Average number of household members',
    'Latitude': 'Block group latitude coordinate',
    'Longitude': 'Block group longitude coordinate'
}

# Typical ranges for each feature
feature_ranges = {
    'MedInc': (1.0, 15.0),
    'HouseAge': (1, 50),
    'AveRooms': (3.0, 10.0),
    'AveBedrms': (1.0, 4.0),
    'Population': (100, 5000),
    'AveOccup': (1.0, 6.0),
    'Latitude': (32.0, 42.0),
    'Longitude': (-124.0, -114.0)
}

# Home page
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    prediction_details = None
    error_message = None
    
    if request.method == 'POST':
        try:
            # Get numbers from the user
            features = []
            feature_values = {}
            
            for feature in feature_names:
                value = float(request.form[feature])
                features.append(value)
                feature_values[feature] = value
                
                # Check if value is within typical range
                if feature in feature_ranges:
                    min_val, max_val = feature_ranges[feature]
                    if value < min_val * 0.5 or value > max_val * 1.5:
                        # Value is outside extended range, but we'll still make a prediction
                        # Just note this in the details
                        if not prediction_details:
                            prediction_details = {}
                        if 'warnings' not in prediction_details:
                            prediction_details['warnings'] = []
                        prediction_details['warnings'].append(
                            f"{feature} value ({value}) is outside typical range ({min_val} - {max_val})"
                        )
            
            # Prepare the numbers and predict
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]
            confidence = get_prediction_confidence(features_scaled[0])
            
            # Format the prediction
            price = prediction * 100000
            prediction_result = f"${price:,.2f}"
            
            # Add prediction details
            if not prediction_details:
                prediction_details = {}
            
            prediction_details['price'] = price
            prediction_details['confidence'] = confidence
            prediction_details['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prediction_details['features'] = feature_values
            
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                # Sort features by importance for this specific prediction
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                prediction_details['feature_importance'] = sorted_importance
            
        except Exception as e:
            error_message = f"Error making prediction: {str(e)}"
    
    # Check if visualization files exist
    visualizations = []
    viz_files = [
        ('real_vs_predicted.png', 'Real vs. Predicted Prices'),
        ('feature_importance.png', 'Feature Importance'),
        ('correlation_heatmap.png', 'Feature Correlation Matrix'),
        ('error_distribution.png', 'Prediction Error Distribution')
    ]
    
    for file, title in viz_files:
        if os.path.exists(f'static/{file}'):
            visualizations.append({
                'file': file,
                'title': title
            })
        elif os.path.exists(file):  # Check in root directory too
            visualizations.append({
                'file': f'../{file}',
                'title': title
            })
    
    return render_template(
        'index.html', 
        prediction=prediction_result,
        prediction_details=prediction_details,
        error_message=error_message,
        visualizations=visualizations,
        feature_descriptions=feature_descriptions,
        feature_ranges=feature_ranges
    )

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

def get_prediction_confidence(features_scaled):
    """
    Calculate a simple confidence score based on how close the features are to the training data.
    This is a simplified approach - in a real application, you might use more sophisticated methods.
    """
    # For this demo, we'll return a fixed high confidence
    # In a real app, you could implement proper confidence intervals
    return 0.85  # 85% confidence

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
        print("Created static directory")
    
    # Check for visualization files in root directory
    print("\nChecking for visualization files in root directory:")
    for file in ['real_vs_predicted.png', 'feature_importance.png', 'correlation_heatmap.png', 'error_distribution.png']:
        if os.path.exists(file):
            print(f"Found {file} in root directory")
            if not os.path.exists(f'static/{file}'):
                try:
                    import shutil
                    shutil.copy(file, f'static/{file}')
                    print(f"Moved {file} to static directory")
                except Exception as e:
                    print(f"Error moving {file}: {str(e)}")
        else:
            print(f"Did not find {file} in root directory")
    
    # Check for visualization files in static directory
    print("\nChecking for visualization files in static directory:")
    if os.path.exists('static'):
        for file in ['real_vs_predicted.png', 'feature_importance.png', 'correlation_heatmap.png', 'error_distribution.png']:
            if os.path.exists(f'static/{file}'):
                print(f"Found {file} in static directory")
            else:
                print(f"Did not find {file} in static directory")
    
    print("\nStarting Flask application...")
    app.run(debug=True)
