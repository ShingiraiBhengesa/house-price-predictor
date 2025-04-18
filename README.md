# House Price Predictor

![House Price Predictor](https://img.shields.io/badge/ML-House%20Price%20Predictor-blue)
![Python](https://img.shields.io/badge/Python-3.x-green)
![Flask](https://img.shields.io/badge/Flask-Web%20App-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)

A web application that predicts house prices in California using machine learning. This project uses the California Housing Dataset and implements a Random Forest Regression model to provide accurate price predictions based on various housing features.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

This application allows users to input various housing characteristics such as median income, house age, average rooms, population, and geographic coordinates to predict the median house value in a neighborhood. The prediction model is trained on the California Housing Dataset using a Random Forest Regressor algorithm.

The web interface provides an intuitive form for entering housing details and displays the predicted price along with confidence metrics and visualizations of the model's performance.

## Features

- **User-friendly Web Interface**: Clean, responsive design with Bootstrap
- **Real-time Predictions**: Instant house price predictions based on user inputs
- **Data Validation**: Input validation with helpful tooltips and typical value ranges
- **Confidence Metrics**: Displays prediction confidence and feature importance
- **Interactive Visualizations**: Shows model performance and feature relationships
- **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
house-price-predictor/
├── app.py                  # Flask web application
├── main.py                 # Model training script
├── house_model.pkl         # Trained Random Forest model
├── scaler.pkl              # Feature scaler for data preprocessing
├── real_vs_predicted.png   # Model performance visualization
├── static/                 # Static files (images, CSS)
│   ├── real_vs_predicted.png
│   ├── feature_importance.png
│   ├── correlation_heatmap.png
│   └── error_distribution.png
├── templates/              # HTML templates
│   └── index.html          # Main web interface
└── README.md               # Project documentation
```

## Model Information

The prediction model uses a **Random Forest Regressor** algorithm trained on the California Housing Dataset. The model:

- Uses 8 input features (median income, house age, average rooms, etc.)
- Was trained with hyperparameter tuning using GridSearchCV
- Achieves high accuracy with low prediction error
- Provides feature importance analysis

### Features Used

| Feature | Description | Typical Range |
|---------|-------------|---------------|
| MedInc | Median income in block group (in tens of thousands) | 1.0 - 15.0 |
| HouseAge | Median house age in block group (in years) | 1 - 50 |
| AveRooms | Average number of rooms per household | 3.0 - 10.0 |
| AveBedrms | Average number of bedrooms per household | 1.0 - 4.0 |
| Population | Block group population | 100 - 5000 |
| AveOccup | Average number of household members | 1.0 - 6.0 |
| Latitude | Block group latitude coordinate | 32.0 - 42.0 |
| Longitude | Block group longitude coordinate | -124.0 - -114.0 |

## Installation

### Prerequisites

- Python 3.6+
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-predictor.git
   cd house-price-predictor
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install flask numpy pandas scikit-learn matplotlib seaborn joblib
   ```

## Usage

### Training the Model

If you want to retrain the model with the latest data:

```bash
python main.py
```

This script:
1. Loads the California Housing Dataset
2. Preprocesses the data and performs feature scaling
3. Trains a Random Forest model with hyperparameter tuning
4. Evaluates the model performance
5. Generates visualizations
6. Saves the trained model and scaler

### Running the Web Application

To start the web application:

```bash
python app.py
```

Then open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### Making Predictions

1. Fill in the form with housing details
2. Click the "Predict House Price" button
3. View the predicted price and additional information

## Screenshots

### Web Interface

The application features a clean, modern interface with an intuitive form for entering housing details:

![Web Interface Screenshot](static/web_interface.jpg)

*Note: The actual web interface looks like the image below when running the application*

```
+-----------------------------------------------+
|  House Price Predictor                      |
+-----------------------------------------------+
|                                             |
|  Predict Your House Price                   |
|  Enter your house details below...          |
|                                             |
|  +-----------------------------------+      |
|  | House Details                     |      |
|  +-----------------------------------+      |
|  | Median Income: [____]             |      |
|  | House Age: [____]                 |      |
|  | Average Rooms: [____]             |      |
|  | Average Bedrooms: [____]          |      |
|  | Population: [____]                |      |
|  | Average Occupancy: [____]         |      |
|  | Latitude: [____]                  |      |
|  | Longitude: [____]                 |      |
|  |                                   |      |
|  | [Predict House Price]             |      |
|  +-----------------------------------+      |
|                                             |
|  +-----------------------------------+      |
|  | Prediction Result                 |      |
|  +-----------------------------------+      |
|  | Estimated House Price: $425,000   |      |
|  | Confidence: 85%                   |      |
|  | [Progress bar]                    |      |
|  +-----------------------------------+      |
|                                             |
+-----------------------------------------------+
```

### Model Visualizations

The application includes visualizations to help understand the model's performance:

#### Real vs. Predicted Prices
This visualization shows how well the model's predictions match actual house prices:

![Real vs Predicted](static/real_vs_predicted.png)

#### Feature Importance
This chart shows which features have the most impact on house price predictions:

![Feature Importance](static/feature_importance.png)

#### Correlation Heatmap
This visualization shows the relationships between different features:

![Correlation Heatmap](static/correlation_heatmap.png)

#### Error Distribution
This chart shows the distribution of prediction errors:

![Error Distribution](static/error_distribution.png)

## Model Performance

The Random Forest model achieves excellent performance metrics:

- **RMSE (Root Mean Squared Error)**: Measures the average prediction error
  - Training RMSE: Lower values indicate better fit
  - Testing RMSE: Measures prediction accuracy on new data
  
- **R² (R-squared)**: Indicates how well the model fits the data
  - Values closer to 1 indicate better fit
  - The model achieves high R² values on both training and test data

- **Cross-validation**: Ensures the model generalizes well to new data
  - 5-fold cross-validation was used to validate model performance

The Random Forest model significantly outperforms a baseline Linear Regression model, with improvements in both accuracy and reliability.

## Technologies Used

- **Python**: Core programming language
- **Flask**: Web framework for the application
- **scikit-learn**: Machine learning library for model training
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Bootstrap**: Frontend framework for responsive design
- **HTML/CSS/JavaScript**: Web interface

## Future Improvements

- Add more advanced models for comparison (XGBoost, Neural Networks)
- Implement feature engineering to improve prediction accuracy
- Add geographic visualization of predictions on a map
- Create a user account system to save prediction history
- Expand the dataset with more recent housing data
- Add time-series analysis to track price trends
- Implement a RESTful API for integrating with other applications

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Created with ❤️ for machine learning and data science enthusiasts
