from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

# Set the style for plots
plt.style.use('fivethirtyeight')
sns.set_palette('viridis')

# Create a directory for visualizations if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# Get the California housing data
print("Loading California housing dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Display dataset information
print("\n=== Dataset Overview ===")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print("\nFeature names:", housing.feature_names)
print("\nFirst 5 houses:")
print(X.head())
print("\nTarget (house prices) statistics:")
print(f"Min: ${y.min()*100000:.2f}")
print(f"Max: ${y.max()*100000:.2f}")
print(f"Mean: ${y.mean()*100000:.2f}")
print(f"Median: ${np.median(y)*100000:.2f}")

# Check for missing values
print("\nMissing values:")
print(X.isnull().sum())

# Feature correlation analysis
print("\nCalculating feature correlations...")
correlation_matrix = pd.concat([X, pd.Series(y, name='PRICE')], axis=1).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig('static/correlation_heatmap.png')
plt.close()

# Standardize features
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"\nTraining data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")

# Train and evaluate Linear Regression model for comparison
print("\n=== Training Linear Regression Model (Baseline) ===")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_rmse = np.sqrt(lr_mse)
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

print(f"Linear Regression Results:")
print(f"RMSE (error): {lr_rmse:.3f}")
print(f"MAE: {lr_mae:.3f}")
print(f"R² (fit): {lr_r2:.3f}")

# Perform cross-validation on Linear Regression
lr_cv_scores = cross_val_score(lr_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
lr_cv_rmse = np.sqrt(-lr_cv_scores.mean())
print(f"Cross-validation RMSE: {lr_cv_rmse:.3f}")

# Train Random Forest model with hyperparameter tuning
print("\n=== Training Random Forest Model ===")
# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Use a smaller subset for GridSearchCV to save time
print("Performing hyperparameter tuning (this may take a while)...")
rf_base = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    verbose=1
)
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")
model = grid_search.best_estimator_

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nRandom Forest Results:")
print(f"RMSE (error): {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R² (fit): {r2:.3f}")

# Perform cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores.mean())
print(f"Cross-validation RMSE: {cv_rmse:.3f}")

# Compare models
print("\n=== Model Comparison ===")
print(f"Linear Regression RMSE: {lr_rmse:.3f}")
print(f"Random Forest RMSE: {rmse:.3f}")
print(f"Improvement: {((lr_rmse - rmse) / lr_rmse) * 100:.2f}%")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Create static directory if it doesn't exist
print("\nChecking for static directory...")
if not os.path.exists('static'):
    os.makedirs('static')
    print("Created static directory")
else:
    print("Static directory already exists")

# Plot feature importance
print("\nGenerating feature importance visualization...")
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Random Forest Model', fontsize=16)
plt.tight_layout()
plt.savefig('static/feature_importance.png')
print("Saved feature_importance.png to static directory")
plt.close()

# Plot real vs predicted prices
print("\nGenerating real vs predicted prices visualization...")
plt.figure(figsize=(12, 8))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Real Prices ($100,000s)", fontsize=14)
plt.ylabel("Predicted Prices ($100,000s)", fontsize=14)
plt.title("Real vs. Predicted House Prices", fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/real_vs_predicted.png')
print("Saved real_vs_predicted.png to static directory")
plt.close()

# Plot prediction error distribution
print("\nGenerating prediction error distribution visualization...")
errors = y_test - predictions
plt.figure(figsize=(12, 8))
sns.histplot(errors, kde=True)
plt.xlabel("Prediction Error ($100,000s)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Distribution of Prediction Errors", fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/error_distribution.png')
print("Saved error_distribution.png to static directory")
plt.close()

# Plot correlation heatmap
print("\nGenerating correlation heatmap visualization...")
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig('static/correlation_heatmap.png')
print("Saved correlation_heatmap.png to static directory")
plt.close()

# Save the model and scaler for later use
print("\nSaving model and scaler...")
joblib.dump(model, 'house_model.pkl')
print("Saved house_model.pkl")
joblib.dump(scaler, 'scaler.pkl')
print("Saved scaler.pkl")

# Verify files were created
print("\nVerifying files were created:")
for file in ['feature_importance.png', 'real_vs_predicted.png', 'error_distribution.png', 'correlation_heatmap.png']:
    if os.path.exists(f'static/{file}'):
        print(f"✓ static/{file} exists")
    else:
        print(f"✗ static/{file} does not exist")

print("\nModel training and evaluation complete!")
print("Visualizations saved in the 'static' directory.")
