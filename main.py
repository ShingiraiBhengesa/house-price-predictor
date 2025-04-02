from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target  # Median house value in $100,000s

# Explore data
print(X.head())
print(X.info())
print(f"Target sample: {y[:5]}")

# Check for missing values
print(X.isnull().sum())  # Should be 0

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split into train/test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluate
print("Linear Regression:")
print(f"RMSE: {mean_squared_error(y_test, lr_pred, squared=False):.3f}")
print(f"R²: {r2_score(y_test, lr_pred):.3f}")

print("Random Forest:")
print(f"RMSE: {mean_squared_error(y_test, rf_pred, squared=False):.3f}")
print(f"R²: {r2_score(y_test, rf_pred):.3f}")