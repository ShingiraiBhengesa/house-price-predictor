from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Get the house data
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Look at it
print("First 5 houses:")
print(X.head())
print("\nWhat’s in the data:")
print(X.info())
print("\nFirst 5 prices:")
print(y[:5])

# Check for missing stuff
print("\nMissing values:")
print(X.isnull().sum())  # Should all be 0

# Make numbers “normal” (not too big or small)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split data: some to learn, some to test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)

