from sklearn.datasets import fetch_california_housing
import pandas as pd

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
