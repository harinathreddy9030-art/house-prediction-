# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load Dataset
housing = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target


# Show Dataset
print("First 5 Rows")
print(df.head())


# Dataset Information
print("\nDataset Information")
print(df.info())


# Statistical Summary
print("\nStatistical Summary")
print(df.describe())


# Check Missing Values
print("\nMissing Values")
print(df.isnull().sum())


# -----------------------------
# Exploratory Data Analysis
# -----------------------------

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# Price Distribution
plt.figure(figsize=(6,4))
sns.histplot(df["Price"], bins=30, kde=True)
plt.title("House Price Distribution")
plt.show()


# Scatter Plot
plt.figure(figsize=(6,4))
sns.scatterplot(x=df["AveRooms"], y=df["Price"])
plt.title("Average Rooms vs Price")
plt.show()


# -----------------------------
# Machine Learning Model
# -----------------------------

# Features and Target
X = df.drop("Price", axis=1)
y = df["Price"]


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train Model
model = LinearRegression()

model.fit(X_train, y_train)


# Predictions
predictions = model.predict(X_test)


# -----------------------------
# Model Evaluation
# -----------------------------

rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("RMSE:", rmse)
print("R2 Score:", r2)


# -----------------------------
# Visualization
# -----------------------------

# Actual vs Predicted
plt.figure(figsize=(6,4))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()


# Feature Importance
importance = pd.Series(model.coef_, index=X.columns)

importance.sort_values().plot(kind="barh")

plt.title("Feature Importance")
plt.show()