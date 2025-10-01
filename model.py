import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
# Option 1: Load from CSV
data = pd.read_csv("data/housing.csv")

# Option 2: Load Boston dataset (Note: load_boston is deprecated)
# from sklearn.datasets import fetch_california_housing
# data = fetch_california_housing(as_frame=True)
# df = data.frame
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())  # check for nulls

# Check for outliers or distributions
sns.pairplot(data)
sns.heatmap(data.corr(), annot=True)
plt.show()
# Example: drop irrelevant columns
# data = data.drop(['id', 'date'], axis=1)

# Optional: Encode categorical features
# data = pd.get_dummies(data, drop_first=True)

# Feature & Target
X = data.drop("Price", axis=1)   # features
y = data["Price"]                # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Try Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Try Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Try XGBoost
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
def evaluate(y_true, y_pred, model_name):
    print(f"\n{model_name} Performance:")
    print("R2 Score:", r2_score(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))

evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest")
evaluate(y_test, y_pred_xgb, "XGBoost")
         