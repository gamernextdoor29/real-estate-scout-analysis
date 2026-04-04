from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
file_path_1 = r'C:\Users\Mujeeb Bello\Desktop\DATA ANALYSIS\Project 5\train_2.csv'
df = pd.read_csv(file_path_1)

# 1. Split your data FIRST (80% train, 20% test)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = pd.get_dummies(X_train_raw)
X_test = pd.get_dummies(X_test_raw)

# "Align" them to make sure they have the same columns in the same order
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

y_train_log = np.log1p(y_train)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', RFECV(estimator=LinearRegression(), step=1, cv=5, scoring='r2')),
    ('model', LinearRegression())
])

pipe.fit(X_train, y_train_log)



predictions_log = pipe.predict(X_test)
final_predictions = np.expm1(predictions_log)

# TO SEE THE COLUMNS SELECTED AND COUNT
selector = pipe.named_steps['selector']
print(X_train.columns[selector.support_])
print(selector.n_features_)



#
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate the average dollar error
mae = mean_absolute_error(y_test, final_predictions)
rmse = np.sqrt(mean_squared_error(y_test, final_predictions))

print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=final_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Actual vs. Predicted House Prices')
plt.show()


# ploting histogram of residuals
residuals = y_test - final_predictions

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribution of Prediction Errors (Residuals)')
plt.xlabel('Error ($)')
plt.show()

#ploting a regplot
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=final_predictions)
plt.title('Residual Plot')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.show()

