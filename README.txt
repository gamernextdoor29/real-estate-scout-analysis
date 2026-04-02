
## 📌 Project Overview
This project analyzes real estate data to uncover trends in pricing, location, and property features.

## 🎯 Objectives
- Predict property prices
- Identify key factors affecting value

## 🛠️ Tools & Technologies
- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learns

1.)looked for duplicate in the ID

print(df['Id'].duplicated().sum())


MAKE SURE YOU CLEAN BOTH THE TEST AND TRAINING DATA BUT AVOID DATA LEAKAGE INTO THE TEST DATAFRAME
-> Learn/calculate any imputation rules only from the training data
-> Then Apply those exact same rules to both train and test
2.)# in machine learning missing data tells a story so we can't just drop them, first lets find the missing data and plan how to fill them for both train and test data

missing_data=df.columns[df.isnull().any()]
print(missing_data)

# LETS CLEAN
STEP 1: seperate the numeric datatype column from the string column

STEP 2: if the variable is the type to have a value use either mean or madian to fill for numeric columns and mode for string columns.
 otherswise just put 0 or None, for variable that do not require a value like Pool_Size (missing value means no pool)

CHECK CODES.txt for thr codes 

3.) Checked for skewness of the target variable, SalePrice
print(df['SalePrice']) 
it was very skewed and so we are going to apply TARGET TRANSFORMATION
BECAUSE: The SalePrice was right-skewed, which can bias Linear Regression toward high-value outliers. I applied a Log Transformation to normalize the target distribution and stabilize the variance of the residuals, leading to a more robust mode. Having a skewed target will make the model struggle so will need to transform the price to LOG SCALE then transform it back after predictng.

TO APPLY THIS:
# Transform the price to make it more "Bell Shaped"
df['SalePrice_Log'] = np.log1p(df['SalePrice'])

# To get back to actual Dollars
actual_prediction = np.expm1(model_prediction)


4.) # ACTUAL TRAINING
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

file_path_train = r'C:\Users\Mujeeb Bello\Desktop\DATA ANALYSIS\Project 5\train_2.csv'
file_path_test = r'C:\Users\Mujeeb Bello\Desktop\DATA ANALYSIS\Project 5\test_2.csv'
pd.set_option('display.max_columns', None)

train_df = pd.read_csv(file_path_train)
test_df = pd.read_csv(file_path_test)

# Save the target variable from train
y_train = train_df['SalePrice']
# Identify the features (X) for both
X_train_raw = train_df.drop('SalePrice', axis=1)
X_test_raw = test_df.copy()

# Step: Aligning Dummy Variables
# This also converts covert categorical string columns into one-hot encoded (0/1) columns
# This ensures that both DFs get the same columns even if a category is missing in one
X_train = pd.get_dummies(X_train_raw)
X_test = pd.get_dummies(X_test_raw)

# "Align" them to make sure they have the same columns in the same order
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 1. Transform the target FIRST
y_train_log = np.log1p(y_train)

# 2. Define the Pipeline (same as before)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', RFECV(estimator=LinearRegression(), step=1, cv=5, scoring='r2')),
    ('model', LinearRegression())
])

# 3. FIT using the Logged target
pipe.fit(X_train, y_train_log)

# 4. PREDICT (Result will be in Log scale)
predictions_log = pipe.predict(X_test)

# 5. CONVERT BACK to Dollars for your report
final_predictions = np.expm1(predictions_log)

# Create a clean DataFrame for result




