# real-estate-scout-analysis
yeah, as the name imply, predicted house prices with 95% accuracy

📌 Project Overview

This project focuses on analyzing and modeling real estate data to understand the key factors that influence house prices and to build a predictive model.

It is structured into two main parts:

Data Cleaning & Preprocessing
Machine Learning Model for Price Prediction
🎯 Objectives
Build a reliable model to predict house prices
Identify important features that influence property value
Handle missing data intelligently without losing valuable information
Apply feature engineering and transformations to improve model performance

🛠️ Tools & Technologies
Python
Pandas
NumPy
Matplotlib & Seaborn
Scikit-learn

📂 Project Structure
project/
src/
data_cleaning.py   # Handles preprocessing and missing values
codes.py           # Model training, evaluation, and visualization

data/
train.csv
test.csv
train_2.csv        # Cleaned training data
test_2.csv         # Cleaned test data

output/
scatterplot
residualplot
histplot

README.md

🧹 Data Cleaning Process (data_cleaning.py)
1. Duplicate Check
Checked for duplicate IDs to ensure data integrity:
df['Id'].duplicated().sum()
2. Handling Missing Values

⚠️ Important Rule:
All imputation strategies are learned from training data only to avoid data leakage, and then applied to both train and test datasets.

Approach:
Numeric Features
Use median if skewed
Use mean if normally distributed
Categorical Features
Use mode for common categories
Use 'none' where absence has meaning (e.g., no pool, no garage)
Special Cases
Some features filled with 0 where missing implies non-existence

3. Feature-Specific Strategies
LotFrontage → mean/median based on skewness
MasVnrArea, GarageYrBlt → filled with 0
Basement, garage, pool-related features → filled with 'none'
Electrical → filled with mode
Test-only missing columns handled separately

4. Output
Cleaned datasets are saved as:
train_2.csv
test_2.csv



🤖 Modeling Process (codes.py)
1. Train-Test Split
Since the original test set has no target (SalePrice), we:
Split the training data into:
80% training
20% validation (test)
2. Feature Engineering
Applied one-hot encoding using pd.get_dummies()
Ensured consistent columns using:
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
3. Target Transformation

The target variable (SalePrice) was right-skewed, which can hurt model performance.

✅ Solution: Log Transformation

y_train_log = np.log1p(y_train)

➡️ This:

Normalizes distribution
Reduces impact of outliers
Improves model stability

🔁 Reverse transformation:

final_predictions = np.expm1(predictions_log)
4. Machine Learning Pipeline

A full pipeline was built using Scikit-learn:

Pipeline([
    ('scaler', StandardScaler()),
    ('selector', RFECV(LinearRegression(), cv=5)),
    ('model', LinearRegression())
])
Components:
StandardScaler → Feature scaling
RFECV → Automatic feature selection
LinearRegression → Final prediction model
5. Model Evaluation

Metrics used:

Mean Absolute Error (MAE) → Average prediction error in dollars
Root Mean Squared Error (RMSE) → Penalizes large errors
mae = mean_absolute_error(y_test, final_predictions)
rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
📊 Visualizations

The project includes several visual diagnosticss stored in the output folder:

✔️ Actual vs Predicted Prices
Helps evaluate prediction accuracy
✔️ Residual Distribution
Checks if errors are normally distributed
✔️ Regression Plot
Shows relationship between predictions and actual values

🚀 Key Insights
Missing data can contain valuable signals and should not be blindly removed
Log transformation significantly improves model performance on skewed data
Feature selection helps reduce noise and improve generalization
Proper handling of train/test separation is critical to avoid data leakage

▶️ How to Run

Clone the repository

git clone https://github.com/gamernextdoor29/real-estate-scout-analysis

Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn scipy
Run data cleaning
python src/data_cleaning.py
Run model training
python src/codes.py

📌 Future Improvements
Try advanced models (Random Forest, XGBoost)
Perform hyperparameter tuning
Add cross-validation on full pipeline
Deploy model as a web app (Flask/Streamlit)

👤 Author
Mueeb Bello

Mujeeb Bello
