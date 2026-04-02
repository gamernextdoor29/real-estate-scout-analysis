THIS FOR TESTING ACCURACY OF THE MODE

Because we do not have SalePrice in our test.csv so we do not have y_test
the plan now is to split train.csv into test and training data and get our y_test from it 
1.)
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


2.) NOW LETS TEST ACCURACY AND SEE THE RESULT