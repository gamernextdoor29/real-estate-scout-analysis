import pandas as pd
import numpy as np

file_path_1 = r'C:\Users\Mujeeb Bello\Desktop\DATA ANALYSIS\Project 5\train.csv'
file_path_2 = r'C:\Users\Mujeeb Bello\Desktop\DATA ANALYSIS\Project 5\test.csv'
pd.set_option('display.max_columns', None)

train_df = pd.read_csv(file_path_1)
test_df = pd.read_csv(file_path_2)

# Numeric columns to fill with mean or median
columns_1 = ['LotFrontage']
for col in columns_1:
    skew_v = train_df[col].skew()
    if abs(skew_v)>0.5:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        test_df[col] = test_df[col].fillna(train_df[col].median())
    else:
        train_df[col] = train_df[col].fillna(train_df[col].mean())
        test_df[col] = test_df[col].fillna(train_df[col].mean())

# Numeric columns to fill with 0
columns_2 = ['MasVnrArea', 'GarageYrBlt', ]
for col2 in columns_2:
    train_df[col2] = train_df[col2].fillna(0)
    test_df[col2] = test_df[col2].fillna(0)

# String columns fill with None
columns_3 = ['Alley', 'MasVnrType','BsmtQual', 'BsmtCond',  'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
for col3 in columns_3:
    train_df[col3] = train_df[col3].fillna('none')
    test_df[col3] = train_df[col3].fillna('none')

# String columns to fill with mode
columns_4 = ['Electrical']
for col4 in columns_4:
    train_df[col4]=train_df[col4].fillna(train_df[col4].mode()[0])
    test_df[col4]=test_df[col4].fillna(train_df[col4].mode()[0])

# ONLY test.csv missing data
# fill with mode
columns_5 = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'Functional', 'SaleType']
for col5 in columns_5:
    test_df[col5] = test_df[col5].fillna(train_df[col5].mode()[0])
# fill with 0
columns_6 = [ 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtFullBath','BsmtUnfSF', 'BsmtHalfBath', 'KitchenQual',  'GarageCars', 'GarageArea']

for col6 in columns_6:
    test_df[col6] = test_df[col6].fillna(0)



train_df.to_csv(r'C:\Users\Mujeeb Bello\Desktop\DATA ANALYSIS\Project 5\train_2.csv')
test_df.to_csv(r'C:\Users\Mujeeb Bello\Desktop\DATA ANALYSIS\Project 5\test_2.csv')