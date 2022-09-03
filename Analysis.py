import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt


def load_data_all_variables(predicted_column):
    dataset = pd.read_csv("train.csv", usecols=["MSSubClass", 'MSZoning', 'LotFrontage', 'SalePrice'])
    print(dataset)

    numerical_cols = [col for col in dataset.columns if dataset[col].dtypes != 'object']
    categorical_cols = [col for col in dataset.columns if dataset[col].dtypes == 'object']

    numerical_na_cols = []
    categorical_na_cols = []

    for k, v in dataset[numerical_cols].isnull().sum().to_dict().items():
        if v != 0:
            numerical_na_cols.append(k)

    for k, v in dataset[categorical_cols].isnull().sum().to_dict().items():
        if v != 0:
            categorical_na_cols.append(k)

    print(numerical_na_cols)
    print(categorical_na_cols)

    # Imputing numerical columns with missing data

    my_imputer = SimpleImputer(strategy='constant', fill_value=0)

    train_imputed_values = my_imputer.fit_transform(dataset[numerical_na_cols])
    dataset[numerical_na_cols] = train_imputed_values

    # Dropping the first column from status dataset
    status = pd.get_dummies(dataset['MSZoning'], drop_first=True)

    # Adding the status to the original housing dataframe
    dataset = pd.concat([dataset, status], axis=1)

    # Dropping 'furnishingstatus' as we have created the dummies for it
    dataset.drop(['MSZoning'], axis=1, inplace=True)

    print(dataset)

    # Splitting the data into training and testing
    x = dataset.drop([predicted_column], axis=1)
    y = dataset[predicted_column]

    return x, y

load_data_all_variables('SalePrice')

def mlr(x, y):
    # De trainings- en validatiepartities. 20% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

    # Linear Regression
    linear_regression = LinearRegression(normalize=True)
    linear_regression.fit(x_train, y_train)
    predictions_linear_regression = linear_regression.predict(x_validation)

    print("Multi linear regression: ")
    print(r2_score(y_validation, predictions_linear_regression))
    print("Mean absolute error: " + str(mean_absolute_error(y_validation, predictions_linear_regression)))
    # Return RMSE between logarithm of the actual and predicted values
    print("Mean squared error: " + str(mean_squared_error(y_validation, predictions_linear_regression)))
    print("Root mean squared error: " + str(sqrt(mean_squared_error(y_validation, predictions_linear_regression))))



X, Y = load_data_all_variables("SalePrice")
mlr(X, Y)
