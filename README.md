Cricket Prediction Model

Data PreProcessing
This script is designed to process and clean cricket batting data obtained from a CSV file (batting_card.csv). The code uses the pandas library to manipulate and clean the dataset for further analysis. Below is an overview of the code and its functionalities:
1. Importing Libraries<a name="importing-libraries"></a>
The script starts by importing necessary libraries such as pandas, NLTK (Natural Language Toolkit), and regular expression.
2. Loading Dataset<a name="loading-dataset"></a>
The dataset (batting_card.csv) is loaded into a pandas DataFrame (pddf). Information about the dataset, including its columns, non-null counts, and data types, is displayed using the info() method.
3. Data Cleaning and Preprocessing<a name="data-cleaning-and-preprocessing"></a>
The script defines functions to clean and preprocess text data. It removes unwanted characters, converts text to lowercase, and eliminates stopwords. Additionally, it handles columns like 'Summary' and 'Review' by capitalizing the strings.
4. Handling Missing Values<a name="handling-missing-values"></a>
The script addresses missing values by dropping rows with null values and duplicates. It also handles specific columns like 'product_price' and 'Rate' by converting them to numeric types and removing corresponding rows with missing values.
5. Numeric Conversion and Data Transformation<a name="numeric-conversion-and-data-transformation"></a>
The script converts certain columns to numeric types and rounds off decimal values. It further processes the 'Review' and 'Summary' columns by removing stopwords.
6. Saving the Processed Data<a name="saving-the-processed-data"></a>
The script saves the processed DataFrame (pdDF) to a new CSV file named 'NewCricket.csv' using the to_csv() method.
Note:
There is a warning about setting a value on a copy of a slice from a DataFrame. It is advisable to use .loc[row_indexer, col_indexer] = value to avoid potential issues. Users should review and modify the code as needed for specific use cases. Additionally, the NLTK library should be installed before running the script by executing nltk.download('stopwords').
 
Module
Overview
This repository contains code for a cricket prediction model that predicts the likelihood of a player getting out ('isNotOut') based on various features. The model is implemented in Python and utilizes popular libraries such as NumPy, pandas, and scikit-learn. The prediction model is stored using pickle files, and a custom scaler class is implemented to preprocess the data before making predictions.

Dependencies
Make sure you have the following dependencies installed:
•	NumPy
•	pandas
•	scikit-learn
You can install them using the following command:
pip install numPy pandas scikit-learn

Custom Scaler Class
The ‘CustomScaler’ class is a custom implementation of the scikit-learn ‘BaseEstimator’ and ‘TransformerMixin’ classes. It is designed to scale specified columns of a DataFrame using the ‘StandardScaler’. This class is useful for ensuring consistent scaling of features during both training and prediction.
Model Cricket Class
The ‘model_cricket’ class is the main component that loads a pre-trained prediction model and scaler. It provides methods to load and clean new data, obtain predicted probabilities, predicted output categories, and overall predicted outputs.

Methods
‘__init__(self, model_file, scaler_file)’: Initializes the cricket model by loading the pre-trained model and scaler from the provided files.
‘load_and_clean_data(self, data_file)’: Loads data from a CSV file, removes unnecessary columns ('Runnings', 'isNotOut'), and scales the data using the loaded scaler.
‘predicted_probability(self)’: Returns the predicted probabilities of the players getting out. Ensure that data has been loaded using load_and_clean_data before calling this method.
‘predicted_output_category(self)’: Returns the predicted output categories (e.g., 0 for 'Not Out', 1 for 'Out'). Ensure that data has been loaded using load_and_clean_data before calling this method.
‘predicted_outputs(self)’: Returns a DataFrame with original features, predicted probabilities, and predicted output categories. Ensure that data has been loaded using load_and_clean_data before calling this method.
 
SQL Integration

Overview
This code demonstrates the integration of a cricket prediction model with a MySQL database. The prediction model is implemented in Python and is loaded from a module named module_cricket. The steps include loading a dataset, calling functions to assign the model and scaler as parameters, making predictions, and inserting the results into a MySQL database.

Dependencies
Ensure you have the required dependencies installed:
•	pandas
•	scikit-learn
•	pymysql
You can install them using the following command:
pip install pandas scikit-learn pymysql

Usage

1.	Import the Module Contents:
from module_cricket import *

2.	Load the Dataset:
import pandas as pd
dataset = pd.read_csv('NewCricket.csv')

3.	Call a Function and Assign the Model and Scaler as Parameters:
model = model_cricket('model', 'scaler')
model.load_and_clean_data('NewCricket.csv')

4.	Predict Outputs:
pddf_new_obs = model.predicted_outputs()

5.	Import pymysql After Installing It:
import pymysql

6.	Create a Connection and Cursor, then Execute:
conn = pymysql.connect(database='predicted_outputs', user='root', password='****')
cursor = conn.cursor()
cursor.execute('SELECT * FROM predicted_outputs;')

7.	Insert Data into SQL Database:
insert_query = 'INSERT INTO predicted_outputs VALUES'

for i in range(pddf_new_obs.shape[0]):
    insert_query += '('

    for j in range(pddf_new_obs.shape[1]):
        value = pddf_new_obs.iloc[i, j]
        if isinstance(value, (int, float)):
            insert_query += str(value) + ', '
        else:
            insert_query += f'{value}, '

    insert_query = insert_query[:-2] + '), '

insert_query = insert_query[:-2] + ';'

8.	Load the Data into SQL Workbench:
cursor.execute(insert_query)
conn.commit()
conn.close()

Note
•	The warnings regarding unpickling estimators from different versions of scikit-learn are displayed during model loading. This may lead to breaking code or invalid results, so use at your own risk.
•	Customize the code paths, database credentials, and other parameters as needed for your specific setup.

 
Training

Overview
This code implements a logistic regression model for classifying cricket performance based on various input features. The model is trained on a dataset loaded from 'NewCricket.csv' and includes steps for preprocessing, standardization, training, exploring coefficients, testing, and saving the trained model and scaler.

Usage

1.	Load and Preprocess the Dataset:
import pandas as pd
import numpy as np
pddf = pd.read_csv('NewCricket.csv')
pdDF = pddf.copy()
pdDF = pdDF.drop(['Runnings', 'isNotOut'], axis=1)
# Creating target variable
targets = np.where(pdDF['Runnings'] > pdDF['Runnings'].median(), 1, 0)
pdDF['RunPoints'] = targets
# Selecting inputs
unscaled_inputs = pdDF.iloc[:, :-1].dropna()

2.	Standardize the Data:
from sklearn.preprocessing import StandardScaler
class CustomScaler:
    # ... (as provided in your code)
columns_to_omit = ['runs', 'ballsFaced', 'fours', 'sixes', 'strikeRate', 'Wickets']
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]
running_scaler = CustomScaler(columns_to_scale)
running_scaler.fit(unscaled_inputs)
scaled_inputs = running_scaler.transform(unscaled_inputs)

3.	Train the Model:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size=0.8, random_state=20)
reg = LogisticRegression()
reg.fit(x_train, y_train)

4.	Explore Coefficients:
reg.score(x_train, y_train)
reg.intercept_
reg.coef_
# Creating a summary table
summary_table = pd.DataFrame(columns=['feature_name'], data=unscaled_inputs.columns.values)
summary_table['Coefficient'] = np.transpose(reg.coef_)
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table['Odds_ratios'] = np.exp(summary_table.Coefficient)

5.	Testing:
reg.score(x_test, y_test)
predict_probability = reg.predict_proba(x_test)

6.	Save the Model:
import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(reg, file)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(running_scaler, file)

Note
•	Ensure that you have the required dependencies installed: pandas, numpy, and scikit-learn.
•	Customize the file paths and names as needed.
•	The saved model and scaler can be loaded later for making predictions on new data.
