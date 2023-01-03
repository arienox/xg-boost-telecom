# xg-boost-telecom

This code performs machine learning classification on telecom customer data to predict whether they will churn or not. The data is first loaded into a Pandas dataframe and some basic information such as the number of null values and data types is extracted. The data is then split into input and output variables, with the output variable being the target variable (whether the customer will churn or not). Dummy variables are created for the categorical input variables. The data is then split into training and testing sets, and an XGBoost classifier model is trained on the training data. The model is then used to make predictions on the testing data, and the performance of the model is evaluated using a confusion matrix, classification report, and accuracy score. The code also calculates the true negatives, false positives, false negatives, and true positives from the confusion matrix.

## Steps:
```
Set the working directory and load the telecom customer data into a Pandas dataframe.
Print the first 10 rows of the dataframe and check for null values and data types.
Split the data into input and output variables.
Create dummy variables for the categorical input variables.
Split the data into training and testing sets.
Train an XGBoost classifier model on the training data.
Use the trained model to make predictions on the testing data.
Evaluate the performance of the model using a confusion matrix, classification report, and accuracy score.
Calculate the true negatives, false positives, false negatives, and true positives from the confusion matrix.
```
