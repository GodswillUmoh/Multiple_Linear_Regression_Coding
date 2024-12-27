# Multiple_Linear_Regression_with_Python
## Question:
You are hired as a data scientist to evaluate the correlation among the features used by 50 startups company. Create a model to determine whether the expenditure on the variables amounted to profit. The table contains the dataset for the 50 startups
> In this scenario, we use MLR in a machine learning context to evaluate and predict profitability for 50 startup companies based on their expenditures on key variables. The dataset provides insights into whether spending patterns are correlated with profitability.

# Building_Multiple_Linear_Regression_Model
> Note: The Dataset used here was got from SuperDataScience dataset on Machine Learning.
> The First Seven records of the Sample Dataset is displaced below for insight into the data

## The Dataset: It contains 50 Startups, displaying their expenditure in different areas as seen in the table.
|R&D Spend|	Administration|	Marketing Spend|	State	|Profit|
|----------|---------------|----------------|-------|-------|
|165349.2|	136897.8|	471784.1|	New York|	192261.83|
|162597.7	|151377.59	|443898.53	|California	|191792.06|
|153441.51|	101145.55|	407934.54|	Florida|	191050.39|
|144372.41	|118671.85	|383199.62	|New York	|182901.99|
|142107.34|	91391.77|	366168.42|	Florida|	166187.94|
|131876.9	|99814.71	|362861.36	|New York	|156991.12|

## Note
> We do not need to apply feature scaling here because the coefficient will compensate to put everything on the same scale. hence, we need not apply feature scaling. In multiple linear regression, there's no need apply feature scaling

## Do you have to check the assumptions of multiple linear regression on the dataset all the time?
> There is no need to do that. Go ahead and use the multiple linear regression on the dataset, if it perform poorly, it means there was no correlation, you discard and try another model. If your dataset has linear relationship, it will give high performance.

## Questions often asked in Multiple Linear Regression model building

> ### In MLR, do we have to do something to avoid the dummy variable trap?
> The answer is no because the model class takes care of that with its inbuilt function

> ### Do we have to apply backward elimination or the like to select the best predictors?
> The answer is No because the class of the MLR has inbuilt function to select the best of the P-values.

## Importing the libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Importing the dataset
```python
dataset = pd.read_csv('startups_fifty.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)
```

## Encoding categorical data
```python

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
ct.fit_transform(X)

#convert to array
X = np.array(ct.fit_transform(X))

print(X)

```

## Splitting the dataset into the Training set and Test set
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

```

## Training the Multiple Linear Regression model on the Training set
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
```
## Note:
> Unlike the simple, you cannot visualize all multiple predictors in the graph. Two vectors can be visualized
>E.g vectors of the ten real profit and vectors of ten predicted profit of the test set to see if the prediction is close to the real values.

## Predicting the Test set results
```python
y_pred = lr.predict(X_test)
# To display two decimal numeric values
np.set_printoptions(precision=2)
#reshape helps display it vertically instead of the default horizontal display.
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_pred.reshape(len(y_pred), 1) ), 1))

```

[To view python code result in the terminal, Click Here](https://colab.research.google.com/drive/1r2h0lr7V37XiVTXk7GAs05-8OHr8G6tW#scrollTo=xNkXL1YQBiBT)
