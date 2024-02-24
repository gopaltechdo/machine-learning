import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt  
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
 
#importing the datashet

#note The dependent variable is the one being trained on,   
# whereas the independent variables are those being used to train the model.
dataset= pd.read_csv('Data.csv')  #it will create all data frame in this csv file using pandas
x= dataset.iloc[ :, :-1].values #iloc is used for locate indexes,  :-1 means expert last column 
y= dataset.iloc[ :, -1].values #iloc is used for locate indexes,  -1 means  last column this is depant viable vector

print(x) 
print(y)

#taking care of missing data
#The imputer is an estimator used to fill the missing values in datasets.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1: 3])

x[:, 1:3] =imputer.transform(x[:, 1:3])   

print(x)
