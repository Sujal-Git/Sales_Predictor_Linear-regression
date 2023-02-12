#Importing Libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df=pd.read_csv('advertising.csv') #initializing Features and target
x=df.iloc[:,0:2]
y=df['Sales']

#Feeding data to Model
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)
Model=linear_model.LinearRegression()
Model.fit(x_train.values,y_train.values)

#Predicting The sales Using model
Predicted=Model.predict([[230.1,37.8]])
print(f"Predicted Sales: {Predicted}")







