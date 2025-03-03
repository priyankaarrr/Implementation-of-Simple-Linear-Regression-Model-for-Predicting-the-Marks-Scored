# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:

Developed by: priyanka R
RegisterNumber: 212223220081

~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
df.head()
~~~

## Output:
~~~
![image](https://github.com/user-attachments/assets/8d07ca80-4ce3-4e86-a33c-05290549f4e4)
~~~
df.tail()
## output
![image](https://github.com/user-attachments/assets/b3e46591-aa57-4d26-b9e9-c7b75c052188)

x=df.iloc[:,:-1].values
x
## output
![image](https://github.com/user-attachments/assets/6d33f3fc-f0cc-48a7-aa4c-2d6d95b82a4b)

y=df.iloc[:,1].values
y
## output
![image](https://github.com/user-attachments/assets/4aa7981a-f710-4833-89df-94e1bd7486e7)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

## output
![image](https://github.com/user-attachments/assets/950475ff-c621-4134-9b9b-5b9080a7392a)

y_test

## output

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE =',mae)
rmse=np.sqrt(mse)
print("RMSE =",rmse)













## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
