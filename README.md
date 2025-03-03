# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Store data in a structured format (e.g., CSV, DataFrame).
2.Use a Simple Linear Regression model to fit the training data.
3.Use the trained model to predict values for the test set.
4. Evaluate performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

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
~~~
x=df.iloc[:,:-1].values
x
~~~
## output
![image](https://github.com/user-attachments/assets/6d33f3fc-f0cc-48a7-aa4c-2d6d95b82a4b)
~~~
y=df.iloc[:,1].values
y
~~~
## output
![image](https://github.com/user-attachments/assets/4aa7981a-f710-4833-89df-94e1bd7486e7)
~~~
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
~~~

~~~
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
~~~

y_pred

## output
![image](https://github.com/user-attachments/assets/950475ff-c621-4134-9b9b-5b9080a7392a)

y_test

## output

![image](https://github.com/user-attachments/assets/ddcdfdd5-1da0-4a73-8043-0e3e612f7edd)
~~~
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE =',mae)
rmse=np.sqrt(mse)
print("RMSE =",rmse)
~~~
## output
![image](https://github.com/user-attachments/assets/6c5dd9f0-c9ec-4ac0-9b2c-6545f9253e6a)
~~~

plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
~~~
## output
~~~

plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,y_pred,color="red")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
~~~
## output
![image](https://github.com/user-attachments/assets/f6b4bed0-9c98-46da-be29-8dfef5532431)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
