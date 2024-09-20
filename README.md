# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
Step-1.Start the program

Step-2.prepare the Data 

step-3.Define about Hpothesis

step-4.Explain about Cost Function

step-5.Parameter Update Rule

step-6.Iterative Training

step-7.Model Evaluation

step-8.End the program

## Program:
```py
/*
Program to implement to predict the price of the house and number of occupants in the house with SGD regressor.
Developed by: K.Ligneshwar
RegisterNumber:  212223230113
*/
import pandas as pd
data=pd.read_csv("C:/Users/Admin/Desktop/Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])   
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
```
y_pred
```
![Screenshot 2024-09-06 140350](https://github.com/user-attachments/assets/f335547f-d1b7-43e1-843b-306293036c23)

```
print(classification_report1)
```
![Screenshot 2024-09-06 140430](https://github.com/user-attachments/assets/95a42dc0-5811-4f31-8437-eaa40f631eb5)

```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
![Screenshot 2024-09-06 140445](https://github.com/user-attachments/assets/db6dceb0-9f76-4c63-af96-93bd3b2b1129)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
