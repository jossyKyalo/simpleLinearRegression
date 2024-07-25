import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


data_set=pd.read_csv('Salary_Data.csv')
x= data_set.iloc[:,:-1].values  
print(x)
print("\n")
y= data_set.iloc[:, 1].values   
print(y)
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0)  

#fitting the Simple Linear Regression model training
regressor=LinearRegression()
regressor.fit(x_train, y_train)
print("\n")
print(f"LinearRegression(copy_X={regressor.copy_X}, fit_intercept={regressor.fit_intercept}, "
      f"n_jobs={regressor.n_jobs}, normalize={regressor.normalize})")

#prediction of Test and Training set result

y_pred=regressor.predict(x_test)
x_pred=regressor.predict(x_train)
#visualizing the training set results
mtp.scatter(x_train, y_train, color="green")   
mtp.plot(x_train, x_pred, color="red")    
mtp.title("Salary vs Experience (Training Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show()   
#visualizing the test set results
print("\n")
mtp.scatter(x_test, y_test, color="blue")   
mtp.plot(x_train, x_pred, color="red")    
mtp.title("Salary vs Experience (Test Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show()  