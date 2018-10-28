# Simple Linear Regression 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # 除了最后一行以外都是X
y = dataset.iloc[:, 1].values # 最后一行是y

# Splitting the dataset into the Training set and Test set
# 训练集占用2/3 测试集占用1/3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling 
# 为什么不需要特征缩放?
# 因为简单线性回归已经包含了特征缩放,并不需要额外的处理

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # 构造线性回归器
regressor.fit(X_train, y_train) # 使用线性回归器拟合X和y的训练集

# Predicting the Test set results
# 利用线性回归器进行预测数据,注意参数是测试集X,得到的是预测y
y_pred = regressor.predict(X_test)

# Visualising the Training set results
# 将训练集的X和y画成红色的点
plt.scatter(X_train, y_train, color = 'red') 
# 将训练集和X和根据训练集X预测的y,绘制城蓝色的线
plt.plot(X_train, regressor.predict(X_train), color = 'blue') 
plt.title('Salary VS Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
# 将用训练集训练出来的曲线和测试集的点进行对比
# 观察训练结果的差距
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()