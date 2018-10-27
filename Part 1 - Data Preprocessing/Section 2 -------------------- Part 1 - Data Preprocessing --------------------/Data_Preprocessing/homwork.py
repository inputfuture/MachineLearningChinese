# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



# 引入数据处理的相关类库
import numpy as np #数学计算类库
import matplotlib.pyplot as plt #绘图库,用来绘制好看的模型
import pandas as pd #用来导入数据集



# 引入数据集
dataSet = pd.read_csv('Data.csv') # 读取csv文件
X = dataSet.iloc[:, :-1].values #包含自变量的矩阵X;读取所有的行,读取不包含最后一列的所有列
y = dataSet.iloc[:,3].values #应变量y,获取所有行,和最后一列,和下面这列是等效的
#y1 = dataSet.iloc[:,-1].values



# 处理缺失数据
from sklearn.preprocessing import Imputer
# 处理的缺失数据值是Nan,处理的方法是求平均值填入,最后表明作用的是列(也就是使用的是列的平均值)
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0) 
# 将初始化好的导入器适配X的列数据
imputer = imputer.fit(X[:, 1:3])
# 将适配后的数据作用在新数据上
X[:, 1:3] = imputer.transform(X[:, 1:3])



# 处理分类数据
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# 将第一列地名转换为数字表示,是为了转换为有意义的数值,在方程里面便于计算
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# 将第一列转换好的数字,转换为独热编码(OneHotEncoder),因为数字还是有大小顺序,并不是我们需要的
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
# 将是否购买了产品,转换为数字,并不需要独热编码//TODO:为什么?
labelencoder_y=LabelEncoder()
y = labelencoder_y.fit_transform(y)




# 将数据集划分为训练集和测试集
from sklearn.model_selection import train_test_split
# 将训练集(0.8)和测试集(0.2)划分,注意这个变量的先后顺序,不要赋值错误了
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 ,random_state = 0)





# 特征缩放:这一步的目的是将不同范围的数据,设置到一个范围之内,不然小的数据容易被忽略不计(非必须步骤)
from sklearn.preprocessing import StandardScaler
# 实例化特征缩放器
sc_X =StandardScaler()
# 通过训练数据初始化特征缩放器,特征缩放器就知道了如何转换
X_train = sc_X.fit_transform(X_train)
# 测试集的数据直接transfrom即可,不用再次fit
X_test = sc_X.transform(X_test)
