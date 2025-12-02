import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

'''
拟合的三种状态：1.欠拟合、2.良好拟合、3.过拟合
'''

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def Underfitting():
    '''
    欠拟合展示：模型在训练数据和测试数据上的性能表现都较差。
    '''
    # 准备数据(模拟数据增加噪声)
    np.random.seed(666)

    # 特征生成：random.uniform均匀分布：在[-3,3]之间均匀分布
    x_data=np.random.uniform(-3,3,size=100)
    #print("x_data:", x_data)
    
    # 目标函数：y=0.5 * x^2 + x + 2 + 噪声 :random.normal表示正态分布在均值0,方差1的分布的100个样本
    y_data=0.5 * x_data**2 + x_data + 2 + np.random.normal(0,1,size=100)
    #print("y_data:", y_data)

    # 转换为一维数组
    X=x_data.reshape(-1,1)

    # 执行线性回归,训练模型
    model=LinearRegression()
    model.fit(X, y_data)

    # 预测
    y_pred = model.predict(X)

    # 评估模型,MSE计算均方误差
    mse = mean_squared_error(y_data, y_pred)
    print("均方误差 (MSE):", mse)

    # 可视化展示
    plt.scatter(x_data, y_data, label='欠拟合')
    plt.plot(x_data, y_pred, color='green', label='线性回归预测')
    plt.legend()
    plt.show()


def Well_fitting():
    '''
    良好拟合展示：模型在训练数据和测试数据上的性能表现都较好。
    '''
   # 准备数据(模拟数据增加噪声)
    np.random.seed(666)

    # 特征生成：random.uniform均匀分布：在[-3,3]之间均匀分布
    x_data=np.random.uniform(-3,3,size=100)
    #print("x_data:", x_data)
    
    # 目标函数：y=0.5 * x^2 + x + 2 + 噪声 :random.normal表示正态分布在均值0,方差1的分布的100个样本
    y_data=0.5 * x_data**2 + x_data + 2 + np.random.normal(0,1,size=100)
    #print("y_data:", y_data)

    # 转换为一维数组
    X=x_data.reshape(-1,1)
    X2=np.hstack([X,X**2])

    # 执行线性回归,训练模型
    model=LinearRegression()
    model.fit(X2, y_data)

    # 预测
    y_pred = model.predict(X2)

    # 评估模型,MSE计算均方误差
    mse = mean_squared_error(y_data, y_pred)
    print("均方误差 (MSE):", mse)

    # 可视化展示
    plt.scatter(x_data, y_data, label='良好拟合')
    plt.plot(np.sort(x_data), y_pred[np.argsort(x_data)], color='green', label='线性回归预测')
    plt.show()

def Overfitting():
    '''
    过拟合展示：模型在训练数据上的性能表现很好，但在测试数据上的性能表现较差。
    '''
   # 准备数据(模拟数据增加噪声)
    np.random.seed(666)

    # 特征生成：random.uniform均匀分布：在[-3,3]之间均匀分布
    x_data=np.random.uniform(-3,3,size=100)
    #print("x_data:", x_data)
    
    # 目标函数：y=0.5 * x^2 + x + 2 + 噪声 :random.normal表示正态分布在均值0,方差1的分布的100个样本
    y_data=0.5 * x_data**2 + x_data + 2 + np.random.normal(0,1,size=100)
    #print("y_data:", y_data)

    # 转换为一维数组
    X=x_data.reshape(-1,1)

    # 增加高次项特征，制造过拟合
    X2=np.hstack([X,X**2,X**3,X**4,X**5,X**6,X**7,X**8,X**9,X**10])

    # 执行线性回归,训练模型
    model=LinearRegression()
    model.fit(X2, y_data)

    # 预测
    y_pred = model.predict(X2)

    # 评估模型,MSE计算均方误差
    mse = mean_squared_error(y_data, y_pred)
    print("均方误差 (MSE):", mse)

    # 可视化展示
    plt.scatter(x_data, y_data, label='过拟合')
    plt.plot(np.sort(x_data), y_pred[np.argsort(x_data)], color='green', label='线性回归预测')
    plt.show()

if __name__=='__main__':
    # 欠拟合展示
    Underfitting()

    # 良好拟合展示
    Well_fitting()
    
    # 过拟合展示
    Overfitting()
