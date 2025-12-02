import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

'''
    正则化是一种用于解决过拟合的技术，通过在损失函数中添加一个正则项来惩罚模型的复杂度。
    正则化项通常是模型参数的范数，如L1范数（Lasso回归）或L2范数（Ridge回归）。
    正则化可以将高次项系数限制在一个较小的范围内，减少过拟合。

    1.L1正则化(Lasso回归)：损失函数 = 原始损失函数 + λ * Σ|wi| ,高次项系数变为0    
    2.L1正则化：适用于特征选择，将一些特征的系数设为0，实现特征的自动选择。
'''

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def lasso_regression():
    '''
    通过Lasso回归降低过拟合的效果
    '''
    # 准备数据(模拟数据增加噪声)
    np.random.seed(666)

    # 特征生成：random.uniform均匀分布：在[-3,3]之间均匀分布
    x_data=np.random.uniform(-3,3,size=100)
    #print("x_data:", x_data)
    
    # 目标函数：y=0.5 * x^2 + x + 2 + 噪声 :random.normal表示正态分布在均值0,方差1的分布的100个样本
    y_data=0.5 * x_data**2 + x_data + 2 + np.random.normal(0,1,size=100)

    # 转换为一维数组
    X=x_data.reshape(-1,1)

    # 增加高次项特征,为了增加模型的复杂度,制造过拟合
    X2=np.hstack([X,X**2,X**3,X**4,X**5,X**6,X**7,X**8,X**9,X**10])

    # 执行Lasso回归,训练模型,alpha惩罚系数为0.005,alpha惩罚系数越小,高次项系数越接近0
    lasso = Lasso(alpha=0.005)
    lasso.fit(X2, y_data)
    
    # 预测
    y_pred = lasso.predict(X2)
    print("Lasso回归系数:", lasso.coef_)

    # 评估模型,计算均方误差
    mse = mean_squared_error(y_data, y_pred)

    print("lasso回归模型的均方误差(MSE):", mse)

    # 可视化展示
    plt.scatter(x_data, y_data, label='Lasso降低过拟合')
    plt.plot(np.sort(x_data), y_pred[np.argsort(x_data)], color='blue', label='Lasso回归预测')
    plt.legend()
    plt.show()
    
    return mse


if __name__=='__main__':       
    # 执行Lasso回归
    lasso_regression()
