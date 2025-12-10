
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

'''
1.找到最优模型->最小损失函数->梯度下降法，当特征数量较多，模型复杂时使用

2.梯度下降（Gradient Descent）：
    最小值公式：w = w - α * (X^T * (X * w - y))
    步长：α 是学习率，控制每次迭代的步长。
    迭代次数：需要根据实际情况设置迭代次数，确保模型收敛。
    用途：迭代优化损失函数，找到模型参数的最优解，完成收敛。适合特征数量较多的情况。
    注意：需要设置学习率 α 和迭代次数。  

3.梯度下降算法的种类
        Gradient Descent（梯度下降）：GD
    1.全梯度下降（Full Gradient Descent）：FGD
        每次迭代使用所有样本计算梯度，更新参数。
        适用场景：样本数量较少时。

    2.随机梯度下降（Stochastic Gradient Descent）：SGD
        每次迭代随机选择一个样本计算梯度，更新参数。
        适用场景：样本数量较大时。

    3.小批量梯度下降（Mini-batch Gradient Descent）：MBGD
        每次迭代随机选择一小批量样本计算梯度，batch批次,batch_size每批次的大小
        适用场景：样本数量中等时。

    4.平均随机梯度下降（Stochastic Average Gradient Descent）：SAG
        每次迭代随机选择一个样本的梯度值和以往的梯度值的平均值，更新参数。
        适用场景：样本数量中等时。
'''

def download_boston_data(curr_path):

    url=f"https://lib.stat.cmu.edu/datasets/boston"
    df=pd.read_csv(url,sep="\s+",skiprows=22,header=None)

    #给以下代码切片增加注释，说明切片的作用
    #df.values[::2, :]  表示取出所有偶数行的所有列
    #df.values[1::2, :2]表示取出所有奇数行的前2列，即0，1列

    data=np.hstack([df.values[::2, :], df.values[1::2, :2]])

    #给target增加注释，说明target的作用
    #target表示所有奇数行的第3列，df.values[1::2, 2]表示取出所有奇数行的第3列 ，即目标变量
    target=df.values[1::2, 2]

    #将data与target合并，并保存存到当前目录的data目录下
    df=pd.DataFrame(np.hstack([data, target.reshape(-1, 1)]))

    df.to_csv(f"{curr_path}",index=False,header=None)

    return 

def train_model():
    #读取波士顿数据集
    df=pd.read_csv(curr_path,header=None)
    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]

    # 划分数据集
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=22)

    # 特征标准化，对训练集和测试集进行标准化
    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test) 

    # 随机梯度下降法，训练模型，损失函数：均方误差（squared_loss），学习率是常数learning_rate="constant"：eta0=0.01，迭代次数max_iter=1000 
    # (动态可调整学习率learning_rate="inscaling",eta=eta0/pow(t,power_t=0.25),根据时间t动态调整学习率，默认power_t=0.25)
    model=SGDRegressor(loss="squared_loss",max_iter=1000,learning_rate="constant",eta0=0.01,random_state=22)
    model.fit(x_train,y_train)

    # 模型预测
    y_pred=model.predict(x_test)
    print(f"模型预测结果前5个: {y_pred[:5]}")
    print(f"模型回归系数: {model.coef_}")
    print(f"模型偏置值: {model.intercept_}")

    # 模型评估，MSE和R^2
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    
    print(f"均方误差(MSE): {mse:.2f}")
    print(f"R^2 系数: {r2:.2f}")

if __name__=="__main__":
    #获取当前文件的绝对路径
    abs_path=os.path.abspath(os.path.dirname(__file__))
    curr_path=os.path.join(abs_path, "./data/boston.csv")

    #下载波士顿数据集,如果不存在就下载
    if not os.path.exists(curr_path):
        download_boston_data(curr_path)
    
    # 训练模型
    train_model()