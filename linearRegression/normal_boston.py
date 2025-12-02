
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

'''
1.找到最优模型->最小损失函数->正规方程(直接求导)，当特征数量较少时使用

2.正规方程（Normal Equation）：通过偏导和矩阵转置求解参数
    最小值公式：w=(X^T * X)^-1 * X^T * y    <=  f(x)=|| Xw-y ||^2 对w求最小值，<=使用该范数等于(A^T * A)的逆矩阵
    用途：直接求解线性回归模型的参数，无需迭代，特征数量较少时使用。
    注意：仅适用于方阵，且 X 必须满秩（列数等于样本数）,否则不可逆。所以正规方程有可能无法求解。
    
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

    # 训练模型
    model=LinearRegression(intercept=True)
    model.fit(x_train,y_train)

    # 模型预测
    y_pred=model.predict(x_test)
    print(f"模型预测结果: {y_pred}")
    print(f"模型权重: {model.coef_}")
    print(f"模型偏置: {model.intercept_}")

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