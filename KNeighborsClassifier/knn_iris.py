from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

'''
K值distanc measure三种距离计算方法：
1. 欧氏距离=sqrt((x1-x2)^2+(y1-y2)^2)
2. 曼哈顿距离=|x1-x2|+|y1-y2|
3. 切比雪夫距离=max(|x1-x2|,|y1-y2|)
4. 闵可夫斯基距离=(|x1-x2|^p+|y1-y2|^p)^(1/p)

特征归一化：
1. 最小-最大归一化（Min-Max Scaling）：将特征值缩放到[0, 1]区间 y= (x-x_min)/(x_max-x_min) , x_min和x_max分别为特征的最小值和最大值。放大映射到[1,10]区间 m=y*(max-min)+min
2. 标准归一化（Standard Scaling）：将特征值转换为均值为0，标准差为1的分布，y= (x-x_mean)/x_std, x_mean=0特征的均值 和 x_std 标准差，x_std的平方=方差=1,标准归一化实现正态分布。
3. Z-分数归一化（Z-Score Normalization）：将特征值转换为标准正态分布。
4. 最大绝对值归一化（Max Absolute Scaling）：将特征值缩放到[-1, 1]区间。

'''

def knn_classifier_model(X_train, y_train, X_test, n_neighbors=3):
    # 创建KNN分类模型
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # 训练模型
    knn.fit(X_train, y_train)
    
    # 预测
    y_pred = knn.predict(X_test)
    
    return y_pred

def knn_regressor_model(X_train, y_train, X_test, n_neighbors=3):
    # 创建KNN回归模型
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    
    # 训练模型
    knn.fit(X_train, y_train)
    
    # 预测
    y_pred = knn.predict(X_test)
    
    return y_pred

def knn_maxminScaler():
    # 原始数据（特征1范围大，特征2范围小）
    data = np.array([[100, 1], [200, 2], [300, 3]])

    # 创建MinMaxScaler最小-最大归一化对象
    scaler = MinMaxScaler(feature_range=(1, 10))
    
    # 对数据进行归一化缩放
    scaled_data = scaler.fit_transform(data)    
    print("特征归一化后数据：\n", scaled_data)


def knn_standardScaler():
    # 原始数据（特征1范围大，特征2范围小）
    data = np.array([[100, 1], [200, 2], [300, 3]])

    # 创建StandardScaler标准归一化对象
    scaler = StandardScaler()
    
    # 对数据进行归一化缩放
    scaled_data = scaler.fit_transform(data)    
    print("特征标准归一化后数据：\n", scaled_data)
    print("特征标准归一化后数据的均值:", scaler.mean_)
    print("特征标准归一化后数据的标准差:", scaler.var_)

def iris_show():
    # 加载鸢尾花数据集    
    iris_data = load_iris()

    # print("特征数据:",iris_data.data[:5])
    # print("目标数据:", iris_data.target)
    # print("特征数据的列名:", iris_data.feature_names)
    # print("目标数据的类别:", iris_data.target_names)
    # print("数据集的文件名:", iris_data.filename)
    # print("数据集的描述:", iris_data.DESCR)

    # 数据集的特征处理，将特征数据转换为DataFrame格式，并增加目标数据列
    iris_df=pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    iris_df['target']=iris_data.target
    
    target_names=iris_data.target_names
    iris_df['target_name']=[target_names[i] for i in iris_df['target']]

    #print("数据集前五行:\n", iris_df.head(20))

    #seaborn可视化    
    sns.lmplot(x='Sepal Length (cm)', y='petal width (cm)', hue='target_name', data=iris_df,fit_reg=False)
    plt.title("Iris Sepal Length vs Petal Width")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.show()

    return 

def iris_knn_predict():
    # 1.加载鸢尾花数据集    
    iris_data = load_iris()

    # 2.数据集分类
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.3, random_state=22)

    # 3.特征标准归一化
    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    # 4.knn分类模型训练
    kc= KNeighborsClassifier(n_neighbors=3)
    kc.fit(x_train, y_train)

    # 5.模型评估
    print("模型准确率:", kc.score(x_test, y_test))

    # 6.模型预测
    new_data = [[5.1,3.5,1.4,0.2],
                [4.6,3.1,1.5,0.2]
            ]

    # 6.1对新数据进行归一化处理
    new_data=scaler.transform(new_data)

    # 6.2预测分类的结果
    new_pred = kc.predict(new_data)

    # 6.3预测分类的概率
    new_pred_proba = kc.predict_proba(new_data)

    print("新数据的预测类别:", new_pred,iris_data.target_names[new_pred][0])
    print("新数据的预测类别概率:", new_pred_proba)
    

if __name__ == '__main__':
    
    # 1.调用KNN分类模型
    # X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
    # y_train = [0, 0, 1, 1]
    # X_test = [[1.5, 2.5], [3.5, 4.5]]    
    # kc= knn_classifier_model(X_train, y_train, X_test)
    # print("分类预测结果:", kc)

    # 2.调用KNN回归模型
    # X_train = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
    # y_train = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    # X_test = [[3.5, 2.5], [3, 4.5]]    
    # kr = knn_regressor_model(X_train, y_train, X_test)
    # print("回归预测结果:", kr)

    # 3.调用归一化函数示例
    #knn_maxminScaler()
    #knn_standardScaler()

    # 4.实例：鸢尾花处理与预测
    #iris_show()
    iris_knn_predict()
