import os 
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

'''
阿里天池网站：https://tianchi.aliyun.com/
阿里天池比赛，数据集下载，有视屏教程，有练习和比赛，有奖金和证书，是学习的好地方

电信客户流失分析，churn.csv数据集下载地址：
https://tianchi.aliyun.com/ 在菜单中找到数据集点击，搜索框输入churn.csv，找到并点击下载，他的全称为WA_Fn-UseC_-Telco-Customer-Churn.csv
'''

def read_data(data_path):
    # 从csv文件中读取数据
    df = pd.read_csv(data_path)

    # 数据预处理one-hot编码,将列Churn转换为二进制值,0表示未流失,1表示流失
    dum_data=pd.get_dummies(df,columns=['Churn'])

    # 查看数据基本信息
    # print(data.head())
    # print(data.describe())
    # print(data.info())

    # 去掉不需要的数据列
    data = dum_data.drop(columns=['Churn_No'],axis=1)  # 特征矩阵（7043行×20列）
    
    # 目标列标签重命名
    data = data.rename(columns={'Churn_Yes': 'target'})

    # 重新查看处理后的数据
    print(data.head())   

    # 查看目标列标签分布
    target_counts = data.target.value_counts()
    print("目标列标签分布:",target_counts)
    print("特征列名称:",data.columns)
    
    return data

def train_model(data):
    #数据处理,选择特征tenure,MonthlyCharges,Contract
    x=data[['tenure','MonthlyCharges','Contract']]

    # 修改这里：将y转换为一维数组而不是DataFrame列，使用.values获取一维numpy数组
    y=data['target'].values 

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    # 特征工程,将Contract列转换为数值类型
    x_train['Contract'] = x_train['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    x_test['Contract'] = x_test['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

    # 模型训练
    model = LogisticRegression(
        solver='sag',       # 使用sag求解器
        penalty='l2',        # L2正则化（稀疏参数）
        C=1.0,               # 惩罚强度（C越小惩罚越强）
        max_iter=1000        # 最大迭代次数（确保收敛）
    )
    model.fit(x_train, y_train)

    # 模型预测
    y_pred = model.predict(x_test)

    # 模型评估
    acc=model.score(x_test,y_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    # 分类报告
    report = classification_report(y_test, y_pred,target_names=['未流失','流失'])
    
    print(f"模型准确率: {acc:.4f}")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"分类报告:\n{report}")

if __name__=='__main__':
    # 读取数据
    curr_path=os.path.dirname(__file__)   
    data_path=os.path.join(curr_path,'./data/Customer-Churn.csv')

    data = read_data(data_path)

    # 模型训练
    train_model(data)