from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

'''
机器学习步骤
    1. 数据预处理：
        1. 缺失值处理：填充缺失值或删除包含缺失值的样本。
        2. 异常值处理：识别并处理异常值，例如使用统计方法（如Z-score或IQR）或可视化方法（如箱线图）。
        3. 特征选择：选择对目标变量有显著影响的特征，减少模型复杂度和过拟合风险。
    2. 特征工程：
        1. 归一化（Normalization）：将特征值缩放到[0, 1]区间，适用于特征范围不同的情况。
        2. 标准化（Standardization）：将特征值转换为均值为0，标准差为1的分布，适用于特征范围不同的情况。
    3. 模型选择：
        1. 选择合适的机器学习算法，根据问题类型（分类或回归）和数据特征（是否有标签样本）。
        2. 考虑模型的复杂度、计算成本和解释性。
    4. 模型训练：
        1. 使用训练集对选择的模型进行训练，调整模型参数以最小化损失函数。
    5. 模型预测：
        1. 使用训练好的模型对新的样本进行预测，得到分类标签或连续值输出。
    6. 模型评估：
        1. 使用测试集对训练好的模型进行评估，计算模型的性能指标（如准确率、精确率、召回率、F1值等）。
    7. 模型调优：
        1. 网格搜索（Grid Search）：通过尝试不同的超参数组合，找到最优的超参数设置。
        2. 交叉验证（Cross-Validation）：将数据集划分为训练集和验证集，多次重复训练和验证，评估模型的性能稳定性。
        3. 随机搜索（Random Search）：在超参数空间中随机采样，找到最优的超参数设置。
        4. 学习曲线（Learning Curve）：绘制模型在训练集和验证集上的性能指标，分析模型是否过拟合或欠拟合。

机器学习算法分类
    1.有监督学习：
        1. 分类问题：目标是分散的。
        2. 回归问题：目标是预测连续值输出。
    2.无监督学习：
        1. 聚类问题：目标是将样本分组为不同的簇，使得同一簇内的样本相似度较高，不同簇之间的样本相似度较低。
        2. 关联规则学习：目标是发现数据中的频繁项集和关联规则，用于发现数据中的模式和关联关系。
    3.半监督学习：
        1. 半监督分类问题：目标是利用有标签样本和无标签样本进行分类。
        2. 半监督回归问题：目标是利用有标签样本和无标签样本进行回归预测。
    4. 强化学习：
        1. 强化学习问题：目标是通过与环境交互，学习到一个策略，使智能体能够在环境中实现最大化奖励。

                        
KNeighbors-K近邻模型
一、K值distanc measure=n_neighbors距离计算方法：
    1. 欧氏距离=sqrt((x1-x2)^2+(y1-y2)^2)
    2. 曼哈顿距离=|x1-x2|+|y1-y2|
    3. 切比雪夫距离=max(|x1-x2|,|y1-y2|)
    4. 闵可夫斯基距离=(|x1-x2|^p+|y1-y2|^p)^(1/p)

二、特征归一化和标准化：
    1. 最小-最大归一化（Min-Max Scaling）：将特征值缩放到[0, 1]区间 y= (x-x_min)/(x_max-x_min) , x_min和x_max分别为特征的最小值和最大值。放大映射到[1,10]区间 m=y*(max-min)+min
    2. 标准化（Standard Scaling）：将特征值转换为均值为0，标准差为1的分布，y= (x-x_mean)/x_std, x_mean=0特征的均值 和 x_std 标准差，x_std的平方=方差=1,标准归一化实现正态分布。

三、交叉验证CV(cross-validation)：数据集划分
    1. 将数据集划分为K个相等大小的子集，通常采用K折交叉验证（K-Fold Cross-Validation）：即将数据集分为K个相等大小的子集，每个子集作为一次验证集，其他子集作为训练集。
    2. 模型训练：使用训练集对KNN模型进行训练，选择合适的K值。
    3. 模型评估：使用测试集对训练好的模型进行评估，计算模型的准确率、精确率、召回率、F1值等指标。
    4. 超参数调优：通过交叉验证选择最优的K值，通常采用网格搜索或随机搜索等方法。

四、网格搜索（Grid Search）：超参数调优
    1. 定义超参数网格：指定要调整的超参数及其可能的取值范围。
    2. 初始化网格搜索器：使用网格搜索器对象，将模型、超参数网格和评估指标传递给它。
    3. 执行网格搜索：调用网格搜索器的fit方法，对数据集进行训练和评估。
    4. 选择最优超参数：根据评估指标，选择具有最佳性能的超参数组合。

五、模型评估
    1.过拟合（Overfitting）：
        过拟合是指模型在训练数据上表现良好，但在未见过的数据上表现较差。
        过拟合通常发生在模型过于复杂或训练数据量不足的情况下。
        解决过拟合的方法包括增加训练数据量、减少模型复杂度、使用正则化技术等。
    2.欠拟合（Underfitting）：
        欠拟合是指模型在训练数据上表现较差，无法捕捉到数据的潜在模式。
        欠拟合通常发生在模型过于简单或训练数据量过少的情况下。
        解决欠拟合的方法包括增加模型复杂度、增加训练数据量、使用更复杂的模型等。

六、泛化（Generalization）：模型的 “举一反三” 能力
    核心定义：模型在未见过的新数据上的预测效果，也就是从训练数据学到的规律，能否迁移到真实场景的未知数据中。
    泛化能力强：模型在训练集和测试集上表现都好，不会 “死记硬背” 训练数据（避免过拟合），能应对真实场景的多变数据。
    泛化能力弱：模型只在训练集上表现好，遇到新数据就预测不准（即过拟合），比如把训练数据中的噪声也当成了规律。
    举例：用 1000 张猫的图片训练分类模型，泛化能力强的模型能准确识别从未见过的新猫图；泛化能力弱的模型可能只认识训练集中的猫，换一只姿势不同的猫就误判。

七、鲁棒性（Robustness）：模型的 “抗干扰” 能力
    核心定义：模型在数据存在噪声、异常值或轻微扰动时，依然能保持稳定预测效果的能力，简单说就是 “抗造”。
    鲁棒性强：数据有小误差（比如输入特征轻微错误、存在少量异常值）时，模型预测结果变化不大。
    鲁棒性弱：数据稍有扰动（比如图片加了一点模糊、数值特征有微小误差），模型就会出现大幅预测偏差。
    举例：一个鲁棒性强的垃圾邮件分类模型，即使邮件中出现少量错别字、特殊符号，依然能准确判断；鲁棒性弱的模型可能因为这些小干扰，把正常邮件误判为垃圾邮件。
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

    # 创建StandardScaler标准化对象
    scaler = StandardScaler()
    
    # 对数据进行标准化缩放
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

    # 2.数据集分类：x_train是特征数据，y_train是目标数据
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.3, random_state=22)

    # 3.特征标准归一化,将特征数据x_train和x_test转换为标准归一化后的格式,目标值不进行归一化处理    
    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    # 4.knn分类模型训练
    kc_model= KNeighborsClassifier(n_neighbors=3)
    kc_model.fit(x_train, y_train)

    # 5.模型评估
    score=kc_model.score(x_test, y_test)
    print("模型在测试集上的准确率:", score)

    # 5.1 对测试集进行预测并得到预测结果y_pred,再与真实目标数据y_test进行比较，计算准确率
    y_pred = kc_model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print("模型预测后的得到的目标数据，与真实目标数据的准确率:", acc)    

    # 6.模型预测
    new_data = [[5.1,3.5,1.4,0.2],
                [4.6,3.1,1.5,0.2]
            ]

    # 6.1对新数据进行归一化处理
    new_data=scaler.transform(new_data)

    # 6.2预测分类的结果
    new_pred = kc_model.predict(new_data)

    # 6.3预测分类的概率
    new_pred_proba = kc_model.predict_proba(new_data)
    print("新数据的预测类别:", new_pred,iris_data.target_names[new_pred][0])
    print("新数据的预测类别概率:", new_pred_proba)


def grid_search_cv():
    # 1.加载鸢尾花数据集    
    iris_data = load_iris()

    # 2.数据集分类：x_train是特征数据，y_train是目标数据
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.3, random_state=22)

    # 3.特征标准归一化,将特征数据x_train和x_test转换为标准归一化后的格式,目标值不进行归一化处理    
    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    # 4. 初始化KNN分类模型,n_neighbors默认值是5
    model=KNeighborsClassifier()

    # 5.网格搜索调参，cv使用5折交叉验证
    param_grid = {'n_neighbors': [3,4,5, 6,7,8]}
    estimator = GridSearchCV(model, param_grid, cv=5)
    estimator.fit(x_train, y_train)

    # 6.输出最佳参数和最佳得分
    print("最佳参数:", estimator.best_params_)
    print("最佳得分:", estimator.best_score_)       
    print("最佳模型:", estimator.best_estimator_)
    print("交叉验证结果:", estimator.cv_results_)

    #7.使用最优超参n_neighbors=6，训练模型并评估
    new_model=KNeighborsClassifier(n_neighbors=6)
    new_model.fit(x_train, y_train)
    score=new_model.score(x_test, y_test)
    print("模型评估的准确率:", score)
    
    #7.1评估测试集
    y_pred = new_model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print("模型预测后的得到的目标数据，与真实目标数据的准确率:", acc)    

    #8.模型预测
    new_data = [[5.1,3.5,1.4,0.2],
                [4.6,3.1,1.5,0.2]
            ]

    # 8.1对新数据进行归一化处理
    new_data=scaler.transform(new_data)

    # 8.2预测分类的结果
    new_pred = new_model.predict(new_data)

    # 8.3预测分类的概率
    new_pred_proba = new_model.predict_proba(new_data)
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
    #iris_knn_predict()

    # 5.使用交叉验证CV(cross-validation)和网格搜索（Grid Search）调参
    grid_search_cv()
