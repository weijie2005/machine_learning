from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd
import numpy as np
import os


'''
决策树的基本原理    
    1. 决策树是一种树形结构的模型,用于分类和回归任务。
    2. 决策树的每个节点表示一个特征(attribute),每个分支表示一个决策规则,每个叶节点表示一个类别(分类任务)或数值(回归任务)。
    3. 决策树的构建过程是一个递归过程,每次选择一个最优特征作为当前节点,将数据集分成多个子集,直到满足停止条件(如最大深度、最小样本数等)。
    4. 决策树是贪心算法（Greedy Algorithm），每次分裂时选择 “当前最优” 的特征，而非全局最优，目的是快速构建有效模型。 

决策树算法（ID3、C4.5、CART），核心流程一致：
    1. 特征选择（Feature Selection）
    2. 树的构建（Tree Construction）
    3. 树的剪枝（Tree Pruning）


决策树的构建过程
    1. 特征选择（Feature Selection）
        目标：从所有候选特征中，选择一个 “最能区分样本类别” 的特征作为当前节点的分裂依据。
        核心指标：通过量化 “节点分裂前后的纯度提升” 选择特征（下文详细讲解）。

        1. 信息增益（Information Gain）：选择能够最大程度减少不纯度的特征作为分裂特征。
        2. 增益率（Gain Ratio）：考虑特征的取值数量,选择增益率最高的特征。
        3. 基尼指数（Gini Index）：选择能够最小化基尼系数的特征作为分裂特征。

    2. 树的构建（Tree Construction）
        递归分裂：每个内部节点分裂后，生成子节点，对子节点重复 “特征选择 - 分裂” 过程。
        停止条件（避免过拟合）：
        节点中所有样本属于同一类别（纯度 = 1）；
        无剩余特征可分裂，或剩余特征无法提升纯度；
        节点样本数小于阈值（如≤5），直接作为叶节点（类别为节点中样本数最多的类）。

    3. 树的剪枝（Tree Pruning）
        问题：未剪枝的树易过拟合（训练集准确率高，测试集差），因过度学习训练集噪声。
        目标：移除冗余分支，降低模型复杂度，提升泛化能力。
        方法：预剪枝（构建时限制深度、样本数）、后剪枝（构建后移除无效分支）。

ID3决策树：
    1. 熵（Entropy）：即指信息熵（Information Entropy）
        定义：熵是信息论中衡量随机变量不确定性的指标(不确定性的度量指标)，值越大表示不确定性越高,无序程度越高。越高也表示信息越丰富和多，值越小则表示信息越有序和信息越少。
        公式：H(D)=-\sum_{i=1}^{n} p_i \log_2 p_i

    2.信息熵
        定义：信息熵是数据集D的不确定性度量，值越大表示数据集越无序，信息越丰富。
        公式：H(D)=-\sum_{i=1}^{n} p_i \log_2 p_i

    3.条件熵
        定义：条件熵是在已知某个特征的情况下，该特征下样本的熵。
        公式：H(D|a)=-\sum_{i=1}^{n} \frac{|D_i|}{|D|} H(D_i)

    4. 信息增益（Information Gain）：G(D,a)=信息熵H(D)-条件熵H(D|a)
        定义：信息增益是熵的减少量，即当前节点的熵减去分裂后子节点的熵的加权平均值。
        公式：G(D,a)=信息熵H(D)-条件熵H(D|a)

    5.ID3决策树构建流程：
        1. 计算数据集的信息熵H(D)。
        2. 对每个特征a，计算其条件熵H(D|a)。
        3. 计算信息增益G(D,a)=H(D)-H(D|a)。
        4. 选择信息增益最大的特征作为当前节点的分裂特征。
        5. 对每个特征值，将数据集分裂为子集，递归构建子树。
        6. 重复以上步骤，直到满足停止条件。

C4.5决策树：
    1.特征熵（Feature Entropy）
        定义：特征熵是指在数据集D中，特征a的不确定性度量，值越大表示特征a的取值越无序，信息越丰富。
        公式：IV(a)=

    2.信息增益率（Gain Ratio）
        定义：增益率是信息增益与特征熵的比值，用于解决信息增益对取值数量敏感的问题。每一个特征的增益率都不同，根据增益率选择特征。
        公式：Gain Ratio(D,a)=G(D,a)/IV(a)
        解释：增益率越大，表示特征a的分裂效果越好。
        物理意义：增益率是用来衡量特征对分类效果的贡献度，增益率越大，表示特征a的分裂效果越好。

    3.惩罚系数（Penalty Coefficient）
        定义：惩罚系数是指在C4.5决策树中，为了避免过拟合，引入的一个超参数，用于平衡模型的复杂度和泛化能力。
        公式：C=1-\frac{1}{|D|}

    4.ID3决策树和C4.5决策树的区别：
        1. 信息增益：ID3使用信息增益选择特征，C4.5使用增益率选择特征。
        2. 处理缺失值：ID3不处理缺失值，C4.5可以处理缺失值。
        3. 处理连续值：ID3不处理连续值，C4.5可以处理连续值。
        4. 剪枝策略：ID3不剪枝，C4.5使用预剪枝和后剪枝。
        5. C4.5解决ID3不足的问题：
            1. 信息增益对取值数量敏感：ID3使用信息增益选择特征，当特征取值数量很多时，信息增益会被偏向于取值数量多的特征。            
            2. 过拟合问题：ID3不剪枝，容易过拟合训练集。
            3. 增益率对取值数量不敏感：C4.5使用增益率选择特征，不考虑特征取值数量，更能反映特征的分裂效果。

        6. ID3的信息增益算法，倾向于选择取值数量多的特征，而C4.5的增益率算法，不考虑特征取值数量，更能反映特征的分裂效果。

        
CART决策树(Classification and Regression Tree)：
    1. 定义：CART决策树是即可用于解决分类，又能用于回归问题。
    2. 分类问题：CART决策树使用基尼指数（Gini Index）选择分裂特征，基尼指数越小，表示特征的分裂效果越好。
    3. 回归问题：CART决策树使用最小二乘法策略，即选择能够最小化均方误差的特征作为分裂特征。


'''

'''
实例：泰坦尼克号幸存人员数据分析
'''

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def download_data(data_path):
    # 下载数据集,读取 GitHub 上的公开泰坦尼克数据集
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    titanic = pd.read_csv(url)
    # 查看数据基本信息
    print(titanic.info())

    # 保存到本地
    titanic.to_csv(data_path, index=False)

def read_data(data_path):
    titanic = pd.read_csv(data_path)

    # 查看数据基本信息
    print(titanic.describe())
    #print(titanic.info())

    # 查看目标列标签分布
    target_counts = titanic.Survived.value_counts()
    #print("目标列标签分布:",target_counts)
    #print("特征列名称:",titanic.columns)

    #print("数据集前5行：")
    print(titanic.head())

    return titanic

def train_model(data):
    # 数据预处理
    # 仅加载需要的特征列
    X = data[['Pclass','Sex','Age']].copy()        # 创建明确的副本 特征矩阵（891行×4列）
    Y = data['Survived'].copy()                         # 目标列（891行×1列）

    # 缺失值处理，使用平均值填充
    X['Age'].fillna(X['Age'].mean(),inplace=True)

    # 类别特征编码：sex转换为数值类型,one-hot编码
    X = pd.get_dummies(X,columns=['Sex'])
    
    # 查看处理后的数据
    print("列名：",X.columns)   
    print("特征列数量:\n",X.count())   

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

    # 模型训练
    model = DecisionTreeClassifier(
        criterion='gini',   # 使用的算法标准是基尼指数（即CART决策树），如果使用信息增益（entropy）则是ID3决策树
        max_depth=3,        # 最大深度，默认值为None，即不限制深度
        min_samples_split=2,    # 最小样本分裂数
        min_samples_leaf=1,     # 最小样本叶子数
        max_features=None,
        random_state=42
    )    
    model.fit(x_train, y_train)

    # 模型评估
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.4f}")

    # 混淆矩阵（查看真阳性/假阳性等）
    print("\n混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))

    # 打印详细分类报告（精确率、召回率、F1值）
    print("分类报告:")
    print(classification_report(y_test, y_pred, target_names=['Dead', 'Survived']))


    # 可视化决策树
    plt.figure(figsize=(20,10))  # 调整图像大小
    plot_tree(model, feature_names=['Pclass','Age','Sex_female', 'Sex_male'], class_names=['Dead', 'Survived'], filled=True)
    plt.title("泰坦尼克生存预测决策树",fontsize=16)
    plt.show()


if __name__ == '__main__':
    # 下载数据
    curr_path=os.path.join(os.path.dirname(__file__))
    data_path = os.path.join(curr_path,'./data/titanic.csv')

    if not os.path.exists(data_path):
        download_data(data_path)

    # 读取数据
    data = read_data(data_path)

    # 训练模型
    train_model(data)