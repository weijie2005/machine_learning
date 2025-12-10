
import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


'''
逻辑回归模型
一、逻辑回归模型的基本原理
    1. 逻辑回归模型是一种用于二分类任务的统计模型,用于预测一个样本属于某个类别的概率。
    2. 逻辑回归模型的假设函数为：h(x) = sigmoid(z)，其中z = w^T * x + b，sigmoid函数将z映射到(0, 1)之间的概率值。
    3. 逻辑回归模型的损失函数为：J(w, b) = -[y * log(h(x)) + (1 - y) * log(1 - h(x))]，其中y是真实标签（0或1）。
    4. 逻辑回归模型的优化目标是最小化损失函数J(w, b)，常用的优化算法有梯度下降法。
    5. 逻辑回归模型的预测结果为：y_pred = 1 if h(x) >= 0.5 else 0。

二、逻辑回归模型的应用场景
    1. 逻辑回归模型广泛应用于二分类任务,如 spam 邮件分类、疾病诊断等。
    2. 逻辑回归模型的假设函数可以解释为样本属于某个类别的概率,因此可以用于风险评估、概率预测等任务。

三、逻辑回归模型的优缺点
    1. 逻辑回归模型的主要优点是简单、易于实现、计算效率高。
    2. 逻辑回归模型的主要缺点是假设数据服从逻辑分布,如果数据不满足该假设,模型的性能可能会下降。

四、逻辑回的数学基础    
    1.sigmoid函数：sigmoid(z) = 1 / (1 + exp(-z))
    2. 逻辑回归模型的损失函数：J(w, b) = -[y * log(h(x)) + (1 - y) * log(1 - h(x))]
    3. 逻辑回归模型的梯度下降法：w = w - α * ∂J(w, b) / ∂w, b = b - α * ∂J(w, b) / ∂b

五、逻辑回归模型的数学概率解释
    1. 逻辑回归模型的假设函数可以解释为样本属于某个类别的概率,即：P(y=1|x) = h(x) = sigmoid(z)
    2. 逻辑回归模型的预测结果为：y_pred = 1 if h(x) >= 0.5 else 0
    3. 逻辑回归模型的预测结果可以解释为：当样本属于类别1的概率大于等于0.5时,模型将预测为类别1,否则预测为类别0。

六、概率运算核心规则
    1.概率与联合概率
        1. 概率:事件发生的可能性
        2. 联合概率:多个事件同时发生的概率P(A∩B)
        3. 条件概率：P(A|B) = P(A∩B) / P(B)，表示在事件B发生的条件下,事件A发生的概率。

    2.概率的基本公理（柯尔莫哥洛夫公理）
        1.非负性：对任意事件A ，P(A)≥0 （概率不能为负数）
        2.规范性：必然事件Ω的概率为1，即P(Ω)=1 （所有可能结果的概率和为 1)
        3.可列可加性：有限场景下P(A1 U A2 U...U An)=P(A1)+P(A2)+...+P(An)
        4.不可能事件概率:P(∅)=0 (空集无任何结果，概率为 0)
        5.补集概率:P(A')=1-P(A)  (事件A不发生的概率 = 1 - 发生的概率,A'表示A的补集）

        注意：P(A∣B)表示 “在B发生的条件下，A发生的概率
        6.并事件概率（加法公式）：P(A U B) = P(A) + P(B) - P(A∩B)  (事件A或B发生的概率)
        7.互斥事件特例（互斥事件的并事件概率为其概率和）：P(A U B) = P(A) + P(B) （A和B互不相交）
        8.交事件概率（乘法公式）：P(A∩B) = P(A|B) * P(B)，表示事件A和事件B同时发生的概率等于事件B发生的条件下事件A发生的概率乘以事件B发生的概率。
        9.独立事件特例（独立事件的交事件概率为其概率乘积）：P(A∩B) = P(A) * P(B) （A和B互不相关）
        10.差事件概率：P(A−B)=P(A)−P(A∩B)（事件A发生但B不发生的概率（A−B，即A∩B'））

        11.条件概率：P(A|B) = P(A∩B) / P(B)，表示在事件B发生的条件下,事件A发生的概率。
        12.贝叶斯公式：P(A|B) = P(B|A) * P(A) / P(B)，表示在事件A发生的条件下,事件B发生的概率。
        机器学习应用：朴素贝叶斯分类器的核心（如垃圾邮件检测中，由 “邮件含关键词” 反推 “是垃圾邮件” 的概率）

    3.极大似然函数
        1. 极大似然函数：P(yi|xi;w,b)的乘积，yi是第i个样本的真实标签（0或1），xi是第i个样本的特征向量，w是权重向量，b是偏置项。
        2. 极大似然函数的目标是找到使P(yi|xi;w,b)的乘积最大的w和b。

七、对数的性质
        ln=log(e)
        1. 对数函数的性质：log(a*b) = log(a) + log(b)（乘法性质）
        2. 对数函数的性质：log(a/b) = log(a) - log(b)（除法性质）
        3. 对数函数的性质：log(a^b) = b * log(a)（指数性质）
        4. 对数函数的性质：log(a) = exp(log(a))（反函数性质）
        5. 对数函数的性质：log(exp(a)) = a（指数函数性质）

八、逻辑回归的基本思想
    1. 逻辑回归的概念：是一种分类模型，将线性回归的输出作为逻辑回归的输入，输出是0到1之间的值
    2. 逻辑回归模型的基本思想是：通过对输入特征进行线性组合,并使用sigmoid函数将结果映射到(0, 1)之间的概率值,来预测样本属于某个类别的概率。
    3. 假设函数：h(x) = sigmoid(z) = 1 / (1 + exp(-z))
    4. 损失函数：使用交叉熵损失函数 H(P,Q)=- Σ p(x) * log(q(x))

九、什么是熵
    1. 熵（Entropy）是信息论中的一个重要概念,用于衡量随机变量的不确定性或信息的丰富程度。
    2. 熵的公式为：H(X) = - Σ p(x) * log(p(x))，其中p(x)是随机变量X取x的概率。公式中加负号是为了让熵值非负,符合信息论的定义。
    3. 熵的性质：熵的值越大,随机变量的不确定性就越大;熵的值越小,随机变量的不确定性就越小。
        1. 熵（Entropy）是信息论中用于衡量随机变量不确定性的概念。
        2. 熵的公式为：H(X) = - Σ p(x) * log(p(x))，其中p(x)是随机变量X取不同值的概率。
        3. 熵的性质：熵的值越大,随机变量的不确定性就越大。
        4. 熵的性质：熵的值在0到1之间,当随机变量的所有可能值的概率相等时,熵的值为1（最大不确定性）。

十、二分类的最小损失函数：交叉熵损失函数
    0. 交叉熵是衡量 “预测分布” 与 “真实分布” 之间差异的指标—— 两个分布越接近，交叉熵越小；差异越大，交叉熵越大
    1. 交叉熵损失函数（Cross-Entropy Loss）是逻辑回归模型中常用的损失函数,用于衡量模型预测的概率分布与真实标签的差异。
    2. 交叉熵损失函数的公式为：H(P,Q)=- Σ p(x) * log(q(x))，    对于两个离散概率分布P真实分布）和 Q（预测分布），交叉熵公式为
    3. 交叉熵损失函数的性质：当模型预测的概率分布与真实标签的分布越接近时,交叉熵损失函数的值越小。
    4. 交叉熵损失函数的优化目标是最小化损失函数J(w, b)，常用的优化算法有梯度下降法。

    
十一、逻辑回归的评估方法：F1 Score(好的模型需要 精确率和召回率都高，召回率高说明模型对正类的识别能力好，很重要)
    0. 混淆矩阵（Confusion Matrix）：展示模型分类结果的详细矩阵：            
        Positive正,   Negative负/反
        --------------------------
        1.真正例（TP）、伪反例（FN）
        2.伪正例（FP）、真反例（TN）

        真实与预测结果对比：
        1. 真正例（TP）：实际为正类(1)，模型预测也为正类（1）
        2. 伪反例（FN）：实际为正类（1），模型预测为负类（0）
        3. 伪正例（FP）：实际为负类（0），模型预测为正类（1）
        4. 真反例（TN）：实际为负类（0），模型预测也为负类（0）。

    1. 准确率（Accuracy）：分类正确的样本数占总样本数的比例 P=TP+TN/(TP+TN+FP+FN)
    2. 精确率（Precision）：也叫查准率即查得准不准，模型预测为正类的样本中,实际为正类的比例 P=TP/(TP+FP)
    3. 召回率（Recall）：也叫查全率即查得全不全，实际为正类的样本中,模型预测为正类的比例 P=TP/(TP+FN)
    4. F1值（F1 Score）：精确率和召回率的调和平均值,用于综合评估模型的性能。P=2*Precision*Recall/(Precision+Recall)

    5.TPR（True Positive Rate）：也叫查全率即查得全不全，实际为正类的样本中,模型预测为正类的比例,正样本的正确率 P=TP/(TP+FP)
    6.FPR（False Positive Rate）：也叫伪正率即查得伪正全不全，实际为负类的样本中,模型预测为正类的比例，负样本的错误率 P=FP/(FP+TN)

    6. ROC曲线（Receiver Operating Characteristic Curve）：展示模型在不同阈值下的真阳性率（TPR）和假阳性率（FPR）的变化。
    7. AUC值（Area Under the Curve）：ROC曲线下的面积,用于评估模型的分类能力,值越大,模型性能越好。


    

'''



'''
实例：
Breast Cancer Wisconsin" 威斯康星乳腺癌数据集,是机器学习、数据科学领域的经典数据集名称,简称：BCW Dataset
数据集本质是二分类任务数据（标签：良性 / 恶性），是入门机器学习分类算法的标杆数据集
'''

def download_data(data_path):
    # 加载乳腺癌数据集,使用sklearn库中的load_breast_cancer函数
    data = load_breast_cancer()
    x=data.data     # 特征矩阵（569行×30列）
    y=data.target    # 标签（0=良性，1=恶性）
    feature_names = data.feature_names  # 特征名称

    # 查看数据基本信息
    #print(data.DESCR)
    #print(f"数据集样本数：{x.shape[0]}, 特征数：{x.shape[1]}")
    #print(f"良性样本数：{sum(y==0)}, 恶性样本数：{sum(y==1)}")
    #print(f"前5个特征名称：{feature_names[-1]}")

    # 将目标列名转为字符串类型
    column_names =list(feature_names) + ['target']

    # 合并特征数据和目标数据
    y=y.reshape(-1,1)
    new_data=np.hstack([x,y])

    # 创建DataFrame
    df = pd.DataFrame(data=new_data, columns=column_names)

    # 转换目标列数据类型为整数
    df['target']=df['target'].astype(int)

    #将数据保存到csv文件中    
    df.to_csv(data_path, index=False)
    
    return


def read_data(data_path):
    # 从csv文件中读取数据
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])  # 特征矩阵（569行×30列）
    y = df['target']                # 标签（0=良性，1=恶性）
    feature_names = X.columns       # 特征名称

    return X, y, feature_names


def train_model(X,y,feature_names):
    #数据清理,将数据中的缺失值（'?'）替换为NaN值    
    X = X.replace(to_replace='?', value=np.nan)
    # 删除 X 中 包含缺失值（nan）的所有行
    X = X.dropna()

    # 同步更新标签 y, 确保 y 与 X 索引对齐
    y = y[X.index]

    # 数据切割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 实例化逻辑回归模型，建模：solver=liblinear + L1正则化（适合高维数据，可稀疏参数）
    # penalty(惩罚)但在机器学习中是(正则化项)的意思,这里使用L1正则化，C为正则化参数,值越小,正则化越强,
    # solver（求解器，解决方案）的本质是「寻找最优模型参数的优化算法」—— 逻辑回归的目标是最小化「损失函数」（如二分类的交叉熵损失）
    # solver='liblinear' 表示使用liblinear库实现逻辑回归,适用于小数据集。若需用核函数（如 RBF、多项式核）实现非线性逻辑回归，需改用 solver='sag'/'saga'/'lbfgs'
    # liblinear是一个经典的线性分类器优化库，专门针对线性模型（含逻辑回归）和高维稀疏数据设计。样本量限制：样本数超过10万时，收敛速度远慢于 sag/saga（随机梯度下降类求解器）
    
    model = LogisticRegression(
        solver='liblinear',  # 使用liblinear求解器
        penalty='l1',        # L1正则化（稀疏参数）
        C=1.0,               # 惩罚强度（C越小惩罚越强）
        max_iter=1000        # 最大迭代次数（确保收敛）
    )
    # 模型训练
    model.fit(X_train, y_train)

    # 模型预测
    y_pred = model.predict(X_test)

    # 模型评估
    print(f"模型测试准确率：{accuracy_score(y_test, y_pred):.4f}")    


if __name__ == '__main__':
    # 下载数据集及保存路径和文件名    
    curr_path=os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(curr_path,"./data/breast_cancer.csv")
    
    # 如果文件不存在,则下载数据集
    if not os.path.exists(data_path):
        download_data(data_path)

    # 从csv文件中读取数据
    X, y, feature_names = read_data(data_path)

    # 训练逻辑回归模型
    train_model(X,y,feature_names)

