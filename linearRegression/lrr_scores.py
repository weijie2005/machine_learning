from sklearn.linear_model import LinearRegression
import joblib
import os


'''
线性回归模型
一、模型定义
    模型定义：线性回归模型是一种用于预测连续数值的监督学习模型。

二、线性回归方程：
    模型假设：假设输入特征为x，目标变量为y，模型的预测值为y_pred。
    模型公式：y_pred = w1*x1 + w2*x2 + ... + wn*xn + b
    其中，w1, w2, ..., wn 是模型的系数（权重），b 是模型的截距（偏置项）。
    模型目标：通过最小化预测值与真实值之间的均方误差（MSE）来训练模型，即最小化：
    MSE = (1/n) * Σ(y_pred - y_true)^2
    其中，n 是样本数量，y_pred 是模型的预测值，y_true 是真实值。

三、什么是损失函数（Loss Function）
    误差定义：误差是模型预测值与真实值之间的差异,即y_pred - y_true
    损失函数定义：损失函数是一种用于衡量模型预测值与真实值之间差异的函数。   
    损失函数目标：通过最小化损失函数值，来优化模型的系数和截距，使模型预测值与真实值更接近。
    损失函数最小化：通过优化算法（如梯度下降），最小化损失函数，找到最优的模型系数和截距，使模型预测值与真实值的差异最小。

四、损失函数的种类与数学表达式
    1.均方误差（MSE-Mean Squared Error）:预测值与真实值之差的平方的平均值，也叫最小二乘法（OLS）
        损失函数公式：MSE = (1/n) * Σ(y_pred - y_true)^2
        其中，n 是样本数量，y_pred 是模型的预测值，y_true 是真实值。
        特点：
            平方项会放大较大误差，对异常值敏感（适合数据分布较为规整的场景）。
            函数是连续可导的，便于使用梯度下降等优化算法求解最优参数。
        用途：
            线性回归中最常用的损失函数（普通最小二乘法 OLS 的核心）。
       
    2.平均绝对误差（MAE-Mean Absolute Error）:预测值与真实值之差的绝对值的平均值
        损失函数公式：MAE = (1/n) * Σ|y_pred - y_true|
        其中，n 是样本数量，y_pred 是模型的预测值，y_true 是真实值。
        特点：
            对异常值不敏感（鲁棒性更强）。
            在误差为 0 处不可导，优化时可能存在收敛速度慢的问题
        用途：
            适合数据中存在较多异常值的场景

    3.均方根误差（RMSE-Root Mean Squared Error）:预测值与真实值之差的平方的平均值的平方根
        损失函数公式：RMSE = sqrt(MSE)
        其中，MSE 是均方误差。
        特点：
            对量纲与原始数据一致（如预测房价时，RMSE 单位为 “元”），更易解释
            同样对异常值敏感
        用途：
            评估模型性能时常用（比 MSE 更直观）

    4.Huber 损失（Robust Loss）:结合 MSE 和 MAE 的优点，在误差较小时用平方项，误差较大时用绝对值项
        损失函数公式：
            当 |y_pred - y_true| <= δ 时，损失为 (1/2) * (y_pred - y_true)^2
            当 |y_pred - y_true| > δ 时，损失为 δ * |y_pred - y_true| - (1/2) * δ^2
        其中，n 是样本数量，y_pred 是模型的预测值，y_true 是真实值，δ 是一个阈值。
        特点：
            对异常值不敏感（鲁棒性更强），在误差较小时用平方项，误差较大时用绝对值项，平衡了 MSE 和 MAE 的优点。            
        用途：
            适合数据中存在较多异常值的场景，需要平衡误差敏感度和优化效率的场景

五、数据类型
    1.标量（Scalar）：单个数值，如 3、-5.2、100 等。
    2.向量（Vector）：有序的数值列表，如 [1, 2, 3]、[-1.5, 0.5, 2.0] 等。
    3.矩阵（Matrix）：二维数组，如 [[1, 2, 3], [4, 5, 6]]、[[-1.5, 0.5, 2.0], [3.2, -2.1, 1.8]] 等。
    4.张量（Tensor）：多维数组，如 3D 张量、4D 张量等。

六、常用导数公式：加法、乘法、除法、链式法则
    导数公式：
        1.常数函数的导数：
            函数：y = C（C为常数）
            导数：y' = 0
        2.幂函数的导数：
            函数：y = x^n（n为实数）
            导数：y' = nx^(n-1)
        3.指数函数的导数：
            函数：y = e^x（自然底数）
            导数：y' = e^x
        4.对数函数的导数：
            函数：y = log_a(x)（a为大于0且不等于1的数）
            导数：y' = 1/(x*ln(a))
        5.指数函数的导数：
            函数：y = a^x（a为大于0的数）
            导数：y' = a^x * ln(a)
            
    导数四则运算：
        1.加法规则：(u+v)' = u' + v'
        2.乘法规则：(u.v)' = u'v + uv'
        3.除法规则：(u/v)' = (u'v - uv')/v^2
        4.链式法则：(g(f(x)))' = g'(f(x)) * f'(x)

七、偏导（Partial Derivative）
    定义：偏导是多变量函数中，对其中一个变量求导，其他变量保持不变的导数。
    公式：(df/dx) = lim(h->0) [(f(x+h) - f(x))/h]
    其中，df/dx 是函数 f 在变量 x 上的偏导，h 是一个小的变化量。
    用途：
        用于优化多变量函数，找到函数的局部最优解。
        在机器学习中，用于计算梯度，优化模型参数。
    示例，二元函数偏导：
        函数：f(x,y) = x^2 + 3xy + y^3
        对 x 求偏导固定y即y为常数：f_x = 2x + 3y     #(x^2的导数为2x,3xy中y为常数，导数为3y, y^3导数为0) 
        对 y 求偏导固定x即x为常数：f_y = 3x + 3y^2   #(x^2导数为0,3xy中x为常数，导数为3x,y^3导数为3y^2)
'''
def scores_model(study_scores,model_path):
    #训练数据
    X_train = [[80,86], [82,80], [85,78], [90,90], [86,82],[82,90],[78,80],[92,94]]
    y_train = [84.2,80.6,80.1,90,83.2,87.6,79.4,93.4]

    #创建模型
    mymodel = LinearRegression()
    print("模型:",mymodel)
    
    #训练模型
    mymodel.fit(X_train, y_train)
    print("模型系数:",mymodel.coef_)
    print("模型截距:",mymodel.intercept_)    

    #预测
    predictions = mymodel.predict(study_scores)
    print("预测结果:",predictions)

    #保存模型
    joblib.dump(mymodel, model_path)

    return 


if __name__ == '__main__':
    # 获取当前脚本的绝对路径
    script_path = os.path.abspath(__file__)
    # 获取脚本所在的目录
    current_directory = os.path.dirname(script_path)

    # 构建模型文件的完整路径
    model_path = os.path.join(f'{current_directory}', 'models')

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # 构建模型文件的完整路径
    model_path = os.path.join(model_path, 'student_scores_model.pkl')

    #测试数据，并保存训练好的模型
    study_scores = [[90,80]]
    #scores_model(study_scores,model_path)

    # 加载保存的模型
    new_mymodel = joblib.load(model_path)

    # 使用模型进行预测
    new_scores = [[68,80]]
    new_predictions = new_mymodel.predict(new_scores)

    print("模型重新预测结果:",new_predictions)
    