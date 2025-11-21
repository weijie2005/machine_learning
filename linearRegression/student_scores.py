from sklearn.linear_model import LinearRegression
import joblib
import os


'''
机器学习之线性回归模型实例
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
    