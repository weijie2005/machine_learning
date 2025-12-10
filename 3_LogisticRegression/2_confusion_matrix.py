from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


'''
逻辑回归的评估方法：F1 Score(好的模型需要 精确率和召回率都高，召回率高说明模型对正类的识别能力好，很重要)
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


    6. ROC曲线（Receiver Operating Characteristic Curve）：展示模型在不同阈值下的真阳性率（TPR）和假阳性率（FPR）的变化。
        TPR=TP/(TP+FN)
        FPR=FP/(FP+TN)
    7. AUC值（Area Under the Curve）：ROC曲线下的面积,用于评估模型的分类能力,值越大,模型性能越好。
        AUC的计算公式：AUC=1/2*积分(TPR-FPR)  从FPR=0到FPR=1的曲线面积

'''

def demo_confusion_matrix():
    # 假设y_true为真实标签，构建样本数据集
    y_true = ["恶性", "良性", "良性", "恶性", "良性", "恶性", "良性", "良性", "良性", "恶性"]

    # y_pred为模型预测标签
    y_pred = ["良性", "恶性", "恶性", "恶性", "良性", "良性", "良性", "良性", "良性", "良性"]

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=["恶性", "良性"])
    print("混淆矩阵:\n",cm)
    
    
    # 可视化混淆矩阵
    df=pd.DataFrame(cm, index=["恶性-正例", "良性-反例"], columns=["恶性-预测正例TP", "良性-预测反例FN"])
    print("混淆矩阵:\n",df)
    
    # 有了混淆矩阵，就可以计算准确率、精确率、召回率、F1值
    accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])
    precision = cm[0,0]/(cm[0,0]+cm[0,1])
    recall = cm[0,0]/(cm[0,0]+cm[1,0])
    f1_score = 2*precision*recall/(precision+recall)
    roc_auc = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])

    print("准确率:",accuracy)
    print("精确率:",precision)
    print("召回率:",recall)
    print("F1值:",f1_score)
    print("ROC AUC值:",roc_auc)

if __name__=='__main__':
    demo_confusion_matrix()
