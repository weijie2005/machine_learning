from sklearn.neighbors import KNeighborsClassifier

def knn_model(X_train, y_train, X_test, n_neighbors=3):
    # 创建KNN模型
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # 训练模型
    knn.fit(X_train, y_train)
    
    # 预测
    y_pred = knn.predict(X_test)
    
    return y_pred

if __name__ == '__main__':
    # 示例数据
    X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y_train = [0, 0, 1, 1]
    X_test = [[1.5, 2.5], [3.5, 4.5]]
    
    # 调用KNN模型
    predictions = knn_model(X_train, y_train, X_test)
    print("预测结果:", predictions)