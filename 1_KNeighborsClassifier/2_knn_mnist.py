import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import joblib
'''
0.手写数字识别，使用KNN算法训练模型
1.数据集下载
    网站上下载数据集四个文件：https://www.kaggle.com/datasets/hojjatk/mnist-dataset ，也可以在国内找资源下载
    train-images-idx3-ubyte.gz ,train-labels-idx1-ubyte.gz ,t10k-images-idx3-ubyte.gz ,t10k-labels-idx1-ubyte.gz
    下载后手动解压，再通过代码读取并转换为csv文件
2.模型训练
3.模型评估
4.模型预测
'''

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def load_mnist_sklearn():
    # 下载MNIST数据集，无法下载
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    X, y = mnist.data, mnist.target.astype(int)
    
    # 转换为DataFrame格式
    df = pd.DataFrame(X)
    df.columns = [f'pixel_{i}' for i in range(784)]
    df['label'] = y
    
    return df

def mnist_dataset_to_csv(mnist_path):
    """
    手动读取MNIST二进制文件,需要先下载MNIST原始数据到指定路径
    MNIST图像文件遵循特定的二进制格式规范：
        魔数(Magic Number): 4字节，标识文件类型
        图像数量: 4字节，表示文件中包含多少张图片
        行数: 4字节，每张图片的行数(28)
        列数: 4字节，每张图片的列数(28)
        像素数据: 每个像素用一个字节(0-255)表示灰度值

    图片文件格式：
        ├── 魔数(4字节) → 识别文件类型
        ├── 图像数量(4字节) → 知道有多少张图片
        ├── 行数(4字节) → 每张图片高度
        ├── 列数(4字节) → 每张图片宽度
        └── 像素数据(N字节) → N = 图像数量 × 行数 × 列数
    标签文件格式:
        ├── 魔数(4字节) → 通常为2049
        ├── 标签数量(4字节) → 与图像数量对应
        └── 标签数据(N字节) → 每个字节代表一个标签(0-9)

    """
    
    try:
        # 检查路径是否存在
        if not os.path.exists(mnist_path):
            print(f"路径 {mnist_path} 不存在")
            return None
            
        # 手动读取MNIST数据文件
        def read_images(filename):
            with open(os.path.join(mnist_path, filename), 'rb') as f:
                #1. 读取前4个字节作为魔数,标识文件类型,必须为2051
                magic_number = int.from_bytes(f.read(4), 'big')
                if magic_number != 2051:
                    raise ValueError(f"无效的魔数: {magic_number}, 必须为2051")
                
                #2. 读取接下来的4个字节获取图像总数,int.from_bytes() 将二进的字节转换为整数,big-endian 字节序高位字节在前
                num_images = int.from_bytes(f.read(4), 'big')

                #3. 读取行数,分别读取图像的行数和列数(都是28)
                rows = int.from_bytes(f.read(4), 'big')
                cols = int.from_bytes(f.read(4), 'big')
                
                #4. 读取剩余所有字节(像素数据)，np.frombuffer() 将字节数据直接转换为NumPy数组,每个像素用一个字节即8位(0-255)表示灰度值
                images_data = np.frombuffer(f.read(), dtype=np.uint8)

                #5. 重塑数组为(num_images, rows * cols)的二维数组,共有num_images行,其中一行就是一个28*28=784个像素值，每个像素值是0-255的灰度，即可组成一个28*28的图像
                images_data = images_data.reshape(num_images, rows * cols)
            return images_data
            
        def read_labels(filename):
            with open(os.path.join(mnist_path, filename), 'rb') as f:
                #1. 读取魔法数字,标识文件类型,必须为2049
                magic_number = int.from_bytes(f.read(4), 'big')
                if magic_number != 2049:
                    raise ValueError(f"无效的魔数: {magic_number}, 必须为2049")
                
                #2. 读取标签数量,标识标签总数,必须与图像数量对应
                num_labels = int.from_bytes(f.read(4), 'big')
                
                #3. 读取标签数据,np.frombuffer() 将字节数据直接转换为NumPy数组,每个标签用一个字节即8位表示(0-9)的数值
                labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
            
        # 读取训练数据
        train_images = read_images('train-images.idx3-ubyte')
        train_labels = read_labels('train-labels.idx1-ubyte')
        
        # 转换为DataFrame格式
        df = pd.DataFrame(train_images)
        df.columns = [f'pixel_{i}' for i in range(784)]
        df['label'] = train_labels
        
        # 保存为CSV文件
        df.to_csv(os.path.join(mnist_path, './mnist_train.csv'), index=False)

    except Exception as e:
        print(f"读取MNIST数据时出错: {e}")
        return None

def show_minist(filename):
    # 读取CSV文件
    mnist_df = pd.read_csv(filename)
    
    # 显示前五行数据
    print("MNIST数据集前五行:\n", mnist_df.head())
    
    # 显示前几幅图像
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('MNIST Dataset Sample Images')
    
    for i in range(10):
        # 获取第i行的数据
        row = mnist_df.iloc[i]
        # 提取标签
        label = row['label']
        # 提取像素值并重塑为28x28的图像
        image = row.drop('label').values.reshape(28, 28)
        
        # 确定子图位置
        ax = axes[i//5, i%5]
        # 显示图像
        ax.imshow(image, cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 显示数据集的一些统计信息
    print("\n数据集信息:")
    print(f"数据集大小: {mnist_df.shape}")
    print(f"特征数量: {len(mnist_df.columns)-1}")  # 减去标签列
    print(f"类别分布:\n{mnist_df['label'].value_counts().sort_index()}")

def train_model(train_filename, model_path):
    # 读取CSV文件
    mnist_df = pd.read_csv(train_filename)
    
    # 提取特征和标签
    X_train = mnist_df.drop('label', axis=1)
    y_train = mnist_df['label']

    # 归一化特征值到[0,1]范围,提高模型训练效率
    X_train = X_train / 255.0
    
    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=22)
    
    # 初始化KNN分类器
    knn_model = KNeighborsClassifier(n_neighbors=3)

    # 调参网格搜索与交叉验证
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11,1]}
    grid_search = GridSearchCV(knn_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    #获得最佳参数的模型，用于后续的训练和评估
    best_model=grid_search.best_estimator_
    print(f"最佳参数: {grid_search.best_params_}")
    
    # 训练模型
    best_model.fit(X_train, y_train)

    # 评估模型
    accuracy = best_model.score(X_test, y_test)
    print(f"模型评估的准确率: {accuracy:.4f}")

    # 保存模型    
    joblib.dump(best_model, os.path.join(model_path, 'knn_mnist_model.pkl'))
    print(f"模型已保存至: {os.path.join(model_path, 'knn_mnist_model.pkl')}")

    # 预测
    y_pred = best_model.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    print(f"模型预测后的得到的目标数据，与真实目标数据的准确率:: {acc_score:.4f}")
    # 打印预测结果
    print("预测结果:", y_pred)

    return

def preprocess_mnist_image(image_path):
    """
    将要预测的数据进行，预处理图像以匹配MNIST数据格式
    1.图像预处理: 将任意大小的图像调整为28x28像素
    2.颜色转换: 将彩色图像转换为灰度图像
    3.数据归一化: 将像素值归一化到[0,1]范围，与训练时保持一致
    4.形状匹配: 确保输入数据的特征数量与训练数据一致（784个特征）
    """
    # 读取图像
    image = plt.imread(image_path)
    
    # 如果是彩色图像，转换为灰度图像
    if len(image.shape) == 3:
        # 对于RGB图像，转换为灰度图像,image通常是一个形状为 (height, width, channels) 其中channels数组=3，即RGB三个通道,
        # 每个像素由三个通道值组成，分别对应红 (R)、绿 (G)、蓝 (B) 三个颜色通道
        # image[...,:3]其中:3 表示取最后一维的前 3 个元素，确保只取 RGB 三个通道，其中...表示省略号索引，表示自动处理其他维度，确保只取最后一维的前 3 个元素
        # [0.2989, 0.5870, 0.1140]灰度化权重系数，分别对应红 (R)、绿 (G)、蓝 (B) 三个通道的权重
        # np.dot() 执行的是矩阵乘法（或点积）操作,即灰度值 = R*0.2989 + G*0.5870 + B*0.1140
        image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    
    # 调整图像大小为28x28
    from skimage.transform import resize
    image_resized = resize(image, (28, 28), anti_aliasing=True, mode='constant')
    
    # 转换为0-255范围的整数,
    image_resized = (image_resized * 255).astype(np.uint8)
    
    # 展平为1D数组,并确保形状为(1, 784)
    #image_flat = image_resized.flatten()
    #image_flat = image_resized.reshape(1,784)
    image_flat = image_resized.reshape(1,-1)
    
    # 归一化到[0,1]范围
    image_normalized = image_flat / 255.0
    
    return image_normalized

def predict_image(model_path, test_filename):
    # 加载模型
    knn_model = joblib.load(os.path.join(model_path, 'knn_mnist_model.pkl'))
    print(f"模型已加载自: {os.path.join(model_path, 'knn_mnist_model.pkl')}")
    
    # 预处理图像
    processed_image = preprocess_mnist_image(test_filename)
    
    # 显示预处理后的图像，将展平的数组转换为28x28图像，以便显示出来
    image_2d = (processed_image * 255).astype(np.uint8).reshape(28, 28)
    plt.imshow(image_2d, cmap='gray')
    plt.title(f'预处理后的图像: {test_filename}')
    plt.axis('off')
    plt.show()
    
    # 预测
    y_pred = knn_model.predict(processed_image)
    
    # 打印预测结果
    print(f"模型预测的数字为: {y_pred[0]}")
    
    return y_pred

if __name__ == "__main__":
    #1. 从OpenML下载MNIST数据集，无法访问openml.org
    #mnist_data = load_mnist_sklearn()

    mnist_path = r'd:\python_code\machine_learning\1_KNeighborsClassifier\data'
    model_path = r'd:\python_code\machine_learning\1_KNeighborsClassifier\model'
    test_filename = rf'd:\python_code\machine_learning\1_KNeighborsClassifier\test\22.png'

    #2. 加载本地MNIST数据
    if not os.path.exists(os.path.join(mnist_path, './mnist_train.csv')):
        mnist_dataset_to_csv(mnist_path) 
    
    #3. 显示MNIST数据集
    #show_minist(os.path.join(mnist_path, './mnist_train.csv'))

    #4. 训练模型
    if not os.path.exists(os.path.join(model_path, 'knn_mnist_model.pkl')):
        train_model(os.path.join(mnist_path, './mnist_train.csv'), model_path)

    #5. 预测图像
    predict_image(model_path, test_filename)    


