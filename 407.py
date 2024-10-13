import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import scipy.io as sio
from matplotlib.colors import ListedColormap

# 1. 读取 .mat 文件中的数据
data_train = sio.loadmat('C:/Users/WYY/Downloads/data_train.mat')['data_train']
label_train = sio.loadmat('C:/Users/WYY/Downloads/label_train.mat')['label_train'].ravel()
data_test = sio.loadmat('C:/Users/WYY/Downloads/data_test.mat')['data_test']


# 2. 初始化分类器：朴素贝叶斯和贝叶斯决策规则 (QDA)
nb_classifier = GaussianNB()
qda_classifier = QDA()

# 3. 训练 QDA 分类器并打印参数
print("Bayes Decision Rule (QDA) Parameters:")
qda_classifier.fit(data_train, label_train)
print("Means:\n", qda_classifier.means_)
# 尝试打印协方差矩阵
try:
    print("Covariance Matrices:")
    for i, cov in enumerate(qda_classifier.covariances_):
        print(f"Class {i}:\n", cov)
except AttributeError:
    print("Falling back to manual covariance calculation.")
    unique_labels = np.unique(label_train)
    for label in unique_labels:
        class_data = data_train[label_train == label]
        cov_matrix = np.cov(class_data, rowvar=False)
        print(f"Class {label} Covariance Matrix:\n{cov_matrix}")

# 训练 Naive Bayes 分类器并打印参数
print("\nNaive Bayes Parameters:")
nb_classifier.fit(data_train, label_train)
print("Class-wise Variances:\n", nb_classifier.var_)
# 3. 训练分类器
nb_classifier.fit(data_train, label_train)
qda_classifier.fit(data_train, label_train)

# 4. 预测测试数据
nb_predictions = nb_classifier.predict(data_test)
qda_predictions = qda_classifier.predict(data_test)

# 5. 绘制测试数据的分类结果（仅用前两个特征可视化）
x_test = data_test[:, 0]  # 第一个特征
y_test = data_test[:, 1]  # 第二个特征


print("\nNaive Bayes Predictions:")
print(nb_predictions)

print("\nQDA Predictions:")
print(qda_predictions)



# 创建决策边界的网格
x_min, x_max = data_train[:, 0].min() - 1, data_train[:, 0].max() + 1
y_min, y_max = data_train[:, 1].min() - 1, data_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 使用训练集和测试集的前两个特征
data_train_2d = data_train[:, :2]  # 只使用前两个特征
data_test_2d = data_test[:, :2]    # 只使用前两个特征

# 重新训练分类器
nb_classifier.fit(data_train_2d, label_train)
qda_classifier.fit(data_train_2d, label_train)

# 预测网格点
Z_nb = nb_classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_qda = qda_classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
# 设置颜色映射
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ['#FF0000', '#0000FF']

# 6. 绘制分类结果和决策边界
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 5. 绘制分类结果和决策边界
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 朴素贝叶斯的决策边界 (左图)
ax1.contourf(xx, yy, Z_nb, cmap=cmap_light, alpha=0.6)
ax1.scatter(data_train[:, 0], data_train[:, 1], c=label_train, cmap=ListedColormap(cmap_bold), edgecolor='k', s=100)
ax1.scatter(data_test[:, 0], data_test[:, 1], color='green', marker='o', s=150)
ax1.set_title('Naive Bayes Classification', fontsize=16)
ax1.set_xlabel('Feature 1', fontsize=14)
ax1.set_ylabel('Feature 2', fontsize=14)
ax1.text(0.5, -0.25, '(a)', transform=ax1.transAxes, ha='center', va='center', fontsize=16)
ax1.grid(True)

# 贝叶斯决策规则 (QDA) 的决策边界 (右图)
ax2.contourf(xx, yy, Z_qda, cmap=cmap_light, alpha=0.6)
ax2.scatter(data_train[:, 0], data_train[:, 1], c=label_train, cmap=ListedColormap(cmap_bold), edgecolor='k', s=100)
ax2.scatter(data_test[:, 0], data_test[:, 1], color='green', marker='o', s=150)
ax2.set_title('Bayes Decision Rule ', fontsize=16)
ax2.set_xlabel('Feature 1', fontsize=14)
ax2.set_ylabel('Feature 2', fontsize=14)
ax2.text(0.5, -0.25, '(b)', transform=ax2.transAxes, ha='center', va='center', fontsize=16)
ax2.grid(True)


plt.subplots_adjust(bottom=0.2)
plt.show()