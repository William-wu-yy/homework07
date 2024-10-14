import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy import stats
from sklearn.decomposition import PCA


training_data = pd.read_excel("C:/Users/WYY/Downloads/TrainingData.xlsx")
test_data = pd.read_excel("C:/Users/WYY/Downloads/TestData.xlsx")


missing_values = training_data.isnull().sum()
print(missing_values)


training_data[7.7] = pd.to_numeric(training_data[7.7], errors='coerce')
training_data[2.2] = pd.to_numeric(training_data[2.2], errors='coerce')


training_data_cleaned = training_data.fillna(training_data.mean())

z_scores_cleaned = stats.zscore(training_data_cleaned.iloc[:, :-1])
outliers_cleaned = (abs(z_scores_cleaned) > 3).any(axis=1)
final_training_data = training_data_cleaned[~outliers_cleaned]


X_train_final = final_training_data.iloc[:, :-1]
y_train_final = final_training_data.iloc[:, -1]


scaler = StandardScaler()
X_train_final_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(test_data)


svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train_final_scaled, y_train_final)


test_predictions = svm_classifier.predict(X_test_scaled)


print(test_predictions[:29], test_predictions.shape)


pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_final_scaled)
svm_classifier_pca = SVC(kernel='linear', random_state=42)
svm_classifier_pca.fit(X_train_pca, y_train_final)


x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))


Z = svm_classifier_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)


plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_final, edgecolors='k', marker='o')
plt.title("SVM Decision Boundary with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()