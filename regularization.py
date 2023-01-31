# 過学習とは、訓練用データに過度に対応してしまい、予測ができなくなること。
# 今回は過学習を抑えるための「正則化」(Regularization)を行い、テストデータの予測の精度を上げるためのチューニングを行う。

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

import pandas as pd

# 使用するデータセットはUCI機械学習リポジトリの「ウィルコンシン肺癌データ」を使用する
# 肺癌が悪性か良性かを予測する
# sklearn.preprocessingでデータセットのラベル[2]の変数M,Bを数字に変換している
df_cancer = pd.read_csv("./datasets/wdbc.data")
datasets = df_cancer

from sklearn.preprocessing import LabelEncoder
x,y = datasets.iloc[:,2:].values, df_cancer.iloc[:,1].values
encoder = LabelEncoder()
y = encoder.fit_transform(y)

#トレインデータとテストデータに分割
# 今回はデータセットの量から、20%をテストデータに割り当てる。
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# データを標準化するために標準偏差と平均値を求める。
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)

x_train_std = sc.transform(x_train)
x_test_std  = sc.transform(x_test)

# 最適なモデルの選択
# 今回もロジスティック回帰を選択
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

# 主成分分析により、入力データから主成分を抽出する
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train_std,y_train)
x_test_pca  = pca.transform(x_test_std)

#モデルに学習させる
clf.fit(x_train_pca,y_train)

from sklearn.metrics import accuracy_score
y_pred = clf.predict(x_test_pca)
print("正則化前: " + str(accuracy_score(y_test, y_pred) * 100) + "%") # ここまでが正則化なしの流れ

# 正則化
clf_regulation = LogisticRegression(penalty='l1',C=0.1,solver='liblinear')
# clf_regulation = LogisticRegression(penalty='l1',C=0.1,solver='saga')
clf_regulation.fit(x_train_pca,y_train)
y_pred_regulation = clf_regulation.predict(x_test_pca)
print("正則化後: " + str(accuracy_score(y_test, y_pred_regulation) * 100) + "%")
