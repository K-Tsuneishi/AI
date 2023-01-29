#!/usr/bin/env python
# -*- coding:utf-8 -*-

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


# 多数の変数(説明変数)を、より少ない指標や合成変数に要約する手法。データの次元を削減するために用いられる。
import pandas as pd

#データセットの準備
# 今回はUCI機械学習のリポジトリにあるワインデータを使用する。
# このデータセットは178のデータからなり、説明変数は13の変量からなる。
# 予測するものはあやめのときと同じく、ワインの種類を求める。
df_wine = pd.read_csv("./wine.data")
datasets = df_wine
x,y = datasets.iloc[:,1:].values, df_wine.iloc[:,0].values

# 訓練用データとテスト用データに分割
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

# 今回もStandardScaler関数でデータを標準化する。
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#標準化するための平均値と標準偏差を求める。求めるのはトレーニングデータだけを使用して求める。
sc.fit(x_train)

# トレーニングデータとテストデータを標準化する。
x_train_std = sc.transform(x_train)
x_test_std  = sc.transform(x_test)


#最適なモデルを選択する。今回はロジスティック回帰を使用して予測する。
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

# 主成分分析で、上位２つの主成分を抽出する。
# 主成分を抽出することができるPCA関数を用いる
from sklearn.decomposition import PCA
pca = PCA(n_components=2) #n_componentsで抽出する数を指定する。
x_train_pca = pca.fit_transform(x_train_std)
x_test_pca = pca.transform(x_test_std)

#モデルに学習させる
clf.fit(x_train_pca,y_train)

from sklearn.metrics import accuracy_score
y_pred = clf.predict(x_test_pca)
print(str(accuracy_score(y_test, y_pred) * 100) + "%")

# 結果をプロット
plot_decision_regions(x_test_pca, y_test, classifier=clf)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
