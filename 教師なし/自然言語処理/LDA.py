"""主成分分析よりも精度がよく、次元を減らすLDAを使用して予測する。
LDAは分類するクラスの分離が最適化するよう最適のサブ空間を決める手法
LDAを適用するのには条件がある。
1. データが正規分布している。
2. 各クラスが同じ共分散行列を持つ。
3. 変量が統計的に互いに独立している。
"""
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

#データセットの準備
# 今回はUCI機械学習のリポジトリにあるワインデータを使用する。
# このデータセットは178のデータからなり、説明変数は13の変量からなる。
# 予測するものはあやめのときと同じく、ワインの種類を求める。
df_wine = pd.read_csv("./../../datasets/wine.data")
datasets = df_wine
x,y = datasets.iloc[:,1:].values, df_wine.iloc[:,0].values

# トレーニングデータとテストデータに分割する
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

# データを標準化するために標準偏差と平均値を求める。
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)

x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

# 最適なクラスを選択する
# 今回も主成分分析のときと同じくロジスティック回帰を選択
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

# LDAを使用して新しい変量を抽出する(主成分分析と異なる部分)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
x_train_lda = lda.fit_transform(x_train_std,y_train)
x_test_lda  = lda.transform(x_test_std)

#学習させる
clf.fit(x_train_lda,y_train)

#予測する
from sklearn.metrics import accuracy_score
y_pred = clf.predict(x_test_lda)
print(str(accuracy_score(y_test, y_pred) * 100) + "%")

#結果をプロット
plot_decision_regions(x_train_lda, y_train, classifier=clf)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()