# k近傍法(KNN)とは、教師データを取り込み、予測したいものの近くになる教師データをもとに予測する手法
# 予測するデータの近くのデータを参照して予測→近くのデータと遠くのデータの重み付けは手動で行う

#今回はUCIのウィスコンシン肺癌データを使用して、悪性か良性かの分類を行う

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

# sklearn.preprocessingでデータセットのラベル[2]の変数M,Bを数字に変換している
df_cancer = pd.read_csv("./../../datasets/wdbc.data",header=None)
datasets = df_cancer

from sklearn.preprocessing import LabelEncoder
x,y = datasets.loc[:,2:].values, datasets.loc[:,1].values
le = LabelEncoder()
y = le.fit_transform(y)

# テストデータとトレーニングに分割
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

#データを標準化するために標準偏差と平均値を求める
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std  = sc.transform(x_test)

#最適なモデルを選択する
# 今回はKNNの練習なのでKneighborsClassifierを選択する
# n_neighborsは近傍オブジェクト(ちかくのデータの数),pはミンコフスキー距離で2はユーグリッド距離を指定
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')

#主成分分析で2つの主成分を抽出する
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train_std,y_train)
x_test_pca  = pca.transform(x_test_std)

# 抽出されたトレーニングデータを使用して学習
kn.fit(x_train_pca,y_train)

#モデルを評価
from sklearn.metrics import accuracy_score
y_pred = kn.predict(x_test_pca)
print(str(accuracy_score(y_test, y_pred) * 100) + "%")


plot_decision_regions(x_train_pca, y_train, classifier=kn)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

