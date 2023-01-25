# データセットはもともとパッケージ内にいくつか用意されている
# 今回はiris(日本語で「あやめ」という花)のデータを使って品種を予想します。データセットの中身の説明は後ほどします。
from sklearn.datasets import load_iris
datasets = load_iris()

# [data]には花びら・がく片の長さと幅のデータが格納されている。
xi = datasets.data
# print(datasets['data'])
# targetは答えとなる3つの品種が格納されている。2進数で表記されている。
ti = datasets.target
# print(datasets['target'])
#他にも[feature_name]には特徴量、[target_names]には品種の名前が格納されています。

##### ================================
#            データの分割
##### ================================
# 訓練用とテスト用のデータに分割します。
# たとえでいうなら、受験の問題集を繰り返し勉強するための分と、勉強したことができているかのテスト用の範囲に分ける感じです。(youtubeの動画の引用)
# train_test_split関数はランダムな配列が返却される。test_sizeで割合を指定することが出来る。今回は7:3のする。
from sklearn.model_selection import train_test_split
xi_train, xi_test, ti_train, ti_test = train_test_split(xi,ti,test_size=0.3,random_state=0)

##### ================================
#        データの可視化(おまけ)
##### ================================
import pandas as pd
df = pd.DataFrame(xi_train,columns=datasets.feature_names)
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
# このパッケージは見やすくするためらしい。(初心者向けってことかな？)
# 今回は図に色がついているが、品種ごとに色を付けてくれているらしい。
import mglearn
grr = scatter_matrix(df,c=ti_train,figsize=(15,15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()


##### ================================
#             アルゴリズムを選択
##### ================================
# 今回はMLPClassifierを使って予想する。
# ニューラルネットワークの分類に入り、multi-layer perceptron→多層パーセプロトンと呼ばれ、回帰問題や分類問題の両方に使用できる。
from sklearn.neural_network import MLPClassifier

##### ================================
#         モデルを作成・学習
##### ================================
# max_iterは、最大反復回数。反復回数が少ないと値が収束せずに終わることがあるためエラーが出る時がある。
clf = MLPClassifier(max_iter=1000)
# 学習させる。学習させるときは訓練用データの入力データ(なにで予測するか)と出力データ(なにを予測するか)を引数にする
clf.fit(xi_train,ti_train)

# 訓練用データでスコアを見る
# Pythonは文字と数字の連結はできないためキャスト変換を行う必要がある。(数値→数字 / 数字→数値　など)
score_train = clf.score(xi_train,ti_train)
print("訓練用データのMLP-clfスコア: " + str(score_train))

# テスト用データでスコアを見る
# 基本的にはテスト用データの方がスコアは低くなる傾向がある(やったかな？普通そうよな。)
score_test = clf.score(xi_test,ti_test)
print("テスト用データのMLP-clfスコア: " + str(score_test))

#作成したモデルを使用して予測する
predict = clf.predict(xi_test)
print("予測した値: " + str(predict))
print("答え: " + str(ti_test))

# ================================
#  　　違うモデルでも検証
# ================================
# SVCでも分類・回帰問題を予測することが出来ます。
from sklearn.svm import SVC

#モデルを生成・学習の流れは上と同じ
clf = SVC()
clf.fit(xi_train,ti_train)
score_train = clf.score(xi_train,ti_train)
score_test = clf.score(xi_test,ti_test)
print("訓練用データのSVCスコア: " + str(score_train))
print("テスト用データのSVCスコア: " + str(score_test))

predict = clf.predict(xi_test)
print("予測した値: " + str(predict))
print("答え: " + str(ti_test))





