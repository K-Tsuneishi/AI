# 分類のayameのファイルはもっと細かく説明しています

#今回はscikit-learnを使って回帰問題を解く
#使用するデータセットは、カリフォルニアの住宅価格のデータセット
from sklearn.datasets import fetch_california_housing
datasets = fetch_california_housing()

# 今回のデータセットの[data]には、住環境の情報が8つ入っている。
# [0]:生体所得と中央値 [1]:家の築年数 [2]:部屋の平均数 [3]:寝室の平均数 [4]:居住人数の合計\
#  [5]:世帯人数の平均 [6]:平均緯度 [7]:平均経度
xi = datasets.data
ti = datasets.target

# ===========================
#     データセットを分割
# ===========================
from sklearn.model_selection import train_test_split

# 訓練用データセットとテスト用データセットへの分割（訓練用70%、テスト用30%）
xi_train,xi_test,ti_train,ti_test = train_test_split(xi,ti,test_size=0.3)

# ===========================
#     モデルの決定
# ===========================
# 今回はLinearRegressionを使用して予測するモデルを作成します。
from sklearn.linear_model import LinearRegression
clf = LinearRegression()

# モデルの学習
clf.fit(xi_train,ti_train)

score_train = clf.score(xi_train,ti_train)
score_test = clf.score(xi_test,ti_test)
print("訓練用データのLinearRegressionスコア: " + str(score_train))
print("テスト用データのLinearRegressionスコア: " + str(score_test))

predict = clf.predict(xi_test)
print("予想した値:" + str(predict))
print("実際の値: " + str(ti_test))

# 次にMLPRegressorを使用して予測していきます
from sklearn.neural_network import MLPRegressor

# モデルの作成・学習
clf = MLPRegressor()
clf.fit(xi_test,ti_test)

score_train = clf.score(xi_train,ti_train)
score_test = clf.score(xi_test,ti_test)
print("訓練用データのMLPRegressorスコア: " + str(score_train))
print("テスト用データのMLPRegressorスコア: " + str(score_test))

predict = clf.predict(xi_test)
print("予想した値:" + str(predict))
print("実際の値: " + str(ti_test))
