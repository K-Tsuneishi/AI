import pandas as pd
import inspect
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

#データセットの準備
datasets = fetch_california_housing()

#pd使う時
housing_df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
housing_df['Price'] = datasets.target

#取り出した値は同じ(はず)
print(housing_df['Price'])
print(datasets.target)

# データの標準化→特徴量の比率をそろえることが出来る
std = StandardScaler()
data_std = std.fit_transform(housing_df)
pd.DataFrame(data_sth)
