# ライブラリの読み込み
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lifelines
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lifelines import KaplanMeierFitter
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# データの読み込み
# INPUT_DIRにディレクトリを指定
INPUT_DIR = "xxx"

df_train = pd.read_csv(INPUT_DIR + "train.csv")
df_test = pd.read_csv(INPUT_DIR + "test.csv")
data_dic = pd.read_csv(INPUT_DIR + "data_dictionary.csv")# 辞書データ
sample_sub = pd.read_csv(INPUT_DIR + "sample_submission.csv")

# 生存確率を計算する関数
# efs：無イベント生存率
# efs_time：無イベント生存までの時間
def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter() # インスタンスを作成
    kmf.fit(df[time_col], event_observed=df[event_col]) # カプラン・マイヤー推定量にデータをフィット
    survival_probabilities = kmf.survival_function_at_times(df[time_col]).values.flatten()
    censored_mask = df[event_col] == 0
    return survival_probabilities

# 生存確率を目的変数として代入
df_train["target"] = transform_survival_probability(df_train, time_col='efs_time', event_col='efs')

# 不要になったカラムを削除
drop_cols = ["ID", 'efs', 'efs_time']
df_trainainain = df_train.drop(columns=[col for col in drop_cols if col in df_train.columns])
df_test = df_test.drop(columns=[col for col in drop_cols if col in df_test.columns])
df_train.head()

# 欠損値には一律で同じ値を代入
def replace_nulls_with_default(df, float_default=0.0, object_default="Unknown"):
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = df[col].fillna(float_default)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(object_default)
    return df

df_train = replace_nulls_with_default(df_train, float_default=-1.0, object_default="Missing")
df_test = replace_nulls_with_default(df_test, float_default=-1.0, object_default="Missing")

# ojbect型をcategory型に変換
def convert_object_to_category(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    return df

df_train = convert_object_to_category(df_train)
df_test = convert_object_to_category(df_test)

# データフレームをCSVファイルに保存
df_train.to_csv('./input/02/df_train_02.csv')
df_test.to_csv('./input/02/df_test_02.csv')