import pandas as pd
import numpy as np

df = pd.read_csv("data1.csv")

df.dtypes

df.info()

df.describe()

df.isnull()

df.shape

df.size

df.index

df.head()

df.tail()

df.tail(n=5)

df.columns

df.columns.values

df.iloc[5]

df[0:3]

df.iloc[:,:]

df.iloc[:,3]

df.iloc[1,1]

df['Sepal Length']

cols_2_4=df.columns[2:4]
df[cols_2_4]

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

x=df.iloc[:,:4]
x_scaled = min_max_scaler.fit_transform(x)
df_normalized = pd.DataFrame(x_scaled)
df_normalized

df['Species'].unique()
label_encoder = preprocessing.LabelEncoder()
df['Species']= label_encoder.fit_transform(df['Species'])
df['Species'].unique()