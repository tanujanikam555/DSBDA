import pandas as pd
import numpy as np
df = pd.read_csv("data3.csv")

df.info()
df.describe()

from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc_df = pd.DataFrame(enc.fit_transform(df[['Genre']]).toarray())
enc_df

df1 = df.join(enc_df)
df1

del df1['Genre']
df1

df1.mean()
df1.mean(axis=0)        # To find mean column wise
df1.mean(axis=1)        # To find mean row wise
df1.mean(axis=1)[0:4]   # To find mean row wise
df1.loc[:,'Age'].mean()  #To find mean of specific column

df1.median()

df1.mode()

df1.std()

df1.min()

df1.max()

df.groupby(['Genre'])['Age'].mean()

df.groupby(['Genre'])['Annual Income'].max()