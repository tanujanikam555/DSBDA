import pandas as pd
import numpy as np

df = pd.read_csv("data2.csv")

df.isnull()

series = pd.isnull(df["math score"])
series

df.notnull()

series1 = pd.notnull(df["math score"])
series1

missing_values = ["Na", "na", "Na", "nA"]
df = pd.read_csv("data2.csv", na_values = missing_values)
ndf=df.fillna(0)

df['math score'] = df['math score'].fillna(df['math score'].mean())
df['math score'] 

df['reading score'] = df['reading score'].fillna(df['reading score'].median())
df['reading score'] 

df['writing score'] = df['writing score'].fillna(df['writing score'].std())
df['writing score'] 

ndf1=df 
ndf1.replace(to_replace = np.nan, value = -99)

ndf1.dropna()
ndf1.dropna(how = 'all')
ndf1.dropna(axis = 1)

new_data = ndf1.dropna(axis = 0, how ='any')
new_data

ndf2=df
ndf2

col = ['math score', 'reading score', 'writing score', 'Placement Score', 'placement offer count']
ndf2.boxplot(col)

from scipy import stats

z = np.abs(stats.zscore(df['math score']))

print(z)

threshold = 0.18
sample_outliers = np.where(z <threshold)
sample_outliers

sorted_rscore= sorted(df['reading score'])
sorted_rscore

q1 = np.percentile(sorted_rscore, 25)

q3 = np.percentile(sorted_rscore, 75)

print(q1,q3)

IQR = q3-q1

lwr_bound = q1-(1.5*IQR)
upr_bound = q3+(1.5*IQR)

print(lwr_bound, upr_bound)

r_outliers = []
for i in sorted_rscore:
	if (i<lwr_bound or i>upr_bound):
		r_outliers.append(i)

print(r_outliers)
