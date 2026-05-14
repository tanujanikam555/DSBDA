import numpy as np
import pandas as pd

df = pd.read_csv("data10.csv")

df.columns = ["col1","col2","col3","col4","col5"]

column = len(list(df))
column

df.info()

np.unique(df["col5"])

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

ig, axes = plt.subplots(2, 2, figsize=(16, 8))

axes[0,0].set_title("Distribution of First Column")
axes[0,0].hist(df["col1"]);

axes[0,1].set_title("Distribution of Second Column")
axes[0,1].hist(df["col2"]);

axes[1,0].set_title("Distribution of Third Column")
axes[1,0].hist(df["col3"]);

axes[1,1].set_title("Distribution of Fourth Column")
axes[1,1].hist(df["col4"]);

data_to_plot = [df["col1"],df["col2"],df["col3"],df["col4"]]

sns.set_style("whitegrid")

# Creating a figure instance
fig = plt.figure(1, figsize=(12,8))

# Creating an axes instance
ax = fig.add_subplot(111)

# Creating the boxplot
bp = ax.boxplot(data_to_plot)

sns.boxplot(x='col1', y='col2', data=df)

sns.boxplot(x='col3', y='col4', data=df)