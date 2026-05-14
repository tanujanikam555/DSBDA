import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv('data9.csv')
dataset

sns.boxplot(x='Sex', y='Age', data=dataset)

sns.boxplot(x='Sex', y='Age', data=dataset, hue="Survived")

#Optional 
sns.boxplot(x="Age", y="Fare", hue="Survived",data=dataset,linewidth=3)

sns.boxplot(x='Sex', y='Age', data=dataset, notch=1)

sns.boxplot(x='Sex', y='Age', data=dataset, notch=1, bootstrap=5)

sns.boxplot(data=dataset, orient="v")

sns.boxplot(data=dataset, orient="h")