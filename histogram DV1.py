import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv('data8.csv')
dataset

sns.histplot(dataset['Fare'])

sns.histplot(data=dataset,x='Fare',y='Survived')
  
#Optional 
sns.distplot(dataset['Fare'])

sns.distplot(dataset['Fare'], kde=False)

sns.distplot(dataset['Fare'], kde=False, bins=10)

sns.jointplot(x='Age', y='Fare', data=dataset)

sns.pairplot(dataset)

sns.rugplot(dataset['Fare'])

sns.barplot(x='Sex', y='Age', data=dataset)

sns.countplot(x='Fare', data=dataset)

sns.boxplot(x='Survived', y='Fare', data=dataset)

sns.boxplot(x='Sex', y='Age', data=dataset, hue="Survived")

sns.violinplot(x='Sex', y='Age', data=dataset, hue='Survived')

sns.stripplot(x='Sex', y='Age', data=dataset)
