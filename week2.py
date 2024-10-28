# Econometrics Week 2 Sample Python Code
# By Taylor D.H. Rockhill
# Goldsmiths, University of London, 2024
# Creative Commons Licence

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import math
import csv

df = pd.read_csv('caschools.csv')
df.head()

df['str'] = (df['students']/df['teachers'])
df['score'] = ((df['read']+df['math'])/2)

sns.scatterplot(x='str', y='score', data=df)
plt.xticks(range(10, 31, 5))
plt.yticks(range(600, 721, 20))
plt.show()

df['str'].corr(df['score'])

sns.regplot(x='str', y='score', data=df)
plt.xticks(range(10, 31, 5))
plt.yticks(range(600, 721, 20))
plt.show()

model = smf.ols('score ~ str', data=df).fit()
print(model.summary())

RSS = np.sum(model.resid**2)
RSS

anova_table = sm.stats.anova_lm(model)
anova_table
