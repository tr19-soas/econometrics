# Econometrics Week 3 Sample Python Code
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.iolib.summary2 import summary_col

df = pd.read_csv('caschools.csv')
df.head()

df['str'] = (df['students']/df['teachers'])
df['score'] = ((df['read']+df['math'])/2)

model1 = smf.ols('score ~ income', data =df).fit()
print(model1.summary())

sns.regplot(x='income', y='score', data=df)
plt.xticks(range(10, 31, 5))
plt.yticks(range(600, 721, 20))
plt.show()

df['income2'] = (df['income']**2)

model2 = smf.ols('score ~ income + I(income2)', data=df).fit(cov_type='HC1')
print(model2.summary())

sns.regplot(x='income2', y='score', data=df)
plt.xticks(range(10, 31, 5))
plt.yticks(range(600, 721, 20))
plt.show()

poly = PolynomialFeatures(degree=3, include_bias=False)

income = df['income'].values
income = np.array(income)
income = income.reshape(-1,1)

score = df['score'].values

income_poly = poly.fit_transform(income)
lin2 = LinearRegression()
lin2.fit(income_poly, score)
model3 = lin2.predict(income_poly)

plt.scatter(income,score, c="blue")
plt.plot(income, model3, c="red")
plt.show()

mod1 = model1.fit()
mod2 = model2.fit()
mod3 = model3

dfoutput = summary_col([mod1,mod2,mod3],stars=True)
print(dfoutput)
