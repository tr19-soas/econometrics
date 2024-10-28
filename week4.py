# Econometrics Week 4 Sample Python Code
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
from stargazer.stargazer import Stargazer
import webbrowser

df = pd.read_csv('caschools.csv')
df.head()

df['str'] = (df['students']/df['teachers'])
df['score'] = ((df['read']+df['math'])/2)

df['log_income'] = np.log(df['income'])
df['log_score'] = np.log(df['score'])
df['log_income2'] = (df['log_income']**2)
df['log_income3'] = (df['log_income']**3)

reg1 = smf.ols('score ~ log_income', data = df).fit(cov_type='HC1')
reg2 = smf.ols('log_score ~ income', data = df).fit(cov_type='HC1')
reg3 = smf.ols('log_score ~ log_income', data = df).fit(cov_type='HC1')
reg4 = smf.ols('score ~ log_income + I(log_income2) + I(log_income3)', data = df).fit(cov_type='HC1')

stargazer = Stargazer([reg1, reg2, reg3, reg4])

Func = open('week4_out1.html','w+')
Func.write(stargazer.render_html())
Func.close()

webbrowser.open_new_tab('week4_out1.html')

sns.regplot(x='log_income', y='score', data=df)
plt.show()

sns.regplot(x='income', y='log_score', data=df)
plt.show()

sns.regplot(x='log_score', y='log_income', data=df)
plt.show()


