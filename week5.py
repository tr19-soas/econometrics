# Econometrics Week 5 Sample Python Code
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
from stargazer.stargazer import Stargazer
import webbrowser
from linearmodels import PanelOLS
from linearmodels import RandomEffects


df = pd.read_csv('fatalities.csv')
df.head()
df = df.set_index(['state', 'year'])

df['rate'] = ((df['fatal']/df['pop'])*10000)

fat1982 = df.query('year == 1982')
fat1988 = df.query('year == 1988')

reg1 = smf.ols('rate ~ beertax', data = fat1982).fit(cov_type='HC1')
reg2 = smf.ols('rate ~ beertax', data = fat1988).fit(cov_type='HC1')

stargazer1 = Stargazer([reg1, reg2])

Func = open('week5_out1.html','w+')
Func.write(stargazer1.render_html())
Func.close()

webbrowser.open_new_tab('week5_out1.html')

sns.regplot(x='rate',y='beertax', data=fat1982)
plt.show()

sns.regplot(x='rate', y='beertax', data=fat1988)
plt.show()


diffrate = fat1988['rate'] - fat1982['rate']
difftax = fat1988['beertax'] - fat1982['beertax']

df2 = pd.DataFrame([[diffrate, difftax]],columns=['diffrate','difftax'])

reg3 = smf.ols('diffrate ~ difftax', data=df2).fit(cov_type='HC1')
reg4 = smf.ols('diffrate ~ 0 + difftax', data=df2).fit(cov_type='HC1')

stargazer2 = Stargazer([reg3, reg4])

Func = open('week5_out2.html', 'w+')
Func.write(stargazer2.render_html())
Func.close()

webbrowser.open_new_tab('week5_out2.html')

sns.regplot(x='difftax', y='diffrate', data=df2)
plt.show()

