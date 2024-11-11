# Econometrics Week 6 Sample Python Code
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

hmda = pd.read_csv('hmda.csv')

deny = pd.get_dummies(hmda['deny'], dtype='int')
hmda['deny'] = deny['yes']

reg1 = smf.ols('deny ~ pirat', data = hmda).fit(cov_type='HC1')
reg2 = smf.ols('deny ~ pirat + afam', data = hmda).fit(cov_type='HC1')

regtab1 = Stargazer([reg1, reg2])

Func = open('week6_out1.html', 'w+')
Func.write(regtab1.render_html())
Func.close()

webbrowser.open_new('week6_out1.html')

sns.regplot(x = 'pirat', y = 'deny', data = hmda)
plt.show()


reg3 = smf.probit('deny ~ pirat', data = hmda)
reg4 = smf.probit('deny ~ pirat + afam', data = hmda)
reg3res = reg3.fit_regularized()
reg4res = reg4.fit_regularized()

print(reg3res.summary2())
print(reg4res.summary2())

sns.lmplot(x = 'pirat', y = 'deny', data = hmda, logistic = True)
plt.show()

reg5 = smf.logit('deny ~ pirat', data = hmda)
reg6 = smf.logit('deny ~ pirat + afam', data = hmda)
reg5res = reg5.fit_regularized()
reg6res = reg6.fit_regularized()

print(reg5res.summary2())
print(reg6res.summary2())

hmda['lvrat'] = pd.cut(x = hmda['lvrat'], bins = [0, .8, .95, 1],
                       labels = ['low', 'medium', 'high'])

hmda['mhist'].astype('int')
hmda['chist'].astype('int')

reg7 = smf.ols('deny ~ afam + pirat + hirat + lvrat + chist + mhist + phist + insurance + selfemp', data=hmda).fit(cov_type='HC1')
reg8 = smf.logit('deny ~ afam + pirat + hirat + lvrat + chist + mhist + phist + insurance + selfemp', data=hmda).fit(cov_type='HC1')
reg9 = smf.probit('deny ~ afam + pirat + hirat + lvrat + chist + mhist + phist + insurance + selfemp', data=hmda).fit(cov_type='HC1')
reg10 = smf.probit('deny ~ afam + pirat + hirat + lvrat + chist + mhist + phist + insurance + selfemp + single + hschool + unemp', data=hmda).fit(cov_type='HC1')
reg11 = smf.probit('deny ~ afam + pirat + hirat + lvrat + chist + mhist + phist + insurance + selfemp + single + hschool + unemp + condomin + I(mhist==3) + I(mhist==4) + I(chist==3) + I(chist==4) + I(chist==5) + I(chist==6)', data=hmda).fit(cov_type='HC1')
reg12 = smf.probit('deny ~ afam * (pirat + hirat) + lvrat + chist + mhist + phist + insurance + selfemp + single + hschool + unemp', data=hmda).fit(cov_type='HC1')

regtab2 = Stargazer([reg7, reg8, reg9, reg10, reg11, reg12])

Func = open('week6_out2.html', 'w+')
Func.write(regtab2.render_html())
Func.close()

webbrowser.open_new_tab('week6_out2.html')

