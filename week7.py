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
from stargazer.stargazer import Stargazer
import webbrowser
import linearmodels.iv.model as lm

cigarettes = pd.read_csv('cigarettessw.csv')
cigarettes = sm.add_constant(cigarettes)

cigarettes['log_packs'] = np.log(cigarettes['packs'])
cigarettes['log_price'] = np.log(cigarettes['price'])

cig1995 = cigarettes[cigarettes['year'] == 1995]

reg1 = smf.ols('log_packs ~ log_price', data=cig1995).fit()
print(reg1.summary())

sns.regplot(x = 'log_price', y = 'log_packs', data = cig1995)
plt.show()

cigarettes['rprice'] = (cigarettes['price']/cigarettes['cpi'])
cigarettes['stax'] = ((cigarettes['taxs'] - cigarettes['tax'])/cigarettes['cpi'])
cigarettes['rincome'] = (cigarettes['income']/cigarettes['population']/cigarettes['cpi'])
cigarettes['cigtax'] = (cigarettes['tax']/cigarettes['cpi'])
cigarettes['log_rprice'] = np.log(cigarettes['rprice'])
cigarettes['log_rincome'] = np.log(cigarettes['rincome'])


cig1985 = cigarettes[cigarettes['year'] == 1985]
cig1995 = cigarettes[cigarettes['year'] == 1995]

diff = pd.DataFrame()

diff['demdiff'] = cig1995['log_packs'].values - cig1985['log_packs'].values 
diff['pricediff'] = cig1995['log_rprice'].values - cig1985['log_rprice'].values
diff['incomediff'] = cig1995['log_rincome'].values - cig1985['log_rincome'].values
diff['cigtaxdiff'] = cig1995['cigtax'].values - cig1985['cigtax'].values
diff['staxdiff'] = cig1995['stax'].values - cig1985['stax'].values

diff = sm.add_constant(diff)

reg2 = smf.ols('np.log(price) ~ stax', data = cig1995).fit()
print(reg2.summary())

reg2pre = reg2.fittedvalues.values

reg3 = smf.ols('np.log(packs) ~ reg2pre', data = cig1995).fit(cov_type = 'HC1')
print(reg3.summary())

reg4 = lm.IV2SLS(dependent = cig1995['log_packs'], exog = cig1995['log_price'],
                 endog = cig1995['const'], instruments = cig1995['stax']).fit(cov_type='robust')
print(reg4)

regtab1 = Stargazer([reg3, reg4])
regtab1.custom_columns(['OLS', '2SLS'], [1,1])

Func = open('week7_out1.html', 'w+')
Func.write(regtab1.render_html())
Func.close()

webbrowser.open_new('week7_out1.html')

reg5 = lm.IV2SLS(dependent = cig1995['log_packs'], exog = cig1995[['log_rincome', 'log_rprice']], 
                 endog = cig1995['const'], instruments = cig1995['stax']).fit(cov_type = 'robust')
print(reg5)

reg6 = lm.IV2SLS(dependent = cig1995['log_packs'], exog = cig1995[['log_rincome', 'log_rprice']],
                 endog = cig1995['const'], instruments = cig1995[['stax', 'cigtax']]).fit(cov_type='robust')
print(reg6)

regtab2 = Stargazer([reg5, reg6])
regtab2.add_line('IVs', ['stax', 'stax, cigtax'])

Func = open('week7_out2.html', 'w+')
Func.write(regtab2.render_html())
Func.close()

webbrowser.open_new_tab('week7_out2.html')

reg7 = lm.IV2SLS(dependent = diff['demdiff'], exog = diff[['pricediff', 'incomediff']],
                 endog = diff['const'], instruments = diff['staxdiff']).fit(cov_type = 'robust')
reg8 = lm.IV2SLS(dependent = diff['demdiff'], exog = diff[['pricediff', 'incomediff']],
                 endog = diff['const'], instruments = diff['cigtaxdiff']).fit(cov_type = 'robust')
reg9 = lm.IV2SLS(dependent = diff['demdiff'], exog = diff[['pricediff', 'incomediff']],
                 endog = diff['const'], instruments = diff[['staxdiff', 'cigtaxdiff']]).fit(cov_type = 'robust')

regtab3 = Stargazer([reg7, reg8, reg9])
regtab3.add_line('IVs', ['staxdiff', 'cigtaxdiff', 'staxdiff, cigtaxdiff'])

Func = open('week7_out3.html', 'w+')
Func.write(regtab3.render_html())
Func.close()

webbrowser.open_new_tab('week7_out3.html')

reg9res = reg9.resids

reg10 = smf.ols('pricediff ~ incomediff + staxdiff', data = diff).fit(cov_type = 'HC1')
reg11 = smf.ols('pricediff ~ incomediff + cigtaxdiff', data = diff).fit(cov_type = 'HC1')
reg12 = smf.ols('pricediff ~ incomediff + cigtaxdiff + staxdiff', data = diff).fit(cov_type = 'HC1')

regtab4 = Stargazer([reg10, reg11, reg12])

Func = open('week7_out4.html', 'w+')
Func.write(regtab4.render_html())
Func.close()

webbrowser.open_new_tab('week7_out4.html')

reg13 = smf.ols('reg9res ~ incomediff + staxdiff + cigtaxdiff', data = diff).fit(cov_type = 'HC1')
print(reg13.summary())
