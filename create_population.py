# -*- coding: utf-8 -*-
"""
@author: ijlee
This code is an example of generating N numbers of virtual players 
with sets of cmu, csigma, nu, delta, and cl,
according to the distributions modeled in the following paper:
Automated Playtesting with a Cognitive Model of Sensorimotor Coordination, ACM MM21
"""

import scipy.stats as st
import pandas as pd
import numpy as np

def generate_players(population):
    # generate players
    cmu = {'func': st.gennorm, 'title': 'cmu', #'mstr': '$c_{mu}$',
        'beta': 0.4779771330429379, 'loc': 0.28120999989528495, 'scale': 0.01392626664919034}
    csigma = {'func': st.gennorm, 'title': 'cssigma', #'mstr': '',
        'beta': 0.5052910730545627, 'loc': 0.09375599982793718, 'scale': 0.004389738464077227}
    nu = {'func': st.johnsonsu, 'title': 'nu', #'mstr': '',
        'a': -0.0832213149947845, 'b': 0.4734376059805999, 'loc': 680.4992397676942, 'scale': 10.48370381919491}
    delta = {'func': st.johnsonsu, 'title': 'delta', #'mstr': '',
        'a': 0.36087454780387324, 'b': 1.0330937541014005, 'loc': 0.5088284177874871, 'scale': 0.08697729157828178}
    cl = {'func': st.johnsonsu, 'title': 'cl', #'mstr': '',
        'a': 0.6518096793779063, 'b': 0.9066785182918338, 'loc': 0.520672091388702, 'scale': 0.08421354625930168}

    playernum = population

    players = pd.DataFrame(columns=['cmu', 'csigma', 'nu', 'delta', 'cl'])

    for i in [cmu, csigma, nu, delta, cl]: #, csigma

        if i['title'] == 'cmu':
            x = i['func'].rvs(beta=i['beta'] , loc=i['loc'], scale=i['scale'], size=playernum)
            x = x[(x >= 0) & (x <=0.5)]
            while len(x) < playernum:
                gap = playernum - len(x)
                x = np.append(x, i['func'].rvs(beta=i['beta'] , loc=i['loc'], scale=i['scale'], size=gap) )
                x = x[(x >= 0) & (x <=0.5)]
        elif i['title'] == 'csigma':
            x = i['func'].rvs(beta=i['beta'] , loc=i['loc'], scale=i['scale'], size=playernum)
            x = x[(x >= 0)]
            while len(x) < playernum:
                gap = playernum - len(x)
                x = np.append(x, i['func'].rvs(beta=i['beta'] , loc=i['loc'], scale=i['scale'], size=gap) )
                x = x[(x >= 0)]
        elif any(i['title'] in x for x in ['csigma', 'nu', 'delta']):
            x = i['func'].rvs(a=i['a'], b=i['b'], loc=i['loc'], scale=i['scale'], size=playernum)
            x = x[(x >= 0)]
            while len(x) < playernum:
                gap = playernum - len(x)
                x = np.append(x, i['func'].rvs(a=i['a'], b=i['b'], loc=i['loc'], scale=i['scale'], size=gap) )
                x = x[(x >= 0)]

        players[i['title']] = x

    return players.values.tolist()
