# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:13:19 2024

@author: jwesse03
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.patches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
import matplotlib as mpl
from matplotlib.path import Path
import math, bisect
import matplotlib.colors as mcolors
from pandas.plotting import parallel_coordinates
import os
from pandas.api.types import CategoricalDtype
from matplotlib.ticker import PercentFormatter
import itertools
import fastparquet, pyarrow

### GENERAL INFORMATION / VARIABLES ###
figpath = './'
datapath = 'figure_data/'
sc_names = pd.read_csv('figure_data/scenario_names.csv')
mercosur = ['Argentina','Brazil','Chile','Paraguay','Uruguay']
periods = [2020,2025,2030,2035,2040,2045,2050]

### INPUTS AND DATA PREP ###

#%% SCENARIOS
sc1,sc2,sc3,sc4 = ['prm15_wcost_re1_c2p1_co2_cut90p_t1','prm15_wcost_re2_c2p1_co2_cut90p_t1',
                   'prm15_wcost_re3_c2p1_co2_cut90p_t1','prm15_wcost_re4_c2p1_co2_cut90p_t1',
                   'prm15_wcost_re5_c2p1_co2_cut90p_t1','prm15_wcost_re6_c2p1_co2_cut90p_t1',
                   'prm15_wcost_re7_c2p1_co2_cut90p_t1','prm15_wcost_re8_c2p1_co2_cut90p_t1',
                   'prm15_wcost_re9_c2p1_co2_cut90p_t1','prm15_wcost_re10_c2p1_co2_cut90p_t1'],\
                   ['prm15_wcost_re11_c2p1_co2_cut90p_t1','prm15_wcost_re12_c2p1_co2_cut90p_t1',
                    'prm15_wcost_re13_c2p1_co2_cut90p_t1','prm15_wcost_re14_c2p1_co2_cut90p_t1',
                    'prm15_wcost_re15_c2p1_co2_cut90p_t1','prm15_wcost_re16_c2p1_co2_cut90p_t1',
                    'prm15_wcost_re17_c2p1_co2_cut90p_t1','prm15_wcost_re18_c2p1_co2_cut90p_t1',
                    'prm15_wcost_re19_c2p1_co2_cut90p_t1','prm15_wcost_re20_c2p1_co2_cut90p_t1'],\
                    ['prm15_wcost_re1_c2p1_t1','prm15_wcost_re2_c2p1_t1',
                     'prm15_wcost_re3_c2p1_t1','prm15_wcost_re4_c2p1_t1',
                     'prm15_wcost_re5_c2p1_t1','prm15_wcost_re6_c2p1_t1',
                     'prm15_wcost_re7_c2p1_t1','prm15_wcost_re8_c2p1_t1',
                     'prm15_wcost_re9_c2p1_t1','prm15_wcost_re10_c2p1_t1'],\
                     ['prm15_wcost_re11_c2p1_t1','prm15_wcost_re12_c2p1_t1',
                      'prm15_wcost_re13_c2p1_t1','prm15_wcost_re14_c2p1_t1',
                      'prm15_wcost_re15_c2p1_t1','prm15_wcost_re16_c2p1_t1',
                      'prm15_wcost_re17_c2p1_t1','prm15_wcost_re18_c2p1_t1',
                      'prm15_wcost_re19_c2p1_t1','prm15_wcost_re20_c2p1_t1']  
sc1_suffix, sc2_suffix, sc3_suffix, sc4_suffix = 'trd_1axisPV_cut90p','trd_fixedPV_cut90p','trd_1axisPV_no_target','trd_fixedPV_no_target'
scs = sc3 + sc4 + sc1 + sc2
scs1 = ['prm15_wcost_re1_c2p1_tn','prm15_wcost_re2_c2p1_tn','prm15_wcost_re3_c2p1_tn','prm15_wcost_re4_c2p1_tn',\
       'prm15_wcost_re5_c2p1_tn','prm15_wcost_re6_c2p1_tn','prm15_wcost_re7_c2p1_tn','prm15_wcost_re8_c2p1_tn',\
       'prm15_wcost_re9_c2p1_tn','prm15_wcost_re10_c2p1_tn','prm15_wcost_re11_c2p1_tn','prm15_wcost_re12_c2p1_tn',
       'prm15_wcost_re13_c2p1_tn','prm15_wcost_re14_c2p1_tn','prm15_wcost_re15_c2p1_tn','prm15_wcost_re16_c2p1_tn',\
       'prm15_wcost_re17_c2p1_tn','prm15_wcost_re18_c2p1_tn','prm15_wcost_re19_c2p1_tn','prm15_wcost_re20_c2p1_tn',\
       'prm15_wcost_re1_c2p1_co2_cut90p_tn','prm15_wcost_re2_c2p1_co2_cut90p_tn','prm15_wcost_re3_c2p1_co2_cut90p_tn',\
       'prm15_wcost_re4_c2p1_co2_cut90p_tn','prm15_wcost_re5_c2p1_co2_cut90p_tn','prm15_wcost_re6_c2p1_co2_cut90p_tn',\
       'prm15_wcost_re7_c2p1_co2_cut90p_tn','prm15_wcost_re8_c2p1_co2_cut90p_tn','prm15_wcost_re9_c2p1_co2_cut90p_tn',\
       'prm15_wcost_re10_c2p1_co2_cut90p_tn','prm15_wcost_re11_c2p1_co2_cut90p_tn','prm15_wcost_re12_c2p1_co2_cut90p_tn',\
       'prm15_wcost_re13_c2p1_co2_cut90p_tn','prm15_wcost_re14_c2p1_co2_cut90p_tn','prm15_wcost_re15_c2p1_co2_cut90p_tn',\
       'prm15_wcost_re16_c2p1_co2_cut90p_tn','prm15_wcost_re17_c2p1_co2_cut90p_tn','prm15_wcost_re18_c2p1_co2_cut90p_tn',\
       'prm15_wcost_re19_c2p1_co2_cut90p_tn','prm15_wcost_re20_c2p1_co2_cut90p_tn']
scs80 = scs+scs1
scs_tick = []
for y in range(len(scs80)):        
    for i in range(len(sc_names)):
        if scs80[y]==sc_names['short_name'][i]:
            scs_tick.append(sc_names['full_name'][i])
scs_short = []
for y in range(len(scs80)):        
    for i in range(len(sc_names)):
        if scs80[y]==sc_names['short_name'][i]:
            scs_short.append(sc_names['full_name2'][i])
            
sc_df = pd.DataFrame(zip(scs80,scs_tick), columns=['sc_name','full_name']).set_index('sc_name')
sc_df['short_name'] = scs_short
sc_match = sc_df[['short_name','full_name']].set_index('full_name')
# scenarios for paper one figures (for coding/debugging)
scenario_list = scs

#%% COLOR
sc_colors=[]
for color in mcolors.TABLEAU_COLORS:
    sc_colors.append(mcolors.TABLEAU_COLORS[color])
sc_colors = sc_colors*8
real_names=[sc_names['full_name'][i] for i in range(len(sc_names)) if sc_names['short_name'][i] in scs80]
sc_styl = ['-.','-.','--','-','-','--','--','-','--','-']*8
style_df = pd.DataFrame({'scenario':scs80,'real_name':real_names,'color':sc_colors,'linestyle':sc_styl}).set_index('scenario')

paired_colors = [[['#7DF5E0','#004D40'],['#FF669E','#D81B60']],[['#FFC107','#AD8200'],['#63B6FF','#1E88E5']]] # [[[green],[pink]],[[yellow],[blue]]]
my_colours = ['#7DF5E0','#008770','#FF96BC','#CA1658','#FFCB31','#AD8200','#84C5FF','#0074DA'] # [green,pink,yellow,blue]
med_colours = ['#008770','#7DF5E0','#CA1658','#FF96BC','#AD8200','#FFCB31','#0074DA','#84C5FF']
palette = {'000':'#FF669E','001':'#D81B60','010':'#10EFC9','011':'#004D40','100':'#63B6FF','101':'#0074DA','110':'#FFC107','111':'#AD8200'}
cap_tech = ['Wind','Solar','Hydro','Fossil']
panels = ['Region-wide'] + mercosur

style_df_4color = style_df.copy(deep=True)
color4 = ['#004D40','#FFC107','#D81B60','#1E88E5']
style_df_4color['2color'] = [color4[0]]*20+[color4[1]]*20+[color4[2]]*20+[color4[3]]*20
style_df_4color['styl'] = ['-']*10+['--']*10+['-']*10+['--']*10+['-']*10+['--']*10+['-']*10+['--']*10
new_med_c = ['#CA1658','#FF96BC','#0074DA','#84C5FF']*6

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap=mcolors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'\
             .format(n=cmap.name,a=minval,b=maxval),cmap(np.linspace(minval,maxval,n)))
    return new_cmap

coal_color = '#343a40'
gas_color = '#6c757d'
hydro_color = '#2a648a'
solar_color = '#ef9226'
wind_color = '#8dc0cd'
nuclear_color = 'darkmagenta'
diesel_color = 'darkred'
other_color = '#f07167'
biomass_color = '#6ba661'
geothermal_color = 'slateblue'
battery_color = '#e7c41f'
pstorage_color = '#6a96ac'
trans_color = 'tan'
curtailment_color = 'red'
imports_color ='#D1FA04'
exports_color = '#FF88F8'
bcap_clr = [wind_color, solar_color, gas_color, '#5e798b', '#cac5b2', '#867197', '#7f927c']
bcap_med_clr = ['k','k','k','k','k','k','k']

#%%



###################################################################
################### DATA PROCESSING FOR FIGURES ###################
###################################################################




#%% CAPACITY ADDITIONS
new_cap = pd.read_csv(datapath+'capacity_mix.csv')
new_cap = new_cap[new_cap.period==2050][['scs','period','load_zone','Battery_new','Biomass_new','Coal_new','Diesel_new',
            'Gas_new','Geothermal_new','Hydro_new','Nuclear_new','Solar_new','Wind_new']].reset_index().drop(columns='index')
new_cap['emis_policy'] = new_cap.scs.str.contains('cut90p')*1
new_cap['exist_trd'] = new_cap.scs.str.contains('t1')*1
new_cap['op_char'] = new_cap.scs.str.split('_', expand=True)[2].str[2:].astype(int)
new_cap['fixed_pv'] = (new_cap.op_char > 10)*1
new_cap['Fossil_new'] = new_cap['Coal_new'] + new_cap['Gas_new'] + new_cap['Diesel_new']
new_cap['pol'] = new_cap.emis_policy.astype(str) + new_cap.exist_trd.astype(str)
new_cap['group'] = new_cap.emis_policy.astype(str) + new_cap.exist_trd.astype(str) + new_cap.fixed_pv.astype(str)
new_cap['pol'] = new_cap.pol.map({'00':'NoCut,Full','01':'NoCut,Lim','10':'90%Cut,Full','11':'90%Cut,Lim'})
difforder=CategoricalDtype(['NoCut,Lim','NoCut,Full','90%Cut,Lim','90%Cut,Full'], ordered=True)
new_cap['pol'] = new_cap['pol'].astype(difforder)
new_cap.sort_values('pol',inplace=True)


cap_mix = pd.read_csv(datapath+'capacity_mix.csv')
cap_mix = cap_mix[cap_mix.period==2050].reset_index().drop(columns='index')
cap_mix['all'] = cap_mix[['Battery_new','Biomass','Biomass_new','Coal','Coal_new','Diesel','Diesel_new','Gas','Gas_new','Geothermal',
                    'Geothermal_new','Hydro','Hydro_new','Nuclear','Nuclear_new','Solar','Solar_new','Wind','Wind_new']].sum(axis=1)
cap_mix['Fossil'] = cap_mix[['Coal','Coal_new','Diesel','Diesel_new','Gas','Gas_new']].sum(axis=1)
cap_mix['WSH'] = cap_mix[['Solar','Solar_new','Wind','Wind_new','Hydro','Hydro_new']].sum(axis=1)
cap_mix['Solar_share'] = (cap_mix['Solar_new']+cap_mix['Solar']) / cap_mix['all']
cap_mix['Wind_share'] = (cap_mix['Wind_new']+cap_mix['Wind']) / cap_mix['all']
cap_mix['Hydro_share'] = (cap_mix['Hydro_new']+cap_mix['Hydro']) / cap_mix['all']
cap_mix['Fossil_share'] = cap_mix['Fossil'] / cap_mix['all']
cap_mix['Solar_WSH'] = (cap_mix['Solar_new']+cap_mix['Solar']) / cap_mix['all']
cap_mix['Wind_WSH'] = (cap_mix['Wind_new']+cap_mix['Wind']) / cap_mix['all']
cap_mix['Hydro_WSH'] = (cap_mix['Hydro_new']+cap_mix['Hydro']) / cap_mix['all']
cap_mix['size'] = np.sqrt((cap_mix['all']/cap_mix['all'].max()))*200

res = []
recap = cap_mix.copy(deep=True)
recap['emis_policy'] = recap.scs.str.contains('cut90p')*1
recap['exist_trd'] = recap.scs.str.contains('t1')*1
recap['op_char'] = recap.scs.str.split('_', expand=True)[2].str[2:].astype(int)
recap['fixed_pv'] = (recap.op_char > 10)*1
recap['Fossil_new'] = recap['Coal_new'] + recap['Gas_new'] + recap['Diesel_new']
recap['pol'] = recap.emis_policy.astype(str) + recap.exist_trd.astype(str)
recap['group'] = recap.emis_policy.astype(str) + recap.exist_trd.astype(str) + recap.fixed_pv.astype(str)
recap['pol'] = recap.pol.map({'00':'NoCut,Full','01':'NoCut,Lim','10':'90%Cut,Full','11':'90%Cut,Lim'})
reg_list = ['Region-wide','Argentina','Brazil','Chile','Paraguay','Uruguay']
pols = recap.pol.unique().tolist()
for i in range(len(reg_list)):
    for j in range(len(pols)):
        temp = recap[(recap.load_zone==reg_list[i])&(recap.pol==pols[j])].copy(deep=True)
        try:
            cv1 = np.std(temp['Solar_WSH'])/np.mean(temp['Solar_WSH'])
            p75_1 = np.percentile(temp['Solar_WSH'],75)
            p25_1 = np.percentile(temp['Solar_WSH'],25)
            p75_n1 = np.percentile(temp['Solar_new'],75)
            p25_n1 = np.percentile(temp['Solar_new'],25)
        except:
            cv1,p75_1,p25_1,p75_n1,p25_n1=0,0,0,0,0
        try:
            cv2 = np.std(temp['Wind_WSH'])/np.mean(temp['Wind_WSH'])
            p75_2 = np.percentile(temp['Wind_WSH'],75)
            p25_2 = np.percentile(temp['Wind_WSH'],25)
            p75_n2 = np.percentile(temp['Wind_new'],75)
            p25_n2 = np.percentile(temp['Wind_new'],25)
        except:
            cv2,p75_2,p25_2,p75_n2,p25_n2=0,0,0,0,0
        try:
            cv3 = np.std(temp['Hydro_WSH'])/np.mean(temp['Hydro_WSH'])
            p75_3 = np.percentile(temp['Hydro_WSH'],75)
            p25_3 = np.percentile(temp['Hydro_WSH'],25)
            p75_n3 = np.percentile(temp['Hydro_new'],75)
            p25_n3 = np.percentile(temp['Hydro_new'],25)
        except:
            cv3,p75_3,p25_3,p75_n3,p25_n3=0,0,0,0,0
        try:
            cv4 = np.std(temp['Fossil_share'])/np.mean(temp['Fossil_share'])
            p75_4 = np.percentile(temp['Fossil_share'],75)
            p25_4 = np.percentile(temp['Fossil_share'],25)
            p75_n4 = np.percentile(temp['Fossil_new'],75)
            p25_n4 = np.percentile(temp['Fossil_new'],25)
        except:
            cv4,p75_4,p25_4,p75_n4,p25_n4=0,0,0,0,0
        res.append((reg_list[i],pols[j],cv1,cv2,cv3,cv4,p75_1,p25_1,p75_2,p25_2,p75_3,p25_3,p75_4,p25_4,p75_n1,p25_n1,p75_n2,p25_n2,p75_n3,p25_n3,p75_n4,p25_n4))
cvdf = pd.DataFrame(res,columns=['load_zone','pol','CV_solar','CV_wind','CV_hydro','CV_fossil',
                                 'S75','S25','W75','W25','H75','H25','F75','F25','Sn75','Sn25','Wn75','Wn25','Hn75','Hn25','Fn75','Fn25'])
solpa=cvdf.pivot(index='load_zone',columns='pol',values='CV_solar').reset_index()[['load_zone','NoCut,Lim','NoCut,Full','90%Cut,Lim','90%Cut,Full']].reindex([4,0,1,2,3,5])
winpa=cvdf.pivot(index='load_zone',columns='pol',values='CV_wind').reset_index()[['load_zone','NoCut,Lim','NoCut,Full','90%Cut,Lim','90%Cut,Full']].reindex([4,0,1,2,3,5])
hydpa=cvdf.pivot(index='load_zone',columns='pol',values='CV_hydro').reset_index()[['load_zone','NoCut,Lim','NoCut,Full','90%Cut,Lim','90%Cut,Full']].reindex([4,0,1,2,3,5])
fospa=cvdf.pivot(index='load_zone',columns='pol',values='CV_fossil').reset_index()[['load_zone','NoCut,Lim','NoCut,Full','90%Cut,Lim','90%Cut,Full']].reindex([4,0,1,2,3,5])

### IQR/QCD prep ###
cv_df = cvdf.copy(deep=True)
cv_df['emis'] = cv_df.pol.map({'NoCut,Lim':'Reference','90%Cut,Lim':'Mitigation','NoCut,Full':'Reference','90%Cut,Full':'Mitigation'})
cv_df['coord'] = cv_df.pol.map({'NoCut,Lim':'Limited','90%Cut,Lim':'Limited','NoCut,Full':'Full','90%Cut,Full':'Full'})
cv_df['IQR_solar'] = cv_df['S75']-cv_df['S25']
cv_df['IQR_wind'] = cv_df['W75']-cv_df['W25']
cv_df['IQR_hydro'] = cv_df['H75']-cv_df['H25']
cv_df['IQR_fossil'] = cv_df['F75']-cv_df['F25']
cv_df['IQR_solar_n'] = cv_df['Sn75']-cv_df['Sn25']
cv_df['IQR_wind_n'] = cv_df['Wn75']-cv_df['Wn25']
cv_df['IQR_hydro_n'] = cv_df['Hn75']-cv_df['Hn25']
cv_df['IQR_fossil_n'] = cv_df['Fn75']-cv_df['Fn25']
cv_df['QCD_solar'] = (cv_df['S75']-cv_df['S25'])/(cv_df['S75']+cv_df['S25'])
cv_df['QCD_wind'] = (cv_df['W75']-cv_df['W25'])/(cv_df['W75']+cv_df['W25'])
cv_df['QCD_hydro'] = (cv_df['H75']-cv_df['H25'])/(cv_df['H75']+cv_df['H25'])
cv_df['QCD_fossil'] = (cv_df['F75']-cv_df['F25'])/(cv_df['F75']+cv_df['F25'])
cv_df['QCD_solar_n'] = (cv_df['Sn75']-cv_df['Sn25'])/(cv_df['Sn75']+cv_df['Sn25'])
cv_df['QCD_wind_n'] = (cv_df['Wn75']-cv_df['Wn25'])/(cv_df['Wn75']+cv_df['Wn25'])
cv_df['QCD_hydro_n'] = (cv_df['Hn75']-cv_df['Hn25'])/(cv_df['Hn75']+cv_df['Hn25'])
cv_df['QCD_fossil_n'] = (cv_df['Fn75']-cv_df['Fn25'])/(cv_df['Fn75']+cv_df['Fn25'])
qcd_df=cv_df.drop(columns=['pol','S75','S25','W75','W25','H75','H25','F75','F25','Sn75','Sn25','Wn75','Wn25','Hn75','Hn25','Fn75','Fn25'])[['load_zone',
                                                                                 'emis','coord','CV_solar','CV_wind','CV_hydro','CV_fossil',
                                                                                 'IQR_solar','IQR_wind','IQR_hydro','IQR_fossil',
                                                                                 'IQR_solar_n','IQR_wind_n','IQR_hydro_n','IQR_fossil_n',
                                                                                 'QCD_solar','QCD_wind','QCD_hydro','QCD_fossil',
                                                                                 'QCD_solar_n','QCD_wind_n','QCD_hydro_n','QCD_fossil_n']]
cv_df=cv_df.drop(columns=['pol','S75','S25','W75','W25','H75','H25','F75','F25','Sn75','Sn25','Wn75','Wn25','Hn75','Hn25','Fn75','Fn25'])[['load_zone',
                                                                                'emis','coord','CV_solar','CV_wind','CV_hydro',
                                                                                'CV_fossil','IQR_solar','IQR_wind','IQR_hydro','IQR_fossil',
                                                                                'IQR_solar_n','IQR_wind_n','IQR_hydro_n','IQR_fossil_n',
                                                                                'QCD_solar','QCD_wind','QCD_hydro','QCD_fossil',
                                                                                'QCD_solar_n','QCD_wind_n','QCD_hydro_n','QCD_fossil_n']]







#%% GENERATION SHARE (ALL 80 SCENARIOS)
f5_techs = ['Wind','Solar','Hydro','Fossil']
share_techs = ['Wind','Solar','Fossil','Hydro','BNG']
generation_mix = pd.read_csv(datapath+'generation_mix.csv')
gen_mix = generation_mix.melt(id_vars=['period','load_zone','scs'],
                        value_vars=['Battery','Biomass', 'Coal','Curtailment','Diesel','Gas','Geothermal','Hydro','Nuclear','Solar','Wind'],
                        var_name='technology',value_name='twh')
gen_mix['scs'] = gen_mix['scs'].map(dict(zip(scs_short,scs80))) ### comment out if using v2 costs
gen_mix['tech'] = gen_mix['technology'].map({'Biomass':'BNG','Coal':'Fossil','Curtailment':'Other','Battery':'Other','Diesel':'Fossil',
                                             'Gas':'Fossil','Geothermal':'BNG','Hydro':'Hydro','Nuclear':'BNG','Solar':'Solar','Wind':'Wind'})
share_gen = gen_mix[gen_mix.tech.isin(share_techs)].copy(deep=True)
try:
    gen_tech_tots = share_gen.groupby(['period','load_zone','scs','tech']).sum().reset_index()
except: 
    gen_tech_tots = share_gen.groupby(['period','load_zone','scs','tech']).sum().reset_index().drop(columns='technology')
gen_tots = gen_tech_tots.groupby(['period','load_zone','scs']).sum().reset_index()
gen_share_df = gen_tech_tots.merge(gen_tots,on=['period','load_zone','scs'],suffixes=['_tech','_tot'])
gen_share_df['share'] = 100*gen_share_df['twh_tech']/gen_share_df['twh_tot']
gen_share_df['period'] = gen_share_df['period'].astype(int)
gen_share_df = gen_share_df.rename(columns={'tech_tech':'tech'})
try:
    gen_pt = gen_share_df.pivot(index=['tech','load_zone','scs'],columns='period',values='share').reset_index()
except:
    gen_pt = gen_share_df.pivot(index=['tech_tech','load_zone','scs'],columns='period',values='share').reset_index()

gen_pt['emis_policy'] = gen_pt.scs.str.contains('cut90p')*1
gen_pt['exist_trd'] = gen_pt.scs.str.contains('t1')*1
gen_pt['op_char'] = gen_pt.scs.str.split('_', expand=True)[2].str[2:].astype(int)
gen_pt['fixed_pv'] = (gen_pt.op_char > 10)*1
gen_pt['pol'] = gen_pt.emis_policy.astype(str) + gen_pt.exist_trd.astype(str)
gen_pt['group'] = gen_pt.emis_policy.astype(str) + gen_pt.exist_trd.astype(str) + gen_pt.fixed_pv.astype(str)
gen_pt.pol = gen_pt.pol.map({'00':'NoCut,Full','01':'NoCut,Lim','10':'90%Cut,Full','11':'90%Cut,Lim'})
difforder=CategoricalDtype(['NoCut,Lim','NoCut,Full','90%Cut,Lim','90%Cut,Full'], ordered=True)
gen_pt['pol'] = gen_pt['pol'].astype(difforder)
gen_pt.sort_values('pol',inplace=True)

gen_share_r = gen_pt[gen_pt.load_zone=='Region-wide'].drop(columns='load_zone')
gen_share_a = gen_pt[gen_pt.load_zone=='Argentina'].drop(columns='load_zone')
gen_share_b = gen_pt[gen_pt.load_zone=='Brazil'].drop(columns='load_zone')
gen_share_c = gen_pt[gen_pt.load_zone=='Chile'].drop(columns='load_zone')
gen_share_p = gen_pt[gen_pt.load_zone=='Paraguay'].drop(columns='load_zone')
gen_share_u = gen_pt[gen_pt.load_zone=='Uruguay'].drop(columns='load_zone')
gen_share_co_list = [gen_share_r, gen_share_a, gen_share_b, gen_share_c, gen_share_p, gen_share_u]
min_list, max_list = [], []
sc_groups = gen_pt.pol.unique().tolist()

min_list_co, max_list_co = [[] for i in range(6)], [[] for i in range(6)]
mins_co, maxes_co = [[] for i in range(6)], [[] for i in range(6)]
for k in range(len(gen_share_co_list)):
    for i in range(len(f5_techs)):
        for j in range(len(sc_groups)):
            temp = gen_share_co_list[k][(gen_share_co_list[k].pol==sc_groups[j]) & (gen_share_co_list[k].tech==f5_techs[i])]
            temp_min = temp.min(axis=0).values[1:-1].tolist()
            min_list_co[k].append(tuple([f5_techs[i]]+temp_min+[sc_groups[j]]))
            temp_max = temp.max(axis=0).values[1:-1].tolist()
            max_list_co[k].append(tuple([f5_techs[i]]+temp_max+[sc_groups[j]]))
    mins_co[k] = pd.DataFrame(min_list_co[k],columns=gen_share_co_list[k].columns).set_index(['tech','pol'])
    maxes_co[k] = pd.DataFrame(max_list_co[k],columns=gen_share_co_list[k].columns).set_index(['tech','pol'])

#%% NEW TRADE DATA (TOTAL TRADE AND FOR ALL LINES)

### NEW TOTAL TRADE DATA ###
tot_trd_df = pd.read_csv(datapath + 'total_trade_data.csv')
tot_trd_df['connection'] = tot_trd_df['lz_from'] + '-' + tot_trd_df['lz_to']
tot_trd_df = tot_trd_df.groupby(['lz_from','lz_to','scenario','connection']).sum().reset_index()
merger = sc_df.drop(columns='full_name').reset_index()
all_trd_tot = tot_trd_df.merge(merger,left_on='scenario',right_on='sc_name').drop(columns='scenario')
all_trd_tot['emis_policy'] = all_trd_tot.sc_name.str.contains('cut90p')*1
all_trd_tot['exist_trd'] = all_trd_tot.sc_name.str.contains('t1')*1
all_trd_tot['op_char'] = all_trd_tot.sc_name.str.split('_', expand=True)[2].str[2:].astype(int)
all_trd_tot['fixed_pv'] = (all_trd_tot.op_char > 10)*1
all_trd_tot['pol'] = all_trd_tot.emis_policy.astype(str) + all_trd_tot.exist_trd.astype(str)
all_trd_tot['group'] = all_trd_tot.emis_policy.astype(str) + all_trd_tot.exist_trd.astype(str) + all_trd_tot.fixed_pv.astype(str)
all_trd_tot['pol'] = all_trd_tot.pol.map({'00':'NoCut,Full','01':'NoCut,Lim','10':'90%Cut,Full','11':'90%Cut,Lim'})
difforder=CategoricalDtype(['NoCut,Lim','NoCut,Full','90%Cut,Lim','90%Cut,Full'], ordered=True)
all_trd_tot['pol'] = all_trd_tot['pol'].astype(difforder)
all_trd_tot.sort_values('pol',inplace=True)

tot_exim = pd.read_csv(datapath + 'all_trade_tot_exim_by_lzyr.csv')
tot_exim = tot_exim.groupby(['type','load_zone','multiplier','Scenario']).sum().reset_index().set_index('Scenario')
tot_exim = tot_exim.merge(sc_df.reset_index(),left_on='Scenario',right_on='full_name').rename(columns={'sc_name':'scenario'})
tot_exim['emis_policy'] = tot_exim.scenario.str.contains('cut90p')*1
tot_exim['exist_trd'] = tot_exim.scenario.str.contains('t1')*1
tot_exim['op_char'] = tot_exim.scenario.str.split('_', expand=True)[2].str[2:].astype(int)
tot_exim['fixed_pv'] = (tot_exim.op_char > 10)*1
tot_exim['pol'] = tot_exim.emis_policy.astype(str) + tot_exim.exist_trd.astype(str)
tot_exim['group'] = tot_exim.emis_policy.astype(str) + tot_exim.exist_trd.astype(str) + tot_exim.fixed_pv.astype(str)
tot_exim['pol'] = tot_exim.pol.map({'00':'NoCut,Full','01':'NoCut,Lim','10':'90%Cut,Full','11':'90%Cut,Lim'})
difforder=CategoricalDtype(['NoCut,Lim','NoCut,Full','90%Cut,Lim','90%Cut,Full'], ordered=True)
tot_exim['pol'] = tot_exim['pol'].astype(difforder)
df_f = tot_exim[tot_exim.fixed_pv==True]
df_s = tot_exim[tot_exim.fixed_pv==False]
tot_ex,tot_im = tot_exim[tot_exim.type=='export'].copy(deep=True), tot_exim[tot_exim.type=='import'].copy(deep=True)
tot_ex.sort_values('pol',inplace=True)
tot_im.sort_values('pol',inplace=True)
df_f_ex, df_f_im = df_f[df_f.type=='export'], df_f[df_f.type=='import']
df_s_ex, df_s_im = df_s[df_s.type=='export'], df_s[df_s.type=='import']

#%% cost_emissions, trade diffs

### trade diffs ###
split_tx_cost_df = pd.read_csv(datapath + 'split_tx_cost_df.csv')
cost_full = split_tx_cost_df[split_tx_cost_df.exist_trd == 0]
cost_ref = split_tx_cost_df[split_tx_cost_df.exist_trd == 1]
cost_merge = cost_ref.merge(cost_full,on=['load_zone','emis_policy','op_char','fixed_pv'])
cost_merge['tx_diff'] = cost_merge['split_tx_cost_x'] - cost_merge['split_tx_cost_y']
cost_merge['cap_diff'] = cost_merge['capacity_cost_x'] - cost_merge['capacity_cost_y']
cost_merge['op_diff'] = cost_merge['operations_cost_total_x'] - cost_merge['operations_cost_total_y']
cost_merge['tot_diff'] = cost_merge['new_total_cost_x'] - cost_merge['new_total_cost_y']
cost_diffs = cost_merge[cost_merge.load_zone=='Region-wide']
cost_diff_cols = ['tx_diff','cap_diff','op_diff','tot_diff']
cost_labels = ['Transmission','Generation Capacity','Grid Operations','Total Costs']
cost_diffs = pd.melt(cost_diffs, id_vars=['scs_id_x','fixed_pv','emis_policy'], value_vars=cost_diff_cols)
### co2 policy diffs ###
cost_nocut = split_tx_cost_df[split_tx_cost_df.emis_policy == 0]
cost_cut = split_tx_cost_df[split_tx_cost_df.emis_policy == 1]
cost_merge2 = cost_cut.merge(cost_nocut,on=['load_zone','exist_trd','op_char','fixed_pv'])
cost_merge2['tx_diff'] = cost_merge2['split_tx_cost_x'] - cost_merge2['split_tx_cost_y']
cost_merge2['cap_diff'] = cost_merge2['capacity_cost_x'] - cost_merge2['capacity_cost_y']
cost_merge2['op_diff'] = cost_merge2['operations_cost_total_x'] - cost_merge2['operations_cost_total_y']
cost_merge2['tot_diff'] = cost_merge2['new_total_cost_x'] - cost_merge2['new_total_cost_y']
cost_diffs2 = cost_merge2[cost_merge2.load_zone=='Region-wide']

#%% HOURLY DISPATCH

df_dispatch = pd.read_parquet(datapath+'hourly_dispatch_all80.parquet')
df_t1 = df_dispatch[df_dispatch.scs.str.contains('t1')][['scs','timepoint','load_zone','Battery_discharge','Battery_charge','curt']]
df_t1.timepoint = df_t1.timepoint.astype(str)
df_t1['Year'] = df_t1['timepoint'].str[:4].astype(int)
df_t1['Month'] = df_t1['timepoint'].str[4:6].astype(int)
df_t1['Hour'] = df_t1['timepoint'].str[6:].astype(int)
df_tn = df_dispatch[df_dispatch.scs.str.contains('tn')][['scs','timepoint','load_zone','Battery_discharge','Battery_charge','curt']]
df_tn.timepoint = df_tn.timepoint.astype(str)
df_tn['Year'] = df_tn['timepoint'].str[:4].astype(int)
df_tn['Month'] = df_tn['timepoint'].str[4:6].astype(int)
df_tn['Hour'] = df_tn['timepoint'].str[6:].astype(int)

df_t1_sw_2050 = df_t1[df_t1.Year==2050].drop(columns='load_zone').groupby(['scs','timepoint','Year','Month','Hour']).sum().reset_index()
df_t1_sw_2050 = df_t1_sw_2050.drop(columns=['scs','timepoint']).groupby(['Year','Month','Hour']).mean().reset_index()
df_tn_sw_2050 = df_tn[df_tn.Year==2050].drop(columns='load_zone').groupby(['scs','timepoint','Year','Month','Hour']).sum().reset_index()
df_tn_sw_2050 = df_tn_sw_2050.drop(columns=['scs','timepoint']).groupby(['Year','Month','Hour']).mean().reset_index()

df_t1_sw_2050_dis_pt = df_t1_sw_2050.pivot(index='Month',columns='Hour',values='Battery_discharge')
df_t1_sw_2050_ch_pt = df_t1_sw_2050.pivot(index='Month',columns='Hour',values='Battery_charge')
df_t1_sw_2050_ch_pt *= -1
df_t1_sw_2050_curt_pt = df_t1_sw_2050.pivot(index='Month',columns='Hour',values='curt')

df_tn_sw_2050_dis_pt = df_tn_sw_2050.pivot(index='Month',columns='Hour',values='Battery_discharge')
df_tn_sw_2050_ch_pt = df_tn_sw_2050.pivot(index='Month',columns='Hour',values='Battery_charge')
df_tn_sw_2050_ch_pt *= -1
df_tn_sw_2050_curt_pt = df_tn_sw_2050.pivot(index='Month',columns='Hour',values='curt')

df_list = [df_t1_sw_2050_dis_pt,df_tn_sw_2050_dis_pt,df_t1_sw_2050_ch_pt,
           df_tn_sw_2050_ch_pt,df_t1_sw_2050_curt_pt,df_tn_sw_2050_curt_pt]

df_dispatch_sw = df_dispatch.groupby(['timepoint','scs']).sum().reset_index().set_index('timepoint')
### averages for scenario groupings across all turbines: cut90p.t1.1axis, cut90p.t1.fixed, cut90p.tn.1axis, cut90p.tn.fixedyear,months,hours = '2050',['01','02','03','04','05','06','07','08','09','10','11','12'],['01','24']
df_dispatch_sw = df_dispatch.groupby(['timepoint','scs']).sum().reset_index()
df_dispatch_sw['op_char'] = df_dispatch_sw.scs.str.split('_', expand=True)[2].str[2:].astype(int)
df_dispatch_sw['month'] = df_dispatch_sw.timepoint.astype(str).str[4:6].astype(int)
group1 = df_dispatch_sw[(df_dispatch_sw.scs.str.contains('cut90p_t1')) & (df_dispatch_sw.op_char<=10) & (df_dispatch_sw.month.isin([4,8,12]))]
group2 = df_dispatch_sw[(df_dispatch_sw.scs.str.contains('cut90p_t1')) & (df_dispatch_sw.op_char>10) & (df_dispatch_sw.month.isin([4,8,12]))]
group3 = df_dispatch_sw[(df_dispatch_sw.scs.str.contains('cut90p_tn')) & (df_dispatch_sw.op_char<=10) & (df_dispatch_sw.month.isin([4,8,12]))]
group4 = df_dispatch_sw[(df_dispatch_sw.scs.str.contains('cut90p_tn')) & (df_dispatch_sw.op_char>10) & (df_dispatch_sw.month.isin([4,8,12]))]
try:
    group1 = group1.drop(columns=['scs','op_char','load_zone'])
    group2 = group2.drop(columns=['scs','op_char','load_zone'])
    group3 = group3.drop(columns=['scs','op_char','load_zone'])
    group4 = group4.drop(columns=['scs','op_char','load_zone'])
except:
    pass
group1 = group1.groupby(['timepoint','month']).mean().reset_index()
group1['group'] = 1
group2 = group2.groupby(['timepoint','month']).mean().reset_index()
group2['group'] = 2
group3 = group3.groupby(['timepoint','month']).mean().reset_index()
group3['group'] = 3
group4 = group4.groupby(['timepoint','month']).mean().reset_index()
group4['group'] = 4
mean_disp = pd.concat([group1,group2,group3,group4])
mean_disp = mean_disp.set_index('timepoint')

#%% CURTAIL, STORAGE

# start with battery data
bat_cap = new_cap[['scs', 'period', 'load_zone', 'Battery_new']].copy(deep=True).rename(columns={'scs':'sc_name'})

# add curtailment data
vre_curt = pd.read_csv(datapath + 'vre_curtailment.csv')
curt = vre_curt.groupby(['scs','period','load_zone']).sum().reset_index()
curt_sys = curt.groupby(['scs','period']).sum().reset_index()
curt_sys['load_zone'] = 'Region-wide'
tot_curt = curt.groupby(['scs','load_zone']).sum().reset_index()
tot_curt_sys = curt_sys.groupby(['scs','load_zone']).sum().reset_index()
curt_df = pd.concat([tot_curt,tot_curt_sys]).rename(columns={'scs':'sc_name','curt':'twh'})
decarb = bat_cap.merge(curt_df[['load_zone','sc_name','twh']],on=['sc_name','load_zone'])

# add emissions
emis_df = split_tx_cost_df[['scs_id','load_zone','MtCO2']].copy(deep=True).rename(columns={'scs_id':'sc_name'})
decarb = decarb.merge(emis_df,on=['sc_name','load_zone'])
decarb['emis_policy'] = decarb.sc_name.str.contains('cut90p')*1
decarb['exist_trd'] = decarb.sc_name.str.contains('t1')*1
decarb['op_char'] = decarb.sc_name.str.split('_', expand=True)[2].str[2:].astype(int)
decarb['fixed_pv'] = (decarb.op_char > 10)*1
decarb['pol'] = decarb.emis_policy.astype(str) + decarb.exist_trd.astype(str)
decarb['group'] = decarb.emis_policy.astype(str) + decarb.exist_trd.astype(str) + decarb.fixed_pv.astype(str)
decarb['pol'] = decarb.pol.map({'00':'NoCut,Full','01':'NoCut,Lim','10':'90%Cut,Full','11':'90%Cut,Lim'})
difforder=CategoricalDtype(['NoCut,Lim','NoCut,Full','90%Cut,Lim','90%Cut,Full'], ordered=True)
decarb['pol'] = decarb['pol'].astype(difforder)
decarb.sort_values('pol',inplace=True)

decarb_cols = ['Battery_new', 'twh', 'MtCO2']
decarb_labels = ['Battery Storage Requirement (GW)','Cumulative VRE Curtailment (TWh)','Cumulative CO2 Emissions (MtCO2)']
decarb_titles = ['Battery Storage','Curtailment','Emissions']

#%% VRE CAPACITY FACTORS, VRE PREFERENCE

hue_orders = ['Pampas','Atacama/Andes','Gran Chaco','Caatinga','Cerrado','Monte Desert']
hue_orderw = ['Pampas','Patagonia','Gran Chaco','Caatinga','Coastal']
color_s = ['#88CCEE','#FF9000','#72F34F','#F03842','#917AFF','#924400']
color_w = ['#88CCEE','#E0DA00','#72F34F','#F03842','#0086A9']
turb_cols=['cf_vestas_2','cf_vestas_7','cf_vestas_3','cf_siemens_36','cf_siemens_23','cf_GE_25','cf_GE_15','cf_gamesa_2','cf_enercon_7','cf_enercon_3']
pa_cols=['Wind Coastal','Wind Gran Chaco','Wind Pampas','Wind Patagonia','Wind Caatinga','SolarPV Cerrado',
          'SolarPV Caatinga','SolarPV Atacama/Andes','SolarPV Gran Chaco','SolarPV Monte Desert','SolarPV Pampas']

solar_cf = pd.read_csv(datapath + 'solar_cf.csv')
wind_cf = pd.read_csv(datapath + 'wind_cf.csv')
solar_cf['all_avg'] = (solar_cf['cf_1axis']+solar_cf['cf_fixed'])/2
wind_cf['all_avg'] = wind_cf.iloc[:,-10:].mean(axis=1)

verts = [(0., 0.),  # left, bottom
          (0., 1.),  # left, top
          (0.9, 1.),  # right, top
          (0.9, 0.),  # right, bottom
          (0., 0.),]  # back to left, bottom
codes = [Path.MOVETO, #begin drawing
          Path.LINETO, #straight line
          Path.LINETO,
          Path.LINETO,
          Path.CLOSEPOLY,] #close shape
path = Path(verts, codes)

pa_vre = pd.read_csv(datapath + 'parallel_axis_vre_df_country.csv',index_col='scenario')
sys_vre = pa_vre.reset_index().groupby(['scenario','technology','emis_policy','exist_trd','op_char','fixed_pv','pol','group']).sum().reset_index().set_index('scenario')
sys_vre['load_zone'] = 'system-wide'

pa_vre = pd.concat([pa_vre,sys_vre])
pa_wind = pa_vre[pa_vre.technology=='Wind']
pa_solar = pa_vre[pa_vre.technology=='SolarPV']
padf = pa_wind.merge(pa_solar,on=['load_zone','emis_policy','exist_trd','op_char','fixed_pv','pol','group'],suffixes=['_wind','_solar'])

padf['pref'] = padf['new_build_mw_wind']/(padf['new_build_mw_wind']+padf['new_build_mw_solar'])
difforder=CategoricalDtype([10,11,0,1,110,111,100,101], ordered=True)
padf['group'] = padf['group'].astype(difforder)
padf.sort_values('group',inplace=True)

padf = pa_wind.reset_index().merge(pa_solar.reset_index(),on=['scenario','load_zone','emis_policy','exist_trd','op_char','fixed_pv','pol','group'],suffixes=['_wind','_solar'])
padf['pref'] = padf['new_build_mw_wind']/(padf['new_build_mw_wind']+padf['new_build_mw_solar'])
pref_vre_pt = pd.pivot_table(padf[padf.load_zone!='system-wide'],values='pref',index='scenario',columns='load_zone').reset_index()
#pref_vre_pt = pref_vre_pt.fillna(0.5)
pref_vre_pt['emis_policy'] = pref_vre_pt.scenario.str.contains('cut90p')*1
pref_vre_pt['exist_trd'] = pref_vre_pt.scenario.str.contains('t1')*1
pref_vre_pt['op_char'] = pref_vre_pt.scenario.str.split('_', expand=True)[2].str[2:].astype(int)
pref_vre_pt['fixed_pv'] = (pref_vre_pt.op_char > 10)*1
pref_vre_pt['pol'] = pref_vre_pt.emis_policy.astype(str) + pref_vre_pt.exist_trd.astype(str)
pref_vre_pt['group'] = pref_vre_pt.emis_policy.astype(str) + pref_vre_pt.exist_trd.astype(str) + pref_vre_pt.fixed_pv.astype(str)
pref_vre_pt['pol'] = pref_vre_pt.pol.map({'00':'NoCut,Full','01':'NoCut,Lim','10':'90%Cut,Full','11':'90%Cut,Lim'})
df_s = pref_vre_pt[pref_vre_pt.fixed_pv==0].copy(deep=True)[['Argentina','Brazil','Uruguay','Paraguay','Chile','pol']]
df_f = pref_vre_pt[pref_vre_pt.fixed_pv==1].copy(deep=True)[['Argentina','Brazil','Uruguay','Paraguay','Chile','pol']]
pref_vre_pt = pref_vre_pt[['Argentina','Brazil','Uruguay','Paraguay','Chile','pol']]

new_lin = pd.read_parquet(datapath + 'new_lin.parquet')
new_lin_tot = new_lin[['project','load_zone','technology','scenario','new_build_mw']].copy(deep=True).reset_index().drop(columns='index')
caps = new_lin[['project','cap_mw']].groupby('project').mean().reset_index()
new_lin_totals = new_lin_tot.groupby(['project','load_zone','technology','scenario']).sum().reset_index()
new_lin_totals['cap_mw'] = new_lin_totals['project'].map(caps.set_index('project')['cap_mw'])
new_lin_totals['new_cap_mw'] = new_lin_totals['cap_mw']*(new_lin_totals['new_build_mw'] >= new_lin_totals['cap_mw']) + \
                                      new_lin_totals['new_build_mw']*(new_lin_totals['new_build_mw'] < new_lin_totals['cap_mw'])
new_lin_totals['new_build_mw'] = new_lin_totals['new_cap_mw']
new_lin_systot = new_lin_totals.groupby(['scenario','technology','load_zone']).sum().reset_index()
scaled_df = new_lin_systot.copy(deep=True).reset_index().drop(columns='index')
scaled_df['emis_policy'] = scaled_df.scenario.str.contains('cut90p')*1
scaled_df['exist_trd'] = scaled_df.scenario.str.contains('t1')*1
scaled_df['op_char'] = scaled_df.scenario.str.split('_', expand=True)[2].str[2:].astype(int)
scaled_df['fixed_pv'] = (scaled_df.op_char > 10)*1
scaled_df['pol'] = scaled_df.emis_policy.astype(str) + scaled_df.exist_trd.astype(str)
scaled_df['group'] = scaled_df.emis_policy.astype(str) + scaled_df.exist_trd.astype(str) + scaled_df.fixed_pv.astype(str)
scaled_df['pol'] = scaled_df.pol.map({'00':'NoCut,Full','01':'NoCut,Lim','10':'90%Cut,Full','11':'90%Cut,Lim'})
difforder=CategoricalDtype(['NoCut,Lim','NoCut,Full','90%Cut,Lim','90%Cut,Full'], ordered=True)
scaled_df['pol'] = scaled_df['pol'].astype(difforder)
scaled_df.sort_values('pol',inplace=True)
scaled_df.set_index('scenario', inplace=True)
scaled_df['region'] = scaled_df['technology'] + ' ' + scaled_df['load_zone']
new_vre_pt = pd.pivot_table(scaled_df,values='new_build_mw',index='scenario',columns='region')
scaled_df = new_vre_pt.copy(deep=True).reset_index()
scaled_df['emis_policy'] = scaled_df.scenario.str.contains('cut90p')*1
scaled_df['exist_trd'] = scaled_df.scenario.str.contains('t1')*1
scaled_df['op_char'] = scaled_df.scenario.str.split('_', expand=True)[2].str[2:].astype(int)
scaled_df['fixed_pv'] = (scaled_df.op_char > 10)*1
scaled_df['pol'] = scaled_df.emis_policy.astype(str) + scaled_df.exist_trd.astype(str)
scaled_df['group'] = scaled_df.emis_policy.astype(str) + scaled_df.exist_trd.astype(str) + scaled_df.fixed_pv.astype(str)
scaled_df['pol'] = scaled_df.pol.map({'00':'NoCut,Full','01':'NoCut,Lim','10':'90%Cut,Full','11':'90%Cut,Lim'})
difforder=CategoricalDtype(['NoCut,Lim','NoCut,Full','90%Cut,Lim','90%Cut,Full'], ordered=True)
scaled_df['pol'] = scaled_df['pol'].astype(difforder)
scaled_df.sort_values('pol',inplace=True)
scaled_df.set_index('scenario', inplace=True)
wsdf= scaled_df.copy(deep=True)
wsdf['Argentina'] = wsdf['Wind Argentina'] + wsdf['SolarPV Argentina']
wsdf['Brazil'] = wsdf['Wind Brazil'] + wsdf['SolarPV Brazil']
wsdf['Uruguay'] = wsdf['Wind Uruguay'] + wsdf['SolarPV Uruguay']
wsdf['Paraguay'] = wsdf['Wind Paraguay'] + wsdf['SolarPV Paraguay']
wsdf['Chile'] = wsdf['Wind Chile'] + wsdf['SolarPV Chile']
mm = wsdf.loc[:,'Argentina':'Chile'].max().max()
mm=375000
wsdf['Argentina'] = wsdf['Argentina']/mm
wsdf['Brazil'] = wsdf['Brazil']/mm
wsdf['Uruguay'] = wsdf['Uruguay']/mm
wsdf['Paraguay'] = wsdf['Paraguay']/mm
wsdf['Chile'] = wsdf['Chile']/mm



#%% WIND AND SOLAR RESOURCE POTENTIAL (FOR SI TABLES)

# use solar_cf and wind_cf
solar_country_cap = solar_cf.groupby('load_zone').sum()[['cap_mw']]
solar_country_cf = solar_cf.drop(columns='project').groupby('load_zone').mean()[['all_avg']]
wind_country_cap = wind_cf.groupby('load_zone').sum()[['cap_mw']]
wind_country_cf = wind_cf.drop(columns='project').groupby('load_zone').mean()[['all_avg']]

#%%



###################################################################
############################# FIGURES #############################
###################################################################




#%% BOXPLOT OF COSTS (Figure 2)

plt.rcParams.update(plt.rcParamsDefault)
font = FontProperties()
font.set_name('Open Sans')
plt.rcParams["font.family"] = "Arial"
mpl.rcParams['pdf.fonttype'] = 42
fig = plt.figure(constrained_layout=False, figsize=(8,6.8))
gs = fig.add_gridspec(1,5, width_ratios=[1,.02,1.1,.3,0.85], wspace=1.0)
ylims,yt,ytl = [(-25,10)],[(0,50,100,150,200,250,300)],[('0','50','100','150','200','250','300')]
diff_colors3, diff_med_colors3 = ['#81b961','#EC417F'], ['#244213','#FF84B1']
diff_colors2, diff_med_colors2 = ['#429DED','#E4B321'], ['#0074DA','#AD8200']
diff_colors, diff_med_colors = ['#030E4F','#030E4F','#030E4F'], ['#408298','#408298','#408298']

ax = fig.add_subplot(gs[0,4])
ax4 = fig.add_subplot(gs[0,4])
ax4.axis('off')
ax5 = fig.add_subplot(gs[0,4])
ax5.axis('off')
cd_opp = cost_diffs.copy(deep=True)
cd_opp['value'] = cd_opp['value'] * -1
g = sns.boxplot(data=cd_opp[(cd_opp.emis_policy==1)&(cd_opp.variable=='tx_diff')],x='variable',y='value',width=0.5,ax=ax,
                medianprops={"linewidth": 1.5,'color':'k','label':'_median_','solid_capstyle':'butt'},saturation=1,
                flierprops={"marker":"o",'markerfacecolor':'none','markeredgecolor':'k','markersize':2},boxprops={"linewidth":.6})
g2 = sns.boxplot(data=cd_opp[(cd_opp.emis_policy==1)&(cd_opp.variable=='op_diff')],x='variable',y='value',width=0.5,ax=ax4,
                medianprops={"linewidth": 1.5,'color':'k','label':'_median_','solid_capstyle':'butt'},saturation=1,
                flierprops={"marker":"o",'markerfacecolor':'none','markeredgecolor':'k','markersize':2},boxprops={"linewidth":.6})
g3 = sns.boxplot(data=cd_opp[(cd_opp.emis_policy==1)&(cd_opp.variable=='cap_diff')],x='variable',y='value',width=0.5,ax=ax5,
                medianprops={"linewidth": 1.5,'color':'k','label':'_median_','solid_capstyle':'butt'},saturation=1,
                flierprops={"marker":"o",'markerfacecolor':'none','markeredgecolor':'k','markersize':2},boxprops={"linewidth":.6})
ax.legend([],frameon=False)
for a in [ax,ax4,ax5]:
    boxes = a.findobj(matplotlib.patches.PathPatch)
    for color, box in zip(diff_colors, boxes):
        box.set_facecolor(color)
    box_patches = [patch for patch in a.patches if type(patch) == matplotlib.patches.PathPatch]
    if len(box_patches) == 0:
        box_patches = a.artists
    num_patches = len(box_patches)
    lines_per_boxplot = len(a.lines) // num_patches
    for j, patch in enumerate(box_patches):
        col = patch.get_facecolor()
        patch.set_edgecolor(col)
        for line in a.lines[j * lines_per_boxplot: (j + 1) * lines_per_boxplot]:
            line.set_color(col)
            line.set_mfc(col)  # facecolor of fliers
            line.set_mec(col)  # edgecolor of fliers
    median_lines = [line for line in a.get_lines() if line.get_label() == '_median_']
    for j, line in enumerate(median_lines):
        line.set_color(diff_med_colors[j])
    a.set_ylim(-25,13)
    a.set_xlim(-1.2,0.47)
ax.set_ylabel('Benefits of Coordination under Mitigation',fontsize=16, labelpad=34)
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_xlabel('')
ax.set_yticks([-25., -20., -15., -10.,  -5.,   0.,   5.,  10.])
ax.set_yticklabels(['–$25B','–$20B','–$15B','–$10B','–$5B','$0','$5B','$10B'], fontsize=11)
ax.annotate('Transmission\nexpansion', xy=(1.06,0.773), xycoords='axes fraction', annotation_clip=False,fontsize=11)
ax.annotate('Grid operations\n(includes fuel costs)', xy=(1.06,0.57), xycoords='axes fraction', annotation_clip=False,fontsize=11)
ax.annotate('New capacity', xy=(1.06,0.16), xycoords='axes fraction', annotation_clip=False,fontsize=11)
ax.axhline(0,color='gray',linewidth=0.8, linestyle='--')
ax.arrow(-0.64,0.75,0,6,color='#C70000',width=0.036,head_width=0.14,head_length=.6)
ax.arrow(-0.64,-0.75,0,-6,color='#C70000',width=0.036,head_width=0.14,head_length=.6)
ax.annotate('Cost savings', xy=(0.11,0.56), xycoords='axes fraction', annotation_clip=False,fontsize=8,rotation=90,ha='left',va='center')
ax.annotate('Expenditures', xy=(0.11,0.764), xycoords='axes fraction', annotation_clip=False,fontsize=8,rotation=90,ha='left',va='center')
plt.annotate('(difference between Full and Limited Coordination)',xy=(-0.98,0.5),xycoords='axes fraction',annotation_clip=False,fontsize=10,rotation=90,ha='left',va='center')

ax2 = fig.add_subplot(gs[0,0])
cost_diffs2['cost_prem'] = (cost_diffs2.new_total_cost_x - cost_diffs2.new_total_cost_y)/cost_diffs2.new_total_cost_y
g = sns.boxplot(data=cost_diffs2,x='exist_trd',y='cost_prem', width=0.55, ax=ax2,
                medianprops={"linewidth": 1.5,'color':'k','label':'_median_','solid_capstyle':'butt'},saturation=1,
                flierprops={"marker": "o",'markerfacecolor':'none','markeredgecolor':'k','markersize':2},
                boxprops={"linewidth": .6,'edgecolor':'k','facecolor':'white'})
ax2.legend([],frameon=False)

boxes = ax2.findobj(matplotlib.patches.PathPatch)
for color, box in zip(diff_colors2, boxes):
    box.set_facecolor(color)
box_patches = [patch for patch in ax2.patches if type(patch) == matplotlib.patches.PathPatch]
if len(box_patches) == 0:
    box_patches = ax2.artists
num_patches = len(box_patches)
lines_per_boxplot = len(ax2.lines) // num_patches
for j, patch in enumerate(box_patches):
    col = patch.get_facecolor()
    patch.set_edgecolor(col)
    for line in ax2.lines[j * lines_per_boxplot: (j + 1) * lines_per_boxplot]:
        line.set_color(col)
        line.set_mfc(col)  # facecolor of fliers
        line.set_mec(col)  # edgecolor of fliers
median_lines = [line for line in ax2.get_lines() if line.get_label() == '_median_']
for j, line in enumerate(median_lines):
    line.set_color(diff_med_colors2[j])
ax2.set_xlabel('')
ax2.set_ylim(0,.20)
vals = ax2.get_yticks()
ax2.set_yticklabels(['{:.1%}'.format(x) for x in vals], fontsize=11)
ax2.set_xlim(-0.5,1.5)
ax2.set_ylabel('Cost premium of Mitigation',fontsize=16, labelpad=13)
ax2.set_xticklabels(['Full'+'\n'+'Coord.','Limited'+'\n'+'Coord.'],fontsize=10)

ax3 = fig.add_subplot(gs[0,2])
cost_diffs3 = cost_diffs[cost_diffs.variable=='tot_diff'].copy(deep=True)
cost_diffs3['value'] *= -1
g3 = sns.boxplot(cost_diffs3, x='emis_policy',y='value', order=[1,0], ax=ax3, width=0.58,
                  medianprops={"linewidth": 1.5,'color':'k','label':'_median_','solid_capstyle':'butt'},saturation=1,
                  flierprops={"marker": "o",'markerfacecolor':'none','markeredgecolor':'k','markersize':2},
                  boxprops={"linewidth": .6,'edgecolor':'k','facecolor':'white'})
ax3.legend([],frameon=False)
boxes = ax3.findobj(matplotlib.patches.PathPatch)
for color, box in zip(diff_colors3, boxes):
    box.set_facecolor(color)
box_patches = [patch for patch in ax3.patches if type(patch) == matplotlib.patches.PathPatch]
if len(box_patches) == 0:
    box_patches = ax3.artists
num_patches = len(box_patches)
lines_per_boxplot = len(ax3.lines) // num_patches
for j, patch in enumerate(box_patches):
    col = patch.get_facecolor()
    patch.set_edgecolor(col)
    for line in ax3.lines[j * lines_per_boxplot: (j + 1) * lines_per_boxplot]:
        line.set_color(col)
        line.set_mfc(col)  # facecolor of fliers
        line.set_mec(col)  # edgecolor of fliers
median_lines = [line for line in ax3.get_lines() if line.get_label() == '_median_']
for j, line in enumerate(median_lines):
    line.set_color(diff_med_colors3[j])
ax3.set_xlabel('')
ax3.set_ylim(-25,13)
ax3.set_yticks([-25,-20,-15,-10,-5,0,5,10])
ax3.set_yticklabels(['–$25B','–$20B','–$15B','–$10B','–$5B','$0','$5B','$10B'], fontsize=11)
ax3.axhline(0,color='gray',linewidth=0.8, linestyle='--')
ax3.set_xlim(-0.6,1.6)
ax3.set_ylabel('Net Cost Savings of Coordination',fontsize=16, labelpad=13)
ax3.set_xticklabels(['             Mitigation' + '  ' + 'Reference',''],fontsize=10)
ax3.tick_params(axis='x', which='major', pad=8)

ax2.annotate('a',xy=(0.21,0.964), xycoords='axes fraction',ha='center',va='center',fontweight='bold',fontsize=16)
ax.annotate('c',xy=(0.23,0.964), xycoords='axes fraction',ha='center',va='center',fontweight='bold',fontsize=16)
ax3.annotate('b',xy=(0.21,0.964), xycoords='axes fraction',ha='center',va='center',fontweight='bold',fontsize=16)
#fig.savefig(figpath + 'figure2.png', bbox_inches='tight', dpi=600)
fig.savefig(figpath + 'figure2.pdf', bbox_inches='tight', dpi=600)
plt.show()

#%% COORDINATION IMPACT BOXPLOTS (Figure 3)

### GEN ###
gms = gen_mix[(gen_mix.load_zone=='Region-wide')&(gen_mix.period>=2050)].copy(deep=True)
gms['tech'] = gms['technology'].map({'Biomass':'Biomass','Coal':'Coal','Curtailment':'Other','Battery':'Battery','Diesel':'Diesel',
                            'Gas':'Gas','Geothermal':'Other','Hydro':'Hydro','Nuclear':'Nuclear','Solar':'Solar','Wind':'Wind'})
gms = gms[gms.tech != 'Other']
gms['twh'] = np.abs(gms['twh'])
gms['emis'] = gms['scs'].str.contains('cut90p')
gms['trd'] = gms['scs'].str.contains('_tn')
gms['rscen'] = gms['scs'].str[:-3]
gmsm=gms[gms.trd==0].merge(gms[gms.trd==1],on=['rscen','emis','tech'],
                    suffixes=['_t1','_tn']).drop(columns=['scs_t1','scs_tn','trd_t1','trd_tn'])
gmsm['diff'] = gmsm['twh_t1'] - gmsm['twh_tn']
gmsm = gmsm[['rscen','tech','emis','diff']].copy(deep=True)
gmsm['emis'] = gmsm['emis'].map({True:'cut90p',False:'nocut'})
gmsm['tech'] = gmsm['tech'].map({'Biomass':'Biomass','Coal':'Fossil','Battery':'Battery','Diesel':'Fossil',
                            'Gas':'Fossil','Hydro':'Hydro','Nuclear':'Nuclear','Solar':'Solar','Wind':'Wind'})

difforder=CategoricalDtype(['Wind','Solar','Fossil','Hydro','Battery','Nuclear','Biomass'], ordered=True)
gmsm['tech'] = gmsm['tech'].astype(difforder)
gmsm.sort_values('tech',inplace=True)

### COST ###
costt_df = pd.read_csv(datapath + 'costs_by_tech.csv').rename(columns={'scs_id':'scs'})
costt_df['capacity_cost'] /= 1000000000
costt_df['emis'] = costt_df['scs'].str.contains('cut90p')
costt_df['trd'] = costt_df['scs'].str.contains('_tn')
costt_df['rscen'] = costt_df['scs'].str[:-3]
ctsm=costt_df[costt_df.trd==0].merge(costt_df[costt_df.trd==1],on=['rscen','emis','tech'],
                    suffixes=['_t1','_tn']).drop(columns=['scs_t1','scs_tn','trd_t1','trd_tn'])
ctsm['diff'] = ctsm['capacity_cost_t1'] - ctsm['capacity_cost_tn']
ctsm = ctsm[['rscen','tech','emis','diff']].copy(deep=True)
ctsm['emis'] = ctsm['emis'].map({True:'cut90p',False:'nocut'})
ctsm = ctsm[ctsm.tech != 'Geothermal']
difforder=CategoricalDtype(['Wind','Solar','Fossil','Hydro','Battery','Nuclear','Biomass'], ordered=True)
ctsm['tech'] = ctsm['tech'].astype(difforder)
ctsm.sort_values('tech',inplace=True)

bcap_clr = ['#4069ff', '#e87b00', '#9d0000', '#cadae5', '#cac5b2', '#b8adc2', '#c2cdc0']
bcap_med_clr = ['#79d4ff','#bb4a00','#d63434','#166296','#84722b','#5e4f6c','#4a7542']
ctsm['diff'] *= -1
gmsm['diff'] *= -1
ctsm = ctsm[ctsm.emis=='cut90p'].copy(deep=True).drop(columns='emis')
gmsm = gmsm[gmsm.emis=='cut90p'].copy(deep=True).drop(columns='emis')
fossil_agg = gmsm[gmsm.tech=='Fossil'].copy(deep=True)
fossil_agg = fossil_agg.groupby(['rscen','tech'], observed=True).sum().reset_index()
gmsm = pd.concat([gmsm[gmsm.tech != 'Fossil'], fossil_agg])
gmsm.to_csv('sd/fig3b.csv',index=False)

plt.rcParams.update(plt.rcParamsDefault)
font = FontProperties()
font.set_name('Open Sans')
plt.rcParams["font.family"] = "Arial"
mpl.rcParams['pdf.fonttype'] = 42
fig = plt.figure(constrained_layout=False, figsize = (5.5,3.2))
gs = fig.add_gridspec(1,2, height_ratios=[1], width_ratios=[1,1], wspace=1.04)
ax1,ax2 = fig.add_subplot(gs[0]),fig.add_subplot(gs[1])
axs, metdfs, yl, lp, pl = [ax1,ax2], [ctsm,gmsm], [8,160], 9, ['a','b']
ytk = [['-$8B','-$6B','-$4B','-$2B','$0','$2B','$4B','$6B','$8B'],['','-150 TWh','-100 TWh','-50 TWh','0 TWh','50 TWh','100 TWh','150 TWh']]

for i in range(len(axs)):
    
    g = sns.boxplot(data=metdfs[i],y='diff',x='tech',ax=axs[i],saturation=1,width=0.8,
                    medianprops={"linewidth": .9,'color':'k','label':'_median_','solid_capstyle':'butt'},
                    boxprops={"linewidth": .3,'edgecolor':'k'},whiskerprops={"linewidth": 2,'color':'k','solid_capstyle':'butt'},
                    flierprops={"marker": "d",'markerfacecolor':'none','markeredgecolor':'k','markersize':3},
                    capprops={"linewidth": 2,'color':'k'})
    axs[i].axhline(y=0, color='k',alpha=0.6,linewidth=0.4,linestyle='--',zorder=0)
    axs[i].set_ylim(-1*yl[i],yl[i])
    axs[i].set_xlim([-0.63,6.63])
    axs[i].set_xlabel('')
    axs[i].set_xticks([j for j in range(7)])
    axs[i].set_xticklabels(['Wind','Solar','Fossil','Hydro','Battery','Nuclear','Biomass'])
    plt.setp(axs[i].get_xticklabels(), rotation=90, fontsize=9)
    axs[i].annotate(pl[i],xy=(0.5,0.94),xycoords='axes fraction',ha='center',va='center',fontsize=12,fontweight='bold')
    axs[i].set_yticklabels(ytk[i])
    
    boxes = axs[i].findobj(matplotlib.patches.PathPatch)
    for color, box in zip(bcap_clr, boxes):
        box.set_facecolor(color)
    box_patches = [patch for patch in axs[i].patches if type(patch) == matplotlib.patches.PathPatch]
    if len(box_patches) == 0:
        box_patches = axs[i].artists
    num_patches = len(box_patches)
    if num_patches > 0:
        lines_per_boxplot = len(axs[i].lines) // num_patches
        for k, patch in enumerate(box_patches):
            col = patch.get_facecolor()
            patch.set_edgecolor(col)
            for line in axs[i].lines[k * lines_per_boxplot: (k + 1) * lines_per_boxplot]:
                line.set_color(col)
                line.set_mfc(col)  # facecolor of fliers
                line.set_mec(col)  # edgecolor of fliers
        median_lines = [line for line in axs[i].get_lines() if line.get_label() == '_median_']
        for k, line in enumerate(median_lines):
            line.set_color(bcap_med_clr[k])

ax1.set_ylabel('Impact of Full Coordination' + '\non Capacity Investments',labelpad=lp,fontsize=12,linespacing=1.5)
ax2.set_ylabel('Impact of Full Coordination' + '\non Total Generation',labelpad=lp,fontsize=12,linespacing=1.5)

plt.savefig(figpath + 'figure3.pdf', bbox_inches='tight', dpi=800)
plt.show()

#%% TRADE BOXPLOTS PLUS BARPLOTS (Figure 4)

nopv_pal = {1:'#429DED',0:'#EC417F'}
nopv_med_c = ['#ff84b1']*6 + ['#0074DA']*6
plt.rcParams.update(plt.rcParamsDefault)
font = FontProperties()
font.set_name('Open Sans')
plt.rcParams["font.family"] = "Arial"
mpl.rcParams['pdf.fonttype'] = 42
fig = plt.figure(constrained_layout=False, figsize = (10.2,4.9))
gs = fig.add_gridspec(2,2, height_ratios=[1,1], hspace=0.12, width_ratios=[1.45,1], wspace=0.21)
ax1,ax2,ax3 = fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[:,0])
g = sns.barplot(x='load_zone',y='trade',data=tot_ex[tot_ex.group.isin(['001','101','000','100'])],ax=ax1,saturation=1,hue='emis_policy',
            palette=nopv_pal,estimator=np.mean,errorbar=('pi',100),capsize=0,order=mercosur,err_kws={'linewidth':2},hue_order=[0,1])
ax1.set_xticklabels(mercosur, fontsize=8)
ax1.tick_params(axis='x', which='major', pad=1.1)
ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, bottom=False, top=False, left=True, right=False)
ax1.set_ylim(0,310)
ax1.set_yticks([0,100,200,300])
ax1.set_yticklabels(['0','100','200','300'], fontsize=9)
ax1.set_ylabel('Export from (TWh)',labelpad=8, fontsize=13)
ax1.legend([],frameon=False)
g = sns.barplot(x='load_zone',y='trade',data=tot_im[tot_im.group.isin(['001','101','000','100'])],ax=ax2,saturation=1,hue='emis_policy',
            palette=nopv_pal,estimator=np.mean,errorbar=('pi',100),capsize=0,order=mercosur,err_kws={'linewidth':2},hue_order=[0,1])
ax2.set_xticklabels([])
ax2.set_ylim(0,310)
ax2.set_yticks([0,100,200,300])
ax2.set_yticklabels(['0','100','200','300'], fontsize=9)
ax2.set_ylabel('Import to (TWh)',labelpad=8, fontsize=13)
ax1.set_xlabel('')
ax2.set_xlabel('')
ax2.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False, bottom=False, top=False, left=True, right=False)
ax2.legend([],frameon=False)
plt.sca(ax2)
plt.gca().invert_yaxis()

corridorder = ['Argentina-Paraguay','Argentina-Uruguay','Argentina-Chile','Argentina-Brazil','Brazil-Paraguay','Brazil-Uruguay']
corridorder_lab=['AR {} PY'.format(u"\u2212"),'AR {} UY'.format(u"\u2212"),'AR {} CL'.format(u"\u2212"),
                  'AR {} BR'.format(u"\u2212"),'PY {} BR'.format(u"\u2212"),'UY {} BR'.format(u"\u2212")]
g = sns.boxplot(data=all_trd_tot[(all_trd_tot.exist_trd==0)].sort_values(by='lz_to'),x='connection',y='net_flow_twh',hue='emis_policy',ax=ax3,
                width=0.7,medianprops={"linewidth":1,'color':'k','label': '_median_','solid_capstyle':'butt'},boxprops={"linewidth":0.3,'edgecolor':'k'},
                flierprops={"marker": "o",'markerfacecolor':'none','markeredgecolor':'k','markersize':3},palette=nopv_pal,saturation=1,zorder=10,
                hue_order=[0,1],order=corridorder)
box_patches = [patch for patch in ax3.patches if type(patch) == matplotlib.patches.PathPatch]
if len(box_patches) == 0:
    box_patches = ax3.artists
num_patches = len(box_patches)
lines_per_boxplot = len(ax3.lines) // num_patches
for i, patch in enumerate(box_patches):
    col = patch.get_facecolor()
    patch.set_edgecolor(col)
    for line in ax3.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
        line.set_color(col)
        line.set_mfc(col)  # facecolor of fliers
        line.set_mec(col)  # edgecolor of fliers
median_lines = [line for line in ax3.get_lines() if line.get_label() == '_median_']
for i, line in enumerate(median_lines):
    line.set_color(nopv_med_c[i])
ax3.set_ylim(0,250)
ax3.set_ylabel('Cumulative Electricity Trade (TWh)',labelpad=10,fontsize=13)
ax3.set_xticklabels(corridorder_lab,fontsize=9)
ax3.set_xlabel('Interconnection',labelpad=8,fontsize=13)
ax3.tick_params(axis='x', which='major', pad=1.5, length=2.5)
ax3.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, bottom=True, top=False, left=True, right=False)
lim_coord_col = '#e4b321'
plt.sca(ax3)
plt.fill_between((0.6,1.4),y1=10,y2=12.2,zorder=1,color=lim_coord_col,alpha=0.5,edgecolor="None")
plt.fill_between((2.6,3.4),y1=2.9,y2=3.8,zorder=1,color=lim_coord_col,alpha=0.5,edgecolor="None")
plt.fill_between((4.6,5.4),y1=13.4,y2=16.2,zorder=1,color=lim_coord_col,alpha=0.5,edgecolor="None")
plt.fill_between((3.6,4.4),y1=160,y2=161.6,zorder=1,color=lim_coord_col,alpha=0.5,edgecolor="None") #hyd_curt*0.9
ax3.set_xlim(-0.5,5.5)
ax3.legend([],frameon=False)
legend_elements = [Line2D([0], [0],label="Reference (No CO$_\mathrm{2}$ Target)", color=nopv_pal[0], lw=7),
                    Line2D([0], [0],label="Mitigation (90% CO$_\mathrm{2}$ Cut)", color=nopv_pal[1], lw=7),
                    Line2D([0], [0],label="Limited Coordination", color=lim_coord_col, alpha=0.5,lw=5)]
ax3.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0,1), frameon=False, fontsize=7.4,handlelength=0.85,labelspacing=0.8)
ax3.annotate('scenario range',xy=(0.062,0.845),xycoords='axes fraction',ha='left',va='top',fontsize=7.4)

ax3.annotate('AR:  Argentina\nBR:  Brazil\nCL:  Chile\nPY:  Paraguay\nUY:  Uruguay',xy=(0.012,0.76),xycoords='axes fraction',ha='left',va='top',fontsize=7.5)
ax3.annotate('a',(0.95,0.95),xycoords='axes fraction',fontsize=14,fontweight='bold',ha='center',va='center')
ax1.annotate('b',(0.935,0.91),xycoords='axes fraction',fontsize=14,fontweight='bold',ha='center',va='center')
ax2.annotate('c',(0.935,0.09),xycoords='axes fraction',fontsize=14,fontweight='bold',ha='center',va='center')
plt.savefig(figpath + 'figure4.pdf', bbox_inches='tight',dpi=600)
plt.show()


#%% IQR RATIOS for emis (Figure 5)

x2=cv_df[cv_df.emis=='Reference'].merge(cv_df[cv_df.emis=='Mitigation'],on=['load_zone','coord'],suffixes=['_ref','_miti'])
x2['IQRR_solar'] = x2['IQR_solar_miti']/x2['IQR_solar_ref']
x2['IQRR_wind'] = x2['IQR_wind_miti']/x2['IQR_wind_ref']
x2['IQRR_hydro'] = x2['IQR_hydro_miti']/x2['IQR_hydro_ref']
x2['IQRR_fossil'] = x2['IQR_fossil_miti']/x2['IQR_fossil_ref']
x2rat = x2[['load_zone','coord','IQRR_solar','IQRR_wind','IQRR_hydro','IQRR_fossil']].copy(deep=True).fillna(0)
x2rat['size'] = [10,10,7,7,8.2,8.2,6.2,6.2,5,5,5,5]
ss, pp, ec = (80,350), {'Limited':'#429DED','Full':'#E4B321'}, ['#0074DA','#AD8200']*6
labs, panlab = ['a','b','c','d'], ['Solar','Wind','Hydro','Fossil']
yaxlab = 'Capacity Mix Uncertainty Ratio\n'+'$\mathrm{IQR_{Mitigation}}$ / $\mathrm{IQR_{Reference}}$'

plt.rcParams.update(plt.rcParamsDefault)
font = FontProperties()
font.set_name('Open Sans')
plt.rcParams["font.family"] = "Arial"
mpl.rcParams['pdf.fonttype'] = 42
fig, ax = plt.subplots(2,2,figsize=(9,7))
plt.subplots_adjust(hspace=0.28,wspace=0.25)
g1=sns.scatterplot(x='load_zone',y='IQRR_solar',data=x2rat,hue='coord',ax=ax[0,0],style='coord',size='size',sizes=ss,palette=pp)
g2=sns.scatterplot(x='load_zone',y='IQRR_wind',data=x2rat,hue='coord',ax=ax[0,1],style='coord',size='size',sizes=ss,palette=pp)
g3=sns.scatterplot(x='load_zone',y='IQRR_hydro',data=x2rat,hue='coord',ax=ax[1,0],style='coord',size='size',sizes=ss,palette=pp)
g4=sns.scatterplot(x='load_zone',y='IQRR_fossil',data=x2rat,hue='coord',ax=ax[1,1],style='coord',size='size',sizes=ss,palette=pp)
for g in [g1,g2,g3,g4]:
    g.collections[0].set_edgecolors(ec)
for i in range(2):
    for j in range(2):
        ax[i,j].axhline(y=1,linestyle='--',color='gray',linewidth=1,zorder=0)
        ax[i,j].set_ylim(-0.03,2.5)
        ax[i,j].set_xlim(-0.54,5.33)
        ax[i,j].set_xlabel('')
        if j==0:
            ax[i,j].set_ylabel(yaxlab, labelpad=10,linespacing=1.5)
        else:
            ax[i,j].set_ylabel('')
            ax[i,j].arrow(5.16,1.07,0,.45,color='#C70000',width=0.03,head_width=0.12,head_length=.102,clip_on=False)
            ax[i,j].arrow(5.16,0.93,0,-.45,color='#C70000',width=0.03,head_width=0.12,head_length=.102,clip_on=False)
            ax[i,j].annotate('Mitigation'+'\n'+'increases'+'\n'+'uncertainty',xy=(1.02,0.442),xycoords='axes fraction',annotation_clip=False,fontsize=9,rotation=90,ha='left',va='bottom')
            ax[i,j].annotate('Mitigation'+'\n'+'decreases'+'\n'+'uncertainty',xy=(1.147,0.3775),xycoords='axes fraction',annotation_clip=False,fontsize=9,rotation=90,ha='right',va='top')
        ax[i,j].tick_params(axis='x',labelsize=10)
        ax[i,j].set_xticks(ax[i,j].get_xticks())
        ax[i,j].set_xticklabels(['System\nWide','ARG','BRA','CHL','PRY','URY'])
        # for lab in ax[i,j].get_xticklabels()[0:1]:
        #     lab.set_fontweight('bold')
        ax[i,j].legend([],frameon=False)
        ax[i,j].annotate(labs[i*2 + j],xy=(0.03,0.91), xycoords='axes fraction', fontsize=12, fontweight='bold',ha='left')
        ax[i,j].annotate(panlab[i*2 + j],xy=(0.97,0.91),xycoords='axes fraction',fontsize=12,ha='right')
h, l = ax[0,0].get_legend_handles_labels()
leg = ax[1,0].legend(h[1:3], ['Limited Coordination','Full Coordination'],
               title=None,loc='center',bbox_to_anchor=(1.1,-0.31),handletextpad=0.1, ncol=2, frameon=False)
leg.legend_handles[0].set(markersize = 17,mec=ec[0])
leg.legend_handles[1].set(markersize = 17,mec=ec[1])
leg.texts[0].set_x(6)
leg.texts[1].set_x(6)

plt.savefig(figpath+'figure5.pdf',dpi=600,bbox_inches='tight')
plt.show()



### IQR RATIOS for coord (for SI) ###
x1=cv_df[cv_df.coord=='Limited'].merge(cv_df[cv_df.coord=='Full'],on=['load_zone','emis'],suffixes=['_lim','_full'])
x1['IQRR_solar'] = x1['IQR_solar_full']/x1['IQR_solar_lim']
x1['IQRR_wind'] = x1['IQR_wind_full']/x1['IQR_wind_lim']
x1['IQRR_hydro'] = x1['IQR_hydro_full']/x1['IQR_hydro_lim']
x1['IQRR_fossil'] = x1['IQR_fossil_full']/x1['IQR_fossil_lim']
xrat = x1[['load_zone','emis','IQRR_solar','IQRR_wind','IQRR_hydro','IQRR_fossil']].copy(deep=True).fillna(0)
xrat['size'] = [10,10,7,7,8.2,8.2,6.2,6.2,5,5,5,5]
ss, pp, ec = (80,350), {'Reference':'#429DED','Mitigation':'#E4B321'}, ['#0074DA','#AD8200']*6
labs, panlab = ['a','b','c','d'], ['Solar','Wind','Hydro','Fossil']
yaxlab = 'Capacity Mix Uncertainty Ratio\n'+'$\mathrm{IQR_{FullCoord}}$ / $\mathrm{IQR_{LimCoord}}$'

plt.rcParams.update(plt.rcParamsDefault)
font = FontProperties()
font.set_name('Open Sans')
plt.rcParams["font.family"] = "Arial"
mpl.rcParams['pdf.fonttype'] = 42
fig, ax = plt.subplots(2,2,figsize=(9,7))
plt.subplots_adjust(hspace=0.28,wspace=0.25)

g1=sns.scatterplot(x='load_zone',y='IQRR_solar',data=xrat,hue='emis',ax=ax[0,0],style='emis',size='size',sizes=ss,palette=pp)
g2=sns.scatterplot(x='load_zone',y='IQRR_wind',data=xrat,hue='emis',ax=ax[0,1],style='emis',size='size',sizes=ss,palette=pp)
g3=sns.scatterplot(x='load_zone',y='IQRR_hydro',data=xrat,hue='emis',ax=ax[1,0],style='emis',size='size',sizes=ss,palette=pp)
g4=sns.scatterplot(x='load_zone',y='IQRR_fossil',data=xrat,hue='emis',ax=ax[1,1],style='emis',size='size',sizes=ss,palette=pp)
for g in [g1,g2,g3,g4]:
    g.collections[0].set_edgecolors(ec)
for i in range(2):
    for j in range(2):
        ax[i,j].axhline(y=1,linestyle='--',color='gray',linewidth=1,zorder=0)
        ax[i,j].set_ylim(-0.04,3.8)
        ax[i,j].set_xlim(-0.54,5.33)
        ax[i,j].set_xlabel('')
        if j==0:
            ax[i,j].set_ylabel(yaxlab, labelpad=10,linespacing=1.5)
        else:
            ax[i,j].set_ylabel('')
            ax[i,j].arrow(5.48,1.08,0,.7,color='#C70000',width=0.02,head_width=0.11,head_length=.1,clip_on=False)
            ax[i,j].arrow(5.48,0.92,0,-.7,color='#C70000',width=0.02,head_width=0.11,head_length=.1,clip_on=False)
            ax[i,j].annotate('Coordination'+'\n'+'increases'+'\n'+'uncertainty',xy=(1.055,0.3),xycoords='axes fraction',annotation_clip=False,fontsize=6,rotation=90,ha='left',va='bottom')
            ax[i,j].annotate('Coordination'+'\n'+'decreases'+'\n'+'uncertainty',xy=(1.136,0.24),xycoords='axes fraction',annotation_clip=False,fontsize=6,rotation=90,ha='right',va='top')
        ax[i,j].tick_params(axis='x',labelsize=10)
        ax[i,j].set_xticks(ax[i,j].get_xticks())
        ax[i,j].set_xticklabels(['System\nWide','ARG','BRA','CHL','PRY','URY'])
        # for lab in ax[i,j].get_xticklabels()[0:1]:
        #     lab.set_fontweight('bold')
        ax[i,j].legend([],frameon=False)
        ax[i,j].annotate(labs[i*2 + j],xy=(0.03,0.91), xycoords='axes fraction', fontsize=12, fontweight='bold',ha='left')
        ax[i,j].annotate(panlab[i*2 + j],xy=(0.97,0.91),xycoords='axes fraction',fontsize=12,ha='right')
h, l = ax[0,0].get_legend_handles_labels()
leg = ax[1,0].legend(h[1:3], ['Reference','Mitigation'],
                title=None,loc='center',bbox_to_anchor=(1.1,-0.31),handletextpad=0.1, ncol=2, frameon=False)
leg.legend_handles[0].set(markersize = 17,mec=ec[0])
leg.legend_handles[1].set(markersize = 17,mec=ec[1])
leg.texts[0].set_x(6)
leg.texts[1].set_x(6)

plt.savefig(figpath+'figure_S11.png',dpi=400,bbox_inches='tight')
plt.show()

#%% MAP AND VRE PREFERENCE (For SI - GEOPANDAS NEEDED)

import geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.path import Path
import matplotlib as mpl

## prep ###
countries = geopandas.read_file(datapath + 'map_data/Countries_five.shp')


# figure #
plt.rcParams.update(plt.rcParamsDefault)
color4 = ['#D81B60','#004D40','#FFC107','#1E88E5']
fig = plt.figure(constrained_layout=False, figsize = (8.5,6))
gs = fig.add_gridspec(2,1, height_ratios=[1.1,1], hspace=.1)
gs1 = gs[0].subgridspec(1,4, width_ratios=[1,0.001,1,0.15], wspace=.09)
gs2 = gs[1].subgridspec(1,2, width_ratios=[1,0.1], wspace=0)
axp = fig.add_subplot(gs2[0])
axm1,axm3 = fig.add_subplot(gs1[2]),fig.add_subplot(gs1[0])
cmin, cmax, wmin, wmax =0.13, 0.40, 0, 0.7
countries.plot(ax=axm1, color='w',zorder=0,edgecolor='k',linewidth=.15)
g1 = axm1.scatter(x=wind_cf['Longitude'],y=wind_cf['Latitude'],c=wind_cf['all_avg'],marker=path,cmap='viridis',s=12,vmin=wmin,vmax=wmax,edgecolors='none')
plt.sca(axm1)
plt.xlim(-77,-30)
plt.ylim(-57,6)
plt.legend('',frameon=False)
plt.title('Mean Wind\nCapacity Factor', fontsize=11)
countries.plot(ax=axm1, color='none',zorder=10,edgecolor='k',linewidth=.15,linestyle='--',alpha=0.5)
plt.axis('off')
cmap = plt.get_cmap('viridis')
w_cmap=truncate_colormap(cmap,(wind_cf['all_avg'].min()-wmin)/(wmax-wmin),
                                (wind_cf['all_avg'].max()-wmin)/(wmax-wmin))
divider1 = make_axes_locatable(g1.axes)
ax1 = axm1.inset_axes([0.68,0,0.061,0.47],zorder=3)
cbar1 = plt.colorbar(plt.cm.ScalarMappable(cmap=w_cmap,norm=mpl.colors.Normalize(vmin=wind_cf['all_avg'].min(),vmax=wind_cf['all_avg'].max())),
              ticks=np.linspace(0.1,0.6,6),cax=ax1)
cbar1.ax.tick_params(labelsize=9)
countries.plot(ax=axm3, color='w',zorder=0,edgecolor='k',linewidth=.15)
g2 = axm3.scatter(x=solar_cf['Longitude'],y=solar_cf['Latitude'],c=solar_cf['all_avg'],marker=path,cmap='plasma',s=12,vmin=cmin,vmax=cmax,edgecolors='none')
plt.sca(axm3)
plt.xlim(-77,-30)
plt.ylim(-57,6)
plt.legend('',frameon=False)
plt.title('Mean Solar PV\nCapacity Factor', fontsize=11)
countries.plot(ax=axm3, color='none',zorder=10,edgecolor='k',linewidth=.15,linestyle='--',alpha=0.5)
plt.axis('off')
cmap = plt.get_cmap('plasma')
s_cmap=truncate_colormap(cmap,(solar_cf['all_avg'].min()-cmin)/(cmax-cmin),
                                (solar_cf['all_avg'].max()-cmin)/(cmax-cmin))
divider = make_axes_locatable(g2.axes)
ax3 = axm3.inset_axes([0.68,0,0.06,0.47],zorder=3)
cbar3 = plt.colorbar(plt.cm.ScalarMappable(cmap=s_cmap,norm=mpl.colors.Normalize(vmin=solar_cf['all_avg'].min(),vmax=solar_cf['all_avg'].max())),
              ticks=np.linspace(0.20,0.32,4),cax=ax3)
cbar3.ax.tick_params(labelsize=9)

my_colours = {0:'#7DF5E0',1:'#008770',10:'#FF96BC',11:'#CA1658',110:'#FFCB31',111:'#AD8200',100:'#84C5FF',101:'#0074DA'} # [green,pink,yellow,blue]
my_colours = ['#7DF5E0','#008770','#FF96BC','#CA1658','#FFCB31','#AD8200','#84C5FF','#0074DA']*5 # [green,pink,yellow,blue]
med_colours = ['#008770','#7DF5E0','#CA1658','#FF96BC','#AD8200','#FFCB31','#0074DA','#84C5FF']*5
plt.rcParams.update(plt.rcParamsDefault)
color4 = ['#D81B60','#004D40','#FFC107','#1E88E5']
df = padf.copy(deep=True)
df['group'] = df['group'].astype(str)
g = sns.boxplot(data=df[df.load_zone=='system-wide'],x='pref',y='load_zone',hue='group',width=0.95,whis=(0,100),ax=axp,hue_order=['10','11','0','1','110','111','100','101'],
                        medianprops={"linewidth": 1,'color':'k','label':'_median_','solid_capstyle':'butt'},saturation=1,
                        boxprops={"linewidth": .3,'edgecolor':'k'},whiskerprops={"linewidth": 3,'color':'k'},capprops={"linewidth": 0,'color':'k'})
boxes = axp.findobj(matplotlib.patches.PathPatch)
for color, box in zip(my_colours, boxes):
    box.set_facecolor(color)
box_patches = [patch for patch in axp.patches if type(patch) == matplotlib.patches.PathPatch]
if len(box_patches) == 0:
    box_patches = axp.artists
num_patches = len(box_patches)
if num_patches > 0:
    lines_per_boxplot = len(axp.lines) // num_patches
    for k, patch in enumerate(box_patches):
        col = patch.get_facecolor()
        patch.set_edgecolor(col)
        for line in axp.lines[k * lines_per_boxplot: (k + 1) * lines_per_boxplot]:
            line.set_color(col)
            line.set_mfc(col)  # facecolor of fliers
            line.set_mec(col)  # edgecolor of fliers
    median_lines = [line for line in axp.get_lines() if line.get_label() == '_median_']
    for k, line in enumerate(median_lines):
        line.set_color(med_colours[k])
plt.legend([],frameon=False)
axp.set_xlim(0,1)
axp.set_ylabel('')
axp.set_xlabel('← Preference for Solar                     Preference for Wind →',labelpad=12,fontsize=13)
axp.axvline(x=0.5,color='#A0A0A0',linewidth=0.8,linestyle='--')
axp.set_xticks(np.linspace(0,1,11))
axp.set_xticklabels(['100%','90%','80%','70%','60%','50%','60%','70%','80%','90%','100%'],fontsize=10.5)
axp.set_yticks([-0.36,-0.12,0.12,0.36])
axp.set_yticklabels(["No $\mathrm{CO_{2}}$ Target +"+'\nLimited Coordination',"No $\mathrm{CO_{2}}$ Target +"+'\nFull Coordination',
                      "90% $\mathrm{CO_{2}}$ Cut +"+' \nLimited Coordination',"90% $\mathrm{CO_{2}}$ Cut +"+'\nFull Coordination'],fontsize=10.5)
pa1 = Patch(facecolor=my_colours[0], edgecolor=med_colours[0],linewidth=0.5)
pa2 = Patch(facecolor=my_colours[2], edgecolor=med_colours[2],linewidth=0.5)
pa3 = Patch(facecolor=my_colours[4], edgecolor=med_colours[4],linewidth=0.5)
pa4 = Patch(facecolor=my_colours[6], edgecolor=med_colours[6],linewidth=0.5)
pb1 = Patch(facecolor=my_colours[1], edgecolor=med_colours[1],linewidth=0.5)
pb2 = Patch(facecolor=my_colours[3], edgecolor=med_colours[3],linewidth=0.5)
pb3 = Patch(facecolor=my_colours[5], edgecolor=med_colours[5],linewidth=0.5)
pb4 = Patch(facecolor=my_colours[7], edgecolor=med_colours[7],linewidth=0.5)
plt.sca(axp)
l1 = plt.legend(handles=[pa1, pb1, pa2, pb2, pa3, pb3, pa4, pb4],
          labels=['', '', '', '', '', '', '1-Axis Solar', 'Fixed Solar'],
          ncol=4, handletextpad=0.5, handlelength=1.2, columnspacing=-0.5,
          loc='center', fontsize=9.5,frameon=False,bbox_to_anchor=(0.38,-0.47))
axp.add_artist(l1)
axm1.annotate('b',(-81,-1),ha='left',fontsize=11,fontweight='bold', annotation_clip=False)
axm3.annotate('a',(-81,-1),ha='left',fontsize=11,fontweight='bold', annotation_clip=False)
axp.annotate('c',(0.01,-0.41),ha='left',fontsize=11,fontweight='bold')

ax3 = fig.add_subplot(gs[1,:])
g3 = plt.boxplot([-29,-20,-12,-4,5],whiskerprops={'clip_on':False,'linewidth':1.3,'solid_capstyle':'butt'},
                  capprops={'clip_on':False,"linewidth":0,},medianprops={'clip_on':False,'linewidth':1.1},boxprops={'clip_on':False,'linewidth':1.3})
for median in g3['medians']:
    median.set_color('black')
ax3.set_xlim(-1.5,3.4)
ax3.set_ylim(110,370)
plt.sca(ax3)
plt.axis('off')
ax3.annotate('Variability due to Turbine Type',(1.14,-18),ha='left',fontsize=9.5, annotation_clip=False)

plt.savefig(figpath + 'figure_S12.png', bbox_inches='tight',dpi=1200)
plt.show()

#%% SOLAR AND WIND CF (GEOPANDAS) (SI)

### wind CF ###
plt.rcParams.update(plt.rcParamsDefault)
fig = plt.figure(constrained_layout=False, figsize = (12,7))
gs = fig.add_gridspec(2,5, width_ratios=[1,1,1,1,1], height_ratios=[1,1], wspace=-0.1)
turb_cols=['cf_vestas_2','cf_vestas_7','cf_vestas_3','cf_siemens_36','cf_siemens_23','cf_GE_25','cf_GE_15','cf_gamesa_2','cf_enercon_7','cf_enercon_3']
cmin, cmax, clmap, cb_ticks = 0, 0.75, 'viridis', np.linspace(0.1,0.7,7) # actual min/max = 0.06175, 0.6972
for i in range(len(turb_cols)):
    ax = fig.add_subplot(gs[i])
    countries.plot(ax=ax, color='none',zorder=0,edgecolor='k',linewidth=.15)
    g=ax.scatter(x=wind_cf['Longitude'],y=wind_cf['Latitude'],c=wind_cf[turb_cols[i]],marker=path,cmap=clmap,s=7.5,vmin=cmin,vmax=cmax,edgecolors='none')
    plt.sca(ax)
    plt.xlim(-77,-30)
    plt.ylim(-57,6)
    plt.legend('',frameon=False)
    plt.title('Turbine #{}'.format(str(i+1)), fontsize=13)
    countries.plot(ax=ax, color='none',zorder=10,edgecolor='k',linewidth=.15,linestyle='--',alpha=0.5)
    plt.axis('off')
    
    divider = make_axes_locatable(g.axes)
    cax = divider.append_axes("right", size="7%", pad=0.15)
    cmap = plt.get_cmap(clmap)
    w_cmap=truncate_colormap(cmap,(wind_cf[turb_cols].min().min()-cmin)/(cmax-cmin),(wind_cf[turb_cols].max().max()-cmin)/(cmax-cmin))
    cbar=plt.colorbar(plt.cm.ScalarMappable(cmap=clmap,norm=mpl.colors.Normalize(vmin=wind_cf[turb_cols].min().min(),
                                                vmax=wind_cf[turb_cols].max().max())),ticks=cb_ticks,cax=cax)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_ylabel('Mean Annual Capacity Factor',labelpad=13)
    if i != 4:
        plt.axis('off')
        cbar.ax.set_visible(False)

plt.savefig(figpath + 'figure_S15.png', bbox_inches='tight',dpi=1000)
plt.show()

### solar CF ###
plt.rcParams.update(plt.rcParamsDefault)
fig = plt.figure(constrained_layout=False)#, figsize = (6,4))
gs = fig.add_gridspec(1,2, width_ratios=[1,1], wspace=-0.1)
axm1 = fig.add_subplot(gs[0])
axm2 = fig.add_subplot(gs[1])
cmin, cmax = 0.13, 0.43

countries.plot(ax=axm1, color='none',zorder=0,edgecolor='k',linewidth=.15)
g1 = axm1.scatter(x=solar_cf['Longitude'],y=solar_cf['Latitude'],c=solar_cf['cf_1axis'],marker=path,cmap='plasma',s=12.6,vmin=cmin,vmax=cmax,edgecolors='none')
plt.sca(axm1)
plt.xlim(-77,-30)
plt.ylim(-57,6)
plt.legend('',frameon=False)
plt.title('1-axis PV Project Sites', fontsize=13)
countries.plot(ax=axm1, color='none',zorder=10,edgecolor='k',linewidth=.15,linestyle='--',alpha=0.5)
plt.axis('off')

countries.plot(ax=axm2, color='none',zorder=0,edgecolor='k',linewidth=.15)
g2 = axm2.scatter(x=solar_cf['Longitude'],y=solar_cf['Latitude'],c=solar_cf['cf_fixed'],marker=path,cmap='plasma',s=12.6,vmin=cmin,vmax=cmax,edgecolors='none')
plt.sca(axm2)
plt.xlim(-77,-30)
plt.ylim(-57,6)
plt.legend('',frameon=False)
plt.title('Fixed PV Project Sites', fontsize=13)
countries.plot(ax=axm2, color='none',zorder=10,edgecolor='k',linewidth=.15,linestyle='--',alpha=0.5)
plt.axis('off')

divider = make_axes_locatable(g1.axes)
cax = divider.append_axes("right", size="7%", pad=0.15)
cbar1 = plt.colorbar(plt.cm.ScalarMappable(cmap='plasma',norm=mpl.colors.Normalize(vmin=cmin,vmax=cmax)),ticks=np.linspace(0.15,0.4,6),cax=cax)
cbar1.ax.tick_params(labelsize=10)
plt.axis('off')
cbar1.ax.set_visible(False)
cmap = plt.get_cmap('plasma')
s_cmap=truncate_colormap(cmap,(solar_cf[['cf_1axis','cf_fixed']].min().min()-cmin)/(cmax-cmin),(solar_cf[['cf_1axis','cf_fixed']].max().max()-cmin)/(cmax-cmin))
divider = make_axes_locatable(g2.axes)
cax = divider.append_axes("right", size="7%", pad=0.15)
cbar2 = plt.colorbar(plt.cm.ScalarMappable(cmap=s_cmap,norm=mpl.colors.Normalize(vmin=solar_cf[['cf_1axis','cf_fixed']].min().min(),\
                                                    vmax=solar_cf[['cf_1axis','cf_fixed']].max().max())),ticks=np.linspace(0.15,0.4,6),cax=cax)
cbar2.ax.tick_params(labelsize=10)
cbar2.ax.set_ylabel('Mean Annual Capacity Factor',labelpad=13)

plt.savefig(figpath + 'figure_S16.png', bbox_inches='tight',dpi=1000)
plt.show()

#%% NEW CAPACITY REQUIREMENTS BOXPLOTS (For SI)

plt.rcParams.update(plt.rcParamsDefault)
font = FontProperties()
font.set_name('Open Sans')
plt.rcParams["font.family"] = "Arial"
fig = plt.figure(constrained_layout=False, figsize = (13,9))
gs = fig.add_gridspec(4,6, height_ratios=[1,1,1,1], width_ratios=[1,1,1,1,1,1], hspace=.08, wspace=0.26)
location_legend = [-1.15,-0.42]
my_colours = ['#7DF5E0','#FF96BC','#FFCB31','#84C5FF','#008770','#CA1658','#AD8200','#0074DA']
med_colours = ['#008770','#CA1658','#AD8200','#0074DA','#7DF5E0','#FF96BC','#FFCB31','#84C5FF']

# plot capacities
for i in range(len(cap_tech)):
    for j in range(len(panels)):
        ax = fig.add_subplot(gs[i,j])
        if ((i!=2) | (j<4)):
            g = sns.boxplot(data=new_cap[new_cap.load_zone==panels[j]],x='pol',y=cap_tech[i]+'_new',hue='fixed_pv', width=0.7,
                            medianprops={"linewidth": .7,'color':'k','label':'_median_','solid_capstyle':'butt'},saturation=1,
                            flierprops={"marker": "o",'markerfacecolor':'none','markeredgecolor':'k','markersize':2},
                            boxprops={"linewidth": .3,'edgecolor':'k'})#,whiskerprops={"linewidth": .7,'color':'k'},capprops={"linewidth": .7,'color':'k'})
        ax.legend([],frameon=False)
        boxes = ax.findobj(matplotlib.patches.PathPatch)
        for color, box in zip(my_colours, boxes):
            box.set_facecolor(color)
        box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
        if len(box_patches) == 0:
            box_patches = ax.artists
        num_patches = len(box_patches)
        if num_patches > 0:
            lines_per_boxplot = len(ax.lines) // num_patches
            for k, patch in enumerate(box_patches):
                col = patch.get_facecolor()
                patch.set_edgecolor(col)
                for line in ax.lines[k * lines_per_boxplot: (k + 1) * lines_per_boxplot]:
                    line.set_color(col)
                    line.set_mfc(col)  # facecolor of fliers
                    line.set_mec(col)  # edgecolor of fliers
            median_lines = [line for line in ax.get_lines() if line.get_label() == '_median_']
            for k, line in enumerate(median_lines):
                line.set_color(med_colours[k])
        if i==0:
            ax.set_title(panels[j],fontsize=14)
        if j==0:
            [x.set_linewidth(1.5) for x in ax.spines.values()]
            ax.set_ylabel('New ' + cap_tech[i] + ' Capacity (GW)',fontsize=9)
        if j==0:
            ax.set_ylim(0,331.59)
        if j==1:
            ax.set_ylim(0,111.82)
        if j==2:
            ax.set_ylim(0,209.94)
        if j==3:
            ax.set_ylim(0,35.98)
        if j==4:
            ax.set_ylim(0,17.86)
        if j==5:
            ax.set_ylim(0,9.78)
        low,high = ax.get_ylim()[0],ax.get_ylim()[1]
        ax.set_ylim(high*-0.015,high+((low-high*-0.015)*0.015))
        if j!=0:
            ax.set_ylabel('')
        if i<3:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(["No $\mathrm{CO_{2}}$ Target"+'\nLimited Coord.',"No $\mathrm{CO_{2}}$ Target"+'\nFull Coord.',
                                  "90% $\mathrm{CO_{2}}$ Cut"+' \nLimited Coord.',"90% $\mathrm{CO_{2}}$ Cut"+'\nFull Coord.'],fontsize=8,rotation=90)

        if ((i==2)&(j>3)):
            ax.set_yticks([])
            ax.set_yticklabels([])
        if j<3:
            ax.tick_params(axis='y',pad=2)
        ax.set_xlabel('')
pa1 = Patch(facecolor=my_colours[0], edgecolor=med_colours[0],linewidth=0.5)
pa2 = Patch(facecolor=my_colours[1], edgecolor=med_colours[1],linewidth=0.5)
pa3 = Patch(facecolor=my_colours[2], edgecolor=med_colours[2],linewidth=0.5)
pa4 = Patch(facecolor=my_colours[3], edgecolor=med_colours[3],linewidth=0.5)
pb1 = Patch(facecolor=my_colours[4], edgecolor=med_colours[4],linewidth=0.5)
pb2 = Patch(facecolor=my_colours[5], edgecolor=med_colours[5],linewidth=0.5)
pb3 = Patch(facecolor=my_colours[6], edgecolor=med_colours[6],linewidth=0.5)
pb4 = Patch(facecolor=my_colours[7], edgecolor=med_colours[7],linewidth=0.5)
ax.legend(handles=[pa1, pb1, pa2, pb2, pa3, pb3, pa4, pb4],
          labels=['', '', '', '', '', '', '1-Axis Solar', 'Fixed Solar'],
          ncol=4, handletextpad=0.5, handlelength=1.2, columnspacing=-0.5,
          loc='center', fontsize=9,frameon=False,bbox_to_anchor=(-2.63,-0.8))
fig.savefig(figpath + 'figure_S1.png', bbox_inches='tight', dpi=600)
plt.show()

#%% GENERATION SHARE BAND PLOTS (For SI)

plt.rcParams.update(plt.rcParamsDefault)
font = FontProperties()
font.set_name('Open Sans')
plt.rcParams["font.family"] = "Arial"
fig = plt.figure(constrained_layout=False, figsize = (9,10.5))
#set constrained_layout=False and use wspace and hspace params to set amount of width/height reserved for space between subplots
gs = fig.add_gridspec(4,6, height_ratios=[1,1,1,1], width_ratios=[1,1,1,1,1,1], hspace=.08, wspace=0.13)
order1 = [0,1,2,3,4,5,6,7,8,9]
order = [0,2,3,1]
cdict = dict(zip(sc_groups,['#004D40', '#D81B60', '#FFC107', '#1E88E5']))
z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
      21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80]
# plot generation shares
for i in range(len(gen_share_co_list)):
    for j in range(len(f5_techs)):
        ax = fig.add_subplot(gs[j,i])
        for k in range(len(scs80)):
            ydata = gen_share_co_list[i][(gen_share_co_list[i]['tech']==f5_techs[j])&(gen_share_co_list[i]['scs']==scs80[k])].loc[:,2020:2050].values.tolist()[0]
            ax.plot(np.arange(len(ydata)),ydata, linewidth=0.3,linestyle=style_df_4color.loc[scs80[k]].styl,
                                color=style_df_4color.loc[scs80[k]]['2color'],label=style_df_4color.loc[scs80[k]].real_name, zorder = z[k])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel('{} Gen. Share'.format(f5_techs[j]),fontsize=12)
            plt.sca(ax)
            plt.yticks((0,25,50,75,100),('0%','25%','50%','75%','100%'))
            [x.set_linewidth(1.5) for x in ax.spines.values()]
            if j == 0:
                ax.set_title('Region-wide')#,fontweight='bold')
        ax.set_ylim(-5,105)
        ax.set_xticks([0,2,4,6])
        ax.set_xticklabels([])

        ax.legend([], frameon=False)
        if j == 3:
            ax.set_xticks([0,2,4,6])
            ax.set_xticklabels(('2020','2030','2040','2050'))
            plt.setp(ax.get_xticklabels(), rotation=45, fontsize=9, ha='right', rotation_mode='anchor')
        if j < 3:
            ax.tick_params('x', length=2, which='major')
        if (j==0)&(i>0):
            ax.set_title('{}'.format(mercosur[i-1]), fontsize=12)
        for l in range(len(sc_groups)):
            ax.plot(np.arange(7),mins_co[i].loc[(cap_tech[j],sc_groups[l])].tolist()[1:8], linewidth=.5,
                                color=cdict[sc_groups[l]], zorder = 41)
            ax.plot(np.arange(7),maxes_co[i].loc[(cap_tech[j],sc_groups[l])].tolist()[1:8], linewidth=.5,
                                color=cdict[sc_groups[l]], zorder = 41)
            ax.fill_between(np.arange(7),mins_co[i].loc[(cap_tech[j],sc_groups[l])].tolist()[1:8],maxes_co[i].loc[(cap_tech[j],sc_groups[l])].tolist()[1:8],
                                alpha=0.4,color=cdict[sc_groups[l]], zorder=85*(order[l]+1))
custom_lines = [Line2D([0], [0], color=color4[1], lw=10, alpha=0.6),
                Line2D([0], [0], color=color4[0], lw=10, alpha=0.6),
                Line2D([], [], color='black', lw=1, linestyle='-'),
                Line2D([0], [0], color=color4[2], lw=10, alpha=0.6),
                Line2D([0], [0], color=color4[3], lw=10, alpha=0.6),
                Line2D([], [], color='black', lw=1, linestyle='--')]
ax.legend(custom_lines,['No $\mathrm{CO_{2}}$ Target + Limited Coordination','No $\mathrm{CO_{2}}$ Target + Full Coordination','1-Axis Solar',
                              '90% $\mathrm{CO_{2}}$ Cut + Limited Coordination','90% $\mathrm{CO_{2}}$ Cut + Full Coordination','Fixed Solar'],\
                              loc='center', bbox_to_anchor=[-2.3,-0.5], fontsize=10, frameon=False, ncol=2)
plt.savefig(figpath + 'figure_S2.png', bbox_inches='tight', dpi=1000)
plt.show()

#%% COST COMPONENT BOXPLOTS BY COUNTRY (For SI)

plt.rcParams.update(plt.rcParamsDefault)
font = FontProperties()
font.set_name('Open Sans')
plt.rcParams["font.family"] = "Arial"
fig = plt.figure(constrained_layout=False, figsize = (13.5,15))
gs = fig.add_gridspec(4,6, height_ratios=[1,1,1,1], width_ratios=[1,1,1,1,1,1], hspace=.1, wspace=0.25)
cost_cols = ['capacity_cost','operations_cost_total','split_tx_cost','new_total_cost']
cost_labels = ['Gen. Capacity Costs ($B)','Grid Operations Costs ($B)','Transmission Costs ($B)','Total Costs ($B)']
ylims = [[(100,300),(0,60),(0,200),(0,40),(0,10),(0,10)],
          [(0,200),(0,60),(0,200),(0,40),(0,10),(0,10)],
          [(-0.03,10),(-0.03,10),(-0.03,10),(-0.03,10),(-0.03,10),(-0.03,10)],
          [(280,370),(45,75),(190,252),(25,41),(0,10),(0,10)]]

# plot capacities
for i in range(len(cost_cols)):
    for j in range(len(panels)):
        ax = fig.add_subplot(gs[i,j])
        g = sns.boxplot(data=split_tx_cost_df[split_tx_cost_df.load_zone==panels[j]],x='pol',y=cost_cols[i],hue='fixed_pv', width=0.7,
                        medianprops={"linewidth": .7,'color':'k','label':'_median_','solid_capstyle':'butt'},saturation=1,
                        flierprops={"marker": "o",'markerfacecolor':'none','markeredgecolor':'k','markersize':2},
                        boxprops={"linewidth": .3,'edgecolor':'k'})#,whiskerprops={"linewidth": .7,'color':'k'},capprops={"linewidth": .7,'color':'k'})
        ax.legend([],frameon=False)
        boxes = ax.findobj(matplotlib.patches.PathPatch)
        for color, box in zip(my_colours, boxes):
            box.set_facecolor(color)
        box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
        if len(box_patches) == 0:
            box_patches = ax.artists
        num_patches = len(box_patches)
        if num_patches > 0:
            lines_per_boxplot = len(ax.lines) // num_patches
            for k, patch in enumerate(box_patches):
                col = patch.get_facecolor()
                patch.set_edgecolor(col)
                for line in ax.lines[k * lines_per_boxplot: (k + 1) * lines_per_boxplot]:
                    line.set_color(col)
                    line.set_mfc(col)  # facecolor of fliers
                    line.set_mec(col)  # edgecolor of fliers
            median_lines = [line for line in ax.get_lines() if line.get_label() == '_median_']
            for k, line in enumerate(median_lines):
                line.set_color(med_colours[k])
        if i==0:
            ax.set_title(panels[j],fontsize=15)
            # if j==0:
            #     ax.set_title(panels[j],fontsize=15,fontweight='bold')
        if j==0:
            [x.set_linewidth(1.5) for x in ax.spines.values()]
            ax.set_ylabel(cost_labels[i],fontsize=12)
        else:
            ax.set_ylabel('')
        if i<3:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(["No $\mathrm{CO_{2}}$ Target"+'\nLimited Coord.',"No $\mathrm{CO_{2}}$ Target"+'\nFull Coord.',
                                  "90% $\mathrm{CO_{2}}$ Cut"+' \nLimited Coord.',"90% $\mathrm{CO_{2}}$ Cut"+'\nFull Coord.'],
                                fontsize=8,rotation=90,linespacing=1)
        if j<3:
            ax.tick_params(axis='y',pad=2)
        ax.set_xlabel('')
        ax.set_ylim(ylims[i][j])
pa1 = Patch(facecolor=my_colours[0], edgecolor=med_colours[0],linewidth=0.5)
pa2 = Patch(facecolor=my_colours[1], edgecolor=med_colours[1],linewidth=0.5)
pa3 = Patch(facecolor=my_colours[2], edgecolor=med_colours[2],linewidth=0.5)
pa4 = Patch(facecolor=my_colours[3], edgecolor=med_colours[3],linewidth=0.5)
pb1 = Patch(facecolor=my_colours[4], edgecolor=med_colours[4],linewidth=0.5)
pb2 = Patch(facecolor=my_colours[5], edgecolor=med_colours[5],linewidth=0.5)
pb3 = Patch(facecolor=my_colours[6], edgecolor=med_colours[6],linewidth=0.5)
pb4 = Patch(facecolor=my_colours[7], edgecolor=med_colours[7],linewidth=0.5)
ax.legend(handles=[pa1, pb1, pa2, pb2, pa3, pb3, pa4, pb4],
          labels=['', '', '', '', '', '', '1-Axis Solar', 'Fixed Solar'],
          ncol=4, handletextpad=0.5, handlelength=1.2, columnspacing=-0.5,
          loc='center', fontsize=9,frameon=False,bbox_to_anchor=(-2.56,-0.5))
fig.savefig(figpath + 'figure_S3.png', bbox_inches='tight', dpi=600)
plt.show()

#%% NEW TRANSMISSION CAPACITY BOXPLOTS (For SI)

tx_tidy = pd.read_csv(datapath + 'new_tx_cap.csv')
tx_tidy['group'] = tx_tidy['group'].map({0:'000',1:'001',100:'100',101:'101'})
new_med_c = ['#CA1658','#CA1658','#CA1658','#FF96BC','#FF96BC','#FF96BC',
             '#0074DA','#0074DA','#0074DA','#84C5FF','#84C5FF','#84C5FF'] #if using newer seaborn
plt.rcParams.update(plt.rcParamsDefault)
font = FontProperties()
font.set_name('Open Sans')
plt.rcParams["font.family"] = "Arial"
fig, ax = plt.subplots(figsize=(5.5,4.5))
g = sns.boxplot(data=tx_tidy,x='transmission_line',y='new_build_transmission_capacity_mw',hue='group',width=0.8, ax=ax, palette=palette,
                medianprops={"linewidth": 1,'color':'k','label':'_median_','solid_capstyle':'butt'},saturation=1,
                flierprops={"marker": "o",'markerfacecolor':'none','markeredgecolor':'k','markersize':2},
                boxprops={"linewidth": .6,'edgecolor':'k'},hue_order=['000','001','100','101'])

ax.legend([],frameon=False)

box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
if len(box_patches) == 0:
    box_patches = ax.artists
num_patches = len(box_patches)
lines_per_boxplot = len(ax.lines) // num_patches
for i, patch in enumerate(box_patches):
    col = patch.get_facecolor()
    patch.set_edgecolor(col)
    for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
        line.set_color(col)
        line.set_mfc(col)  # facecolor of fliers
        line.set_mec(col)  # edgecolor of fliers
median_lines = [line for line in ax.get_lines() if line.get_label() == '_median_']
for i, line in enumerate(median_lines):
    line.set_color(new_med_c[i])

ax.set_ylabel('Total New Transmission Capacity (GW)',labelpad=10,fontsize=12)
ax.set_xticklabels(['Argentina{}Uruguay'.format(u"\u2212"),'Argentina{}Brazil'.format(u"\u2212"),'Argentina{}Chile'.format(u"\u2212")])
ax.set_xlabel('')

legend_elements = [Line2D([0], [0],label="No $\mathrm{CO_{2}}$ Target" + ' + 1-Axis Solar', color=palette['000'], lw=8),
                    Line2D([0], [0],label="No $\mathrm{CO_{2}}$ Target" + ' + Fixed Solar', color=palette['001'], lw=8),
                    Line2D([0], [0],label="90% $\mathrm{CO_{2}}$ Cut" + ' + 1-Axis Solar', color=palette['100'], lw=8),
                    Line2D([0], [0],label="90% $\mathrm{CO_{2}}$ Cut" + ' + Fixed Solar', color=palette['101'], lw=8)]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0,1.01), frameon=False, fontsize=10,handlelength=0.9)

ax.set_ylim(-0.03,20)
plt.savefig(figpath + 'figure_S8.png', bbox_inches='tight',dpi=600)
plt.show()

#%% BATTERY, CURTAILMENT, EMISSIONS BY COUNTRY (For SI)

plt.rcParams.update(plt.rcParamsDefault)
font = FontProperties()
font.set_name('Open Sans')
plt.rcParams["font.family"] = "Arial"
my_colours = ['#7DF5E0','#FF96BC','#FFCB31','#84C5FF','#008770','#CA1658','#AD8200','#0074DA']
med_colours = ['#008770','#CA1658','#AD8200','#0074DA','#7DF5E0','#FF96BC','#FFCB31','#84C5FF']
fig = plt.figure(constrained_layout=False, figsize = (14,15))
gs = fig.add_gridspec(4,6, height_ratios=[1,1,1,1], width_ratios=[1,1,1,1,1,1], hspace=.11, wspace=0.25)
location_legend = [-1.15,-0.42]
ylims = [[(-0.05,20),(-0.003,1),(-0.003,1),(-0.01,10.4),(-0.01,10.4),(-0.003,1)],
          [(-5,400),(-1.5,200),(-1.5,200),(-0.5,40),(-0.5,40),(-0.01,6)],
          [(0,1700),(0,600),(0,800),(0,300),(-0.01,10),(-0.01,6)]]
decarb_labels = ['Battery Storage Requirement (GW)','Cumulative VRE Curtailment (TWh)','Cumulative $\mathrm{CO_{2}}$ Emissions (Mt$\mathrm{CO_{2}}$)']

# plot capacities
for i in range(len(decarb_cols)):
    for j in range(len(panels)):
        ax = fig.add_subplot(gs[i,j])
        g = sns.boxplot(data=decarb[decarb.load_zone==panels[j]],x='pol',y=decarb_cols[i],hue='fixed_pv', width=0.7,
                        medianprops={"linewidth": .7,'color':'k','label':'_median_','solid_capstyle':'butt'},saturation=1,
                        flierprops={"marker": "o",'markerfacecolor':'none','markeredgecolor':'k','markersize':2},
                        boxprops={"linewidth": .3,'edgecolor':'k'})#,whiskerprops={"linewidth": .7,'color':'k'},capprops={"linewidth": .7,'color':'k'})
        ax.legend([],frameon=False)
        boxes = ax.findobj(matplotlib.patches.PathPatch)
        for color, box in zip(my_colours, boxes):
            box.set_facecolor(color)
        box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
        if len(box_patches) == 0:
            box_patches = ax.artists
        num_patches = len(box_patches)
        if num_patches > 0:
            lines_per_boxplot = len(ax.lines) // num_patches
            for k, patch in enumerate(box_patches):
                col = patch.get_facecolor()
                patch.set_edgecolor(col)
                for line in ax.lines[k * lines_per_boxplot: (k + 1) * lines_per_boxplot]:
                    line.set_color(col)
                    line.set_mfc(col)  # facecolor of fliers
                    line.set_mec(col)  # edgecolor of fliers
            median_lines = [line for line in ax.get_lines() if line.get_label() == '_median_']
            for k, line in enumerate(median_lines):
                line.set_color(med_colours[k])
        if i==0:
            ax.set_title(panels[j],fontsize=14)
            # if j==0:
            #     ax.set_title(panels[j],fontsize=14,fontweight='bold')
        if j==0:
            [x.set_linewidth(1.5) for x in ax.spines.values()]
            ax.set_ylabel(decarb_labels[i],fontsize=9)
        else:
            ax.set_ylabel('')
        if i<2:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(["No $\mathrm{CO_{2}}$ Target"+'\nLimited Coord.',"No $\mathrm{CO_{2}}$ Target"+'\nFull Coord.',
                                  "90% $\mathrm{CO_{2}}$ Cut"+' \nLimited Coord.',"90% $\mathrm{CO_{2}}$ Cut"+'\nFull Coord.'],
                                fontsize=8,rotation=90,linespacing=1)
        ax.set_ylim(ylims[i][j])
        ax.tick_params(axis='y',pad=2)
        ax.set_xlabel('')
pa1 = Patch(facecolor=my_colours[0], edgecolor=med_colours[0],linewidth=0.5)
pa2 = Patch(facecolor=my_colours[1], edgecolor=med_colours[1],linewidth=0.5)
pa3 = Patch(facecolor=my_colours[2], edgecolor=med_colours[2],linewidth=0.5)
pa4 = Patch(facecolor=my_colours[3], edgecolor=med_colours[3],linewidth=0.5)
pb1 = Patch(facecolor=my_colours[4], edgecolor=med_colours[4],linewidth=0.5)
pb2 = Patch(facecolor=my_colours[5], edgecolor=med_colours[5],linewidth=0.5)
pb3 = Patch(facecolor=my_colours[6], edgecolor=med_colours[6],linewidth=0.5)
pb4 = Patch(facecolor=my_colours[7], edgecolor=med_colours[7],linewidth=0.5)
ax.legend(handles=[pa1, pb1, pa2, pb2, pa3, pb3, pa4, pb4],
          labels=['', '', '', '', '', '', '1-Axis Solar', 'Fixed Solar'],
          ncol=4, handletextpad=0.5, handlelength=1.2, columnspacing=-0.5,
          loc='center', fontsize=9,frameon=False,bbox_to_anchor=(-2.6,-0.52))
fig.savefig(figpath + 'figure_S4.png', bbox_inches='tight', dpi=600)
plt.show()

#%% VRE PREFERENCE PARALLEL AXIS MULTI-PANEL (For SI)

plt.rcParams.update(plt.rcParamsDefault)
fig, axs = plt.subplots(2,2,figsize = (11,10))
plt.subplots_adjust(wspace=0.5)
#plt.subplots_adjust(wspace=0.3)
color4 = ['#004D40','#D81B60','#1E88E5','#FFC107']
pols = ['NoCut,Full','NoCut,Lim','90%Cut,Full','90%Cut,Lim']

parallel_coordinates(pref_vre_pt[pref_vre_pt.pol==pols[0]],'pol',linewidth=0,ax=axs[0,1],color=color4[1],alpha=1,axvlines_kwds={'color':'k','alpha':0.5,'linewidth':0.5},zorder=1)
parallel_coordinates(df_f[df_f.pol==pols[0]],'pol',ax=axs[0,1],color=color4[1],alpha=0.75,linestyle='--',axvlines=False,linewidth=1.4,zorder=7)
parallel_coordinates(df_s[df_s.pol==pols[0]],'pol',ax=axs[0,1],color=color4[1],alpha=0.75,linestyle='-',axvlines=False,linewidth=1.4,zorder=6)

parallel_coordinates(pref_vre_pt[pref_vre_pt.pol==pols[1]],'pol',linewidth=0,ax=axs[0,0],color=color4[0],alpha=1,axvlines_kwds={'color':'k','alpha':0.5,'linewidth':0.5},zorder=1)
parallel_coordinates(df_f[df_f.pol==pols[1]],'pol',ax=axs[0,0],color=color4[0],alpha=0.75,linestyle='--',axvlines=False,linewidth=1.4,zorder=7)
parallel_coordinates(df_s[df_s.pol==pols[1]],'pol',ax=axs[0,0],color=color4[0],alpha=0.75,linestyle='-',axvlines=False,linewidth=1.4,zorder=6)

parallel_coordinates(pref_vre_pt[pref_vre_pt.pol==pols[2]],'pol',linewidth=0,ax=axs[1,1],color=color4[2],alpha=1,axvlines_kwds={'color':'k','alpha':0.5,'linewidth':0.5},zorder=1)
parallel_coordinates(df_f[df_f.pol==pols[2]],'pol',ax=axs[1,1],color=color4[2],alpha=0.75,linestyle='--',axvlines=False,linewidth=1.4,zorder=7)
parallel_coordinates(df_s[df_s.pol==pols[2]],'pol',ax=axs[1,1],color=color4[2],alpha=0.75,linestyle='-',axvlines=False,linewidth=1.4,zorder=6)

parallel_coordinates(pref_vre_pt[pref_vre_pt.pol==pols[3]],'pol',linewidth=0,ax=axs[1,0],color=color4[3],alpha=1,axvlines_kwds={'color':'k','alpha':0.5,'linewidth':0.5},zorder=1)
parallel_coordinates(df_f[df_f.pol==pols[3]],'pol',ax=axs[1,0],color=color4[3],alpha=0.75,linestyle='--',axvlines=False,linewidth=1.4,zorder=7)
parallel_coordinates(df_s[df_s.pol==pols[3]],'pol',ax=axs[1,0],color=color4[3],alpha=0.75,linestyle='-',axvlines=False,linewidth=1.4,zorder=6)

lg, dg, ddg = '#E0C9FF', '#B986FF', '#8125FF'
g = sns.boxplot(data=wsdf[wsdf.pol=='NoCut,Lim'].loc[:,'Argentina':'Chile'],ax=axs[0,0],width=0.4,whis=(0,100),color=lg,
                        medianprops={"linewidth": 1,'color':dg,'label':'_median_','solid_capstyle':'butt','zorder':4},saturation=1,
                        boxprops={'zorder':3,"linewidth":1,'edgecolor':lg},whiskerprops={"linewidth":5,'color':lg},capprops={"linewidth": 0})

g = sns.boxplot(data=wsdf[wsdf.pol=='NoCut,Full'].loc[:,'Argentina':'Chile'],ax=axs[0,1],width=0.4,whis=(0,100),color=lg,
                        medianprops={"linewidth": 1,'color':dg,'label':'_median_','solid_capstyle':'butt','zorder':4},saturation=1,
                        boxprops={'zorder':3,"linewidth":1,'edgecolor':lg},whiskerprops={"linewidth":5,'color':lg},capprops={"linewidth": 0})

g = sns.boxplot(data=wsdf[wsdf.pol=='90%Cut,Lim'].loc[:,'Argentina':'Chile'],ax=axs[1,0],width=0.4,whis=(0,100),color=lg,
                        medianprops={"linewidth": 1,'color':dg,'label':'_median_','solid_capstyle':'butt','zorder':4},saturation=1,
                        boxprops={'zorder':3,"linewidth":1,'edgecolor':lg},whiskerprops={"linewidth":5,'color':lg},capprops={"linewidth": 0})

g = sns.boxplot(data=wsdf[wsdf.pol=='90%Cut,Full'].loc[:,'Argentina':'Chile'],ax=axs[1,1],width=0.4,whis=(0,100),color=lg,
                        medianprops={"linewidth": 1,'color':dg,'label':'_median_','solid_capstyle':'butt','zorder':4},saturation=1,
                        boxprops={'zorder':3,"linewidth":1,'edgecolor':lg},whiskerprops={"linewidth":5,'color':lg},capprops={"linewidth": 0})

for ax in [axs[0,0],axs[0,1],axs[1,0],axs[1,1]]:
    plt.sca(ax)
    plt.legend([],bbox_to_anchor=(1,0.97),frameon=False)
    ax.axhline(y=0.5,color='k',linewidth=1,linestyle='--')
    ax.set_ylabel('← Preference for Solar          Preference for Wind →',labelpad=7,fontsize=9)
    vals = ax.get_yticks()
    ax.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in [1,.9,.8,.7,.6,.5,.6,.7,.8,.9,1]])
    ax.set_ylim(-0.003,1.004)
    ax.set_xlim(-0.25,4.25)
    ax.set_xlabel('')

axs[0,1].set_title('No $\mathrm{CO_{2}}$ Target + Full Coordination')
axs[0,0].set_title('No $\mathrm{CO_{2}}$ Target + Limited Coordination')
axs[1,1].set_title('90% $\mathrm{CO_{2}}$ Cut + Full Coordination')
axs[1,0].set_title('90% $\mathrm{CO_{2}}$ Cut + Limited Coordination')

axs[0,0].annotate('a',(0.89,0.94),xycoords='axes fraction',fontsize=12,ha='center',va='center', fontweight='bold')
axs[0,1].annotate('b',(0.89,0.94),xycoords='axes fraction',fontsize=12,ha='center',va='center', fontweight='bold')
axs[1,0].annotate('c',(0.89,0.94),xycoords='axes fraction',fontsize=12,ha='center',va='center', fontweight='bold')
axs[1,1].annotate('d',(0.89,0.94),xycoords='axes fraction',fontsize=12,ha='center',va='center', fontweight='bold')

ax5, ax6, ax7, ax8 = axs[0,0].twinx(), axs[0,1].twinx(), axs[1,0].twinx(), axs[1,1].twinx()
for ax in [ax5,ax6,ax7,ax8]:
    ax.set_yticks([0,.2,.4,.6,.8,1])
    ax.set_yticklabels(['0 GW','75 GW','150 GW','225 GW','300 GW','375 GW'])
    ax.set_ylim(-0.003,1.004)
    ax.set_xlim(-0.25,4.25)
    ax.set_xlabel('')
    ax.tick_params(axis='y', colors=ddg)

lw, location_legend = 2, (1.2,-0.18)
custom_lines = [Line2D([0], [0], color='k', lw=lw, alpha=0.9),
                Line2D([0], [0], color='k', lw=lw-0.15, alpha=0.9, linestyle='--')]
plt.sca(axs[1,0])
l1 = plt.legend(custom_lines, ['Fixed PV','1-axis PV'][::-1],handlelength=3,\
                              loc='center', bbox_to_anchor=location_legend, fontsize=9, frameon=False, ncol=2)
axs[1,0].add_artist(l1)

plt.savefig(figpath + 'figure_S13.png',bbox_inches='tight',dpi=600)
plt.show()

#%% TRANSMISSION DISTANCE MAPS (For SI - NEED GEOPANDAS)

import geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.path import Path
import matplotlib as mpl
countries = geopandas.read_file(datapath + 'map_data/Countries_five.shp')

plt.rcParams.update(plt.rcParamsDefault)
fig = plt.figure(constrained_layout=False)#, figsize = (6,4))
gs = fig.add_gridspec(1,2, width_ratios=[1,1], wspace=-0.06)
axm1 = fig.add_subplot(gs[0])
axm2 = fig.add_subplot(gs[1])
cmin, cmax, wmin, wmax = 0, 300, 0, 300
countries.plot(ax=axm1, color='none',zorder=0,edgecolor='k',linewidth=.15)
cmap = plt.get_cmap('viridis')
w_cmap=truncate_colormap(cmap,(wind_cf['d_trans_km'].min()-wmin)/(wmax-wmin),0.9)
g1 = axm1.scatter(x=wind_cf['Longitude'],y=wind_cf['Latitude'],c=wind_cf['d_trans_km'],marker=path,cmap=w_cmap,s=17,vmin=wmin,vmax=wmax,edgecolors='none')
plt.sca(axm1)
plt.xlim(-77,-30)
plt.ylim(-57,6)
plt.legend('',frameon=False)
countries.plot(ax=axm1, color='none',zorder=10,edgecolor='k',linewidth=.15,linestyle='--',alpha=1)
plt.axis('off')

divider1 = make_axes_locatable(g1.axes)
ax1 = axm1.inset_axes([0.66,0,0.05,0.45],zorder=3)
cbar1 = plt.colorbar(plt.cm.ScalarMappable(cmap=w_cmap,norm=mpl.colors.Normalize(wmin,wmax)),
              cax=ax1)
cbar1.ax.tick_params(labelsize=9)
cbar1.ax.set_yticklabels(['0km','50km','100km','150km','200km','250km','>300km'])
countries.plot(ax=axm2, color='none',zorder=0,edgecolor='k',linewidth=.15)
g2 = axm2.scatter(x=solar_cf['Longitude'],y=solar_cf['Latitude'],c=solar_cf['d_trans_km'],marker=path,cmap='plasma',s=17,vmin=cmin,vmax=solar_cf['d_trans_km'].max(),edgecolors='none')
plt.sca(axm2)
plt.xlim(-77,-30)
plt.ylim(-57,6)
plt.legend('',frameon=False)
countries.plot(ax=axm2, color='none',zorder=10,edgecolor='k',linewidth=.15,linestyle='--',alpha=1)
plt.axis('off')
cmap = plt.get_cmap('plasma')
s_cmap=truncate_colormap(cmap,(solar_cf['d_trans_km'].min()-cmin)/(cmax-cmin),(solar_cf['d_trans_km'].max()-cmin)/(cmax-cmin))
divider = make_axes_locatable(g2.axes)
ax2 = axm2.inset_axes([0.66,0,0.05,0.45],zorder=3)
cbar2 = plt.colorbar(plt.cm.ScalarMappable(cmap=s_cmap,norm=mpl.colors.Normalize(vmin=cmin,vmax=solar_cf['d_trans_km'].max())),cax=ax2)
cbar2.ax.tick_params(labelsize=9)
cbar2.ax.set_yticklabels(['0km','50km','100km','150km','200km','250km'])

axm1.annotate('a',(0,0.94),xycoords='axes fraction',fontsize=12,ha='center',va='center', fontweight='bold')
axm2.annotate('b',(0,0.94),xycoords='axes fraction',fontsize=12,ha='center',va='center', fontweight='bold')

plt.savefig(figpath + 'figure_S14.png',bbox_inches='tight',dpi=1200)
plt.show()

#%% HEATMAP OF BATTERY DISPATCH / CURTAILMENT (For SI)

# hard-coded limits for plotting
monthly_lims, moticks, day_lims, dayticks, colorbar_maxes = [75,75,600], [75,75,600], [75,75,250], [75,75,250], [7,7,50]
hr, wr, ws, hs = [1,4.5], [8,1], 0.032, 0.045
labs = ['Battery Discharge (MW)','Battery Discharge (MW)','Battery Charge (MW)','Battery Charge (MW)','Curtailment (MW)','Curtailment (MW)']

plt.rcParams.update(plt.rcParamsDefault)
#plt.style.use('seaborn-white')
fig = plt.figure(constrained_layout=False, figsize = (12,9))
gs = fig.add_gridspec(3,4, width_ratios=[1,0.05,1,0.04], height_ratios=[1,1,1], wspace = .13, hspace= .29)
gs00 = gs[0,0].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gs01 = gs[0,2].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gs10 = gs[1,0].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gs11 = gs[1,2].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gs20 = gs[2,0].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gs21 = gs[2,2].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gslist = [gs00,gs01,gs10,gs11,gs20,gs21]
ax_cb1,ax_cb2,ax_cb3=fig.add_subplot(gs[0,3]),fig.add_subplot(gs[1,3]),fig.add_subplot(gs[2,3])
cb_list = [ax_cb1,ax_cb2,ax_cb3]
panel_labels = ['a','b','c','d','e','f']
for y in range(6):
    df = df_list[y]
    hourly_values = df.sum(axis=0).reset_index().rename(columns={'Hour':'Hour',0:'Sum'})
    monthly_values = df.sum(axis=1).reset_index().rename(columns={'Month':'Month',0:'Sum'})
    ax_hm = fig.add_subplot(gslist[y][1,0])   #axes for heatmap
    ax_hr = fig.add_subplot(gslist[y][0,0])   #axes for hour bars
    ax_mo = fig.add_subplot(gslist[y][1,1])   #axes for month bars
    ax_lg = fig.add_subplot(gslist[y][0,1])
    ax_lg.axis('off')
    heatmap = sns.heatmap(df, ax=ax_hm, cbar_ax=cb_list[y//2], annot=False, vmin=0, vmax=colorbar_maxes[y//2], fmt='.0f',\
                          xticklabels=np.arange(1,25),yticklabels=np.arange(1,13))
    hour_bars = sns.barplot(x='Hour', y='Sum',data=hourly_values, ax=ax_hr, color='#88CCEE', edgecolor='k', orient='v')
    month_bars = sns.barplot(x='Sum', y='Month',data=monthly_values, ax=ax_mo, color='#88CCEE', edgecolor='k', orient='h')
    
    heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize = 8)
    heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 8)
    heatmap.set_xlabel('Hour of Day', fontsize=10, labelpad=6)
    heatmap.set_ylabel('Month', fontsize=10, labelpad=3)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    t = cbar.get_ticks().tolist()
    if y<5:
        t = [0,1,2,3,4,5,6,7]
    cbar.ax.set_yticks(t,labels=["{:.0f} MW".format(i) for i in t])
    ax_hr.get_xaxis().set_visible(False)
    ax_hr.set_ylim(0,day_lims[y//2])
    ax_hr.set_yticks([dayticks[y//2]])
    ax_hr.set_yticklabels([str(dayticks[y//2])], fontsize=8)
    ax_hr.set_ylabel('Sum', fontsize=8, labelpad=2)
    ax_mo.get_yaxis().set_visible(False)
    ax_mo.set_xlim(0,monthly_lims[y//2])
    ax_mo.set_xticks([moticks[y//2]])
    ax_mo.set_xticklabels([str(moticks[y//2])], fontsize=8)
    ax_mo.set_xlabel('Sum', fontsize=8, labelpad=2)
    ax_hr.annotate(str(labs[y]),(.02,.74),xycoords='axes fraction',annotation_clip=False,
                        ha='left',va='center',fontsize=8)
    ax_lg.annotate(panel_labels[y],(.5,.5),xycoords='axes fraction',annotation_clip=False,
                        fontweight='bold',ha='center',va='center',fontsize=14)
    if y==0:
        ax_hr.set_title('           Average of Limited Coordination Scenarios (2050)',fontsize=12,y=1.14)
    elif y==1:
        ax_hr.set_title('    Average of Full Coordination Scenarios (2050)',fontsize=12,y=1.14)

plt.savefig(figpath + 'figure_S5.png',facecolor='w',bbox_inches='tight',dpi=600)
plt.show()

#%% HOURLY DISPATCH PLOTS (For SI)

plt.rcParams.update(plt.rcParamsDefault)
font = FontProperties()
font.set_name('Open Sans')
plt.rcParams["font.family"] = "Arial"
nrow,ncol=4,3
yrs = ['2035','2050']
fignum = {'2035':'S6','2050':'S7'}
for yr in yrs:
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(ncol*3.4,nrow*2.2), facecolor='white')
    fig.subplots_adjust(hspace=0.09,wspace=0.17)
    dfl = [[np.nan for i in range(ncol)] for j in range(nrow)]
    year,months,hours = yr,['04','08','12'],['01','24']
    mo_labs = ['April','August','December']
    grp_labs = ["90% $\mathrm{CO_{2}}$ Cut" + '\nLimited Coordination\n1-axis Solar',
                "90% $\mathrm{CO_{2}}$ Cut" + '\nLimited Coordination\nFixed Solar',
                "90% $\mathrm{CO_{2}}$ Cut" + '\nFull Coordination\n1-axis Solar',
                "90% $\mathrm{CO_{2}}$ Cut" + '\nFull Coordination\nFixed Solar']
    clr1 = [coal_color, gas_color, diesel_color, nuclear_color, geothermal_color, biomass_color,
            hydro_color, solar_color, wind_color, battery_color, curtailment_color]
    clr2 = [battery_color]
    
    for i in range(nrow):
        for j in range(ncol):
            start_time, end_time = int(yr+months[j]+hours[0]), int(yr+months[j]+hours[1])
            dfl[i][j] = mean_disp[(mean_disp.group==i+1) & (mean_disp.index>=start_time) & (mean_disp.index<=end_time)].copy()
    
            # plot stackplot
            axs[i,j].stackplot(dfl[i][j].index,dfl[i][j].Coal,dfl[i][j].Gas,dfl[i][j].Diesel,dfl[i][j].Nuclear,
                                dfl[i][j].Geothermal,dfl[i][j].Biomass,dfl[i][j].Hydro,dfl[i][j].Solar,dfl[i][j].Wind,
                                dfl[i][j].Battery_discharge,dfl[i][j].curt,labels=['Coal','Gas','Diesel','Nuclear',
                                'Geothermal','Biomass','Hydro', 'Solar', 'Wind', 'Battery','Curtailment'],colors=clr1,alpha=1.0)
            axs[i,j].stackplot(dfl[i][j].index, dfl[i][j].Battery_charge, colors=clr2, alpha=1.0)
            axs[i,j].plot(dfl[i][j].index, dfl[i][j].load, '--k', lw=1, alpha=1.0, label='Load')
    
            axs[i,j].spines['top'].set_visible(False)
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['bottom'].set_visible(False)
            axs[i,j].spines['left'].set_visible(False)
            axs[i,j].tick_params(axis='both', which='major', labelsize=9)
    
            axs[i,j].set_ylim(-20,380)
            axs[i,j].set_xlim(start_time,end_time)
            axs[i,j].set_yticks([0,100,200,300])
    
            if i < nrow-1:
                axs[i,j].set_xticks([])
            else:
                axs[i,j].set_xticks([start_time,start_time+5,start_time+11,start_time+17,start_time+23])
                axs[i,j].set_xticklabels(['0h', '6h', '12h', '18h', '24h'])
            if j == 0:
                axs[i,j].set_ylabel('Power (GW)', fontsize=13)
            if i == 0:
                axs[i,j].text(0.5, 1.1, mo_labs[j], size=13, rotation=0., ha='center', va='center', transform=axs[i,j].transAxes)
            if j == ncol-1:
                axs[i,j].text(1.15,0.5,grp_labs[i],size=10,rotation=90,ha='center',va='center',transform=axs[i,j].transAxes)
    
    handles, labels = axs[0,0].get_legend_handles_labels()
    axs[nrow-1,(ncol//2)-3].legend(handles[1:]+handles[0:1],labels[1:]+labels[0:1],loc='center',bbox_to_anchor=[0.2,-0.4],ncol=6,frameon=False)
    axs[nrow-1,(ncol//2)-1].annotate('Year: '+yr,xy=(2.97,-0.44),xycoords='axes fraction',annotation_clip=False,fontsize=16)
    
    fig.savefig(figpath + 'figure_{}.png'.format(fignum[yr]),bbox_inches='tight',dpi=600)
    plt.show()


#%% HEATMAP OF HOURLY TRADE (BY LINE - For SI)

# HEATMAP OF HOURLY TRADE (BY LINE) (90% emissions cut + full coordination only)
tp_trade = pd.read_parquet(datapath + 'total_trade_by_timepoint_net.parquet')
tp50 = tp_trade[tp_trade.period==2050].groupby(['timepoint','tx_line','scenario']).sum().reset_index().drop(columns='period')
df_tn = tp50[tp50.scenario.str.contains('cut90p_tn')][['timepoint','tx_line','scenario','net_flow_twh']]
df_tn.timepoint = df_tn.timepoint.astype(str)
df_tn['Year'] = df_tn['timepoint'].str[:4].astype(int)
df_tn['Month'] = df_tn['timepoint'].str[4:6].astype(int)
df_tn['Hour'] = df_tn['timepoint'].str[6:].astype(int)
df_tn_avg = df_tn.drop(columns=['scenario','timepoint']).groupby(['tx_line','Year','Month','Hour']).mean().reset_index()
df_tn_avg_list = []
lines = df_tn_avg.tx_line.unique().tolist()
line_labs = ['Argentina → Brazil','Argentina → Chile','Argentina → Paraguay','Argentina → Uruguay','Paraguay → Brazil','Uruguay → Brazil']
bar_labs = ['← To Argentina      To Brazil →      ','← To Argentina      To Chile →        ','← To Argentina     To Paraguay →',
            '← To Argentina     To Uruguay →  ','← To Paraguay      To Brazil →      ','← To Uruguay      To Brazil →    ']
for i in range(len(lines)):
    df_tn_avg_pt = df_tn_avg[df_tn_avg.tx_line==lines[i]].pivot(columns='Hour',index='Month',values='net_flow_twh')*(10**3) #units now in GW
    if i > 3:
        df_tn_avg_pt *=-1
    df_tn_avg_list.append(df_tn_avg_pt)

for i in range(len(lines)):
    df_tn_avg_list[i].to_csv('figS10_{}.csv'.format(panel_labels[i]), index=False)

# hard-coded limits for plotting
monthly_lims,moticks = [(-225,225),(0,275),(-1.5,1.5),(0,12),(0,150),(-15,15)], [[0,225],[0,275],[0,1.5],[0,12],[0,150],[0,15]]
day_lims, dayticks = [(0,30),(-75,200),(0,0.5),(0,3),(0,80),(-1,4)], [[0,30],[0,200],[0,0.5],[0,3],[0,80],[0,4]]
cb_mins, colorbar_maxes = [-7.5,-15,-0.06,-.5,-6,-0.5], [7.5,15,0.06,.5,6,0.5]
hr, wr, ws, hs = [1,4.5], [8,1], 0.045, 0.065

plt.rcParams.update(plt.rcParamsDefault)
font = FontProperties()
font.set_name('Open Sans')
plt.rcParams["font.family"] = "Arial"
fig = plt.figure(constrained_layout=False, figsize = (12.5,9))
gs = fig.add_gridspec(3,6, width_ratios=[1,0.04,0.11,0.11,1,0.04], height_ratios=[1,1,1], wspace = .13, hspace= .31)
gs00 = gs[0,0].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gs01 = gs[0,4].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gs10 = gs[1,0].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gs11 = gs[1,4].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gs20 = gs[2,0].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gs21 = gs[2,4].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gslist = [gs00,gs01,gs10,gs11,gs20,gs21]
ax_cb1,ax_cb2,ax_cb3,ax_cb4,ax_cb5,ax_cb6=fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,5]),fig.add_subplot(gs[1,1]),\
                                            fig.add_subplot(gs[1,5]),fig.add_subplot(gs[2,1]),fig.add_subplot(gs[2,5])
cb_list = [ax_cb1,ax_cb2,ax_cb3,ax_cb4,ax_cb5,ax_cb6]
panel_labels = ['a','b','c','d','e','f']
cbar_formats = ["{:.1f} GW","{:.0f} GW","{:.2f} GW","{:.2f} GW","{:.0f} GW","{:.2f} GW"]
ts = [[-7.5,-5,-2.5,0,2.5,5,7.5],[-15,-10,-5,0,5,10,15],[-.06,-.03,0,.03,.06],[-.5,-.25,0,.25,.5],[-6,-3,0,3,6],[-.5,-.25,0,.25,.5]]
for y in range(6):
    df = df_tn_avg_list[y]
    hourly_values = df.sum(axis=0).reset_index().rename(columns={'Hour':'Hour',0:'Sum'})
    monthly_values = df.sum(axis=1).reset_index().rename(columns={'Month':'Month',0:'Sum'})
    ax_hm = fig.add_subplot(gslist[y][1,0])   #axes for heatmap
    ax_hr = fig.add_subplot(gslist[y][0,0])   #axes for hour bars
    ax_mo = fig.add_subplot(gslist[y][1,1])   #axes for month bars
    ax_lg = fig.add_subplot(gslist[y][0,1])
    ax_lg.axis('off')
    heatmap = sns.heatmap(df, ax=ax_hm, cbar_ax=cb_list[y], annot=False, fmt='.0f', vmin=cb_mins[y], vmax=colorbar_maxes[y],\
                          xticklabels=np.arange(1,25),yticklabels=np.arange(1,13),cmap='coolwarm')
    hour_bars = sns.barplot(x='Hour', y='Sum',data=hourly_values, ax=ax_hr, color='#1E88E5', edgecolor='k', orient='v')
    month_bars = sns.barplot(x='Sum', y='Month',data=monthly_values, ax=ax_mo, color='#1E88E5', edgecolor='k', orient='h')
    
    heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize = 8)
    heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 8)
    heatmap.set_xlabel('Hour of Day', fontsize=9, labelpad=4)
    heatmap.set_ylabel('Month', fontsize=9, labelpad=1)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_yticks(ts[y],labels=[cbar_formats[y].format(i) for i in ts[y]],fontsize=9)
    cbar.ax.set_ylabel(bar_labs[y], fontsize=9, labelpad=6)
    ax_hr.get_xaxis().set_visible(False)
    ax_hr.set_ylim(day_lims[y])
    ax_hr.set_yticks(dayticks[y])
    ax_hr.set_yticklabels([str(l) for l in dayticks[y]], fontsize=7)
    ax_hr.set_ylabel('Sum', fontsize=8, labelpad=2)
    ax_mo.get_yaxis().set_visible(False)
    ax_mo.set_xlim(monthly_lims[y])
    ax_mo.set_xticks(moticks[y])
    ax_mo.set_xticklabels([str(l) for l in moticks[y]], fontsize=7)
    ax_mo.set_xlabel('Sum', fontsize=8, labelpad=2)
    ax_mo.tick_params(axis='x', pad=2)
    ax_hr.tick_params(axis='y', pad=2)
    ax_hr.annotate(str(line_labs[y]),(.02,1.25),xycoords='axes fraction',annotation_clip=False,
                        ha='left',va='center',fontsize=8)
    ax_lg.annotate(panel_labels[y],(.5,.5),xycoords='axes fraction',annotation_clip=False,
                        fontweight='bold',ha='center',va='center',fontsize=14)
plt.savefig(figpath + 'figure_S10.png',facecolor='w',bbox_inches='tight',dpi=600)
plt.show()

#%% HEATMAP OF HOURLY TRADE SYSTEM-WIDE FOR PAIRED REF. SCENARIOS (For SI)

#(THREE-PANEL INCLUDING DIFFERENCE BETWEEN FIRST TWO) 2/18/25

tp_trade = pd.read_parquet(datapath + 'total_trade_by_timepoint.parquet')
tp50 = tp_trade[tp_trade.period==2050].groupby(['timepoint','scenario']).sum().reset_index().drop(columns='period')
ts1 = tp50[tp50.scenario=='prm15_wcost_re11_c2p1_t1'].copy(deep=True)
ts2 = tp50[tp50.scenario=='prm15_wcost_re11_c2p1_tn'].copy(deep=True)
ts3 = tp50[tp50.scenario=='prm15_wcost_re11_c2p1_co2_cut90p_t1'].copy(deep=True)
ts4 = tp50[tp50.scenario=='prm15_wcost_re11_c2p1_co2_cut90p_tn'].copy(deep=True)
tsref = ts1.merge(ts2,on='timepoint', suffixes=['_t1','_tn'])
tscut = ts3.merge(ts4,on='timepoint', suffixes=['_t1','_tn'])
tsref['diff'] = tsref['net_flow_twh_tn'] - tsref['net_flow_twh_t1']
tscut['diff'] = tscut['net_flow_twh_tn'] - tscut['net_flow_twh_t1']
tsref = tsref[['timepoint','diff']]
tscut = tscut[['timepoint','diff']]
tsref.timepoint = tsref.timepoint.astype(str)
tsref['Year'] = tsref['timepoint'].str[:4].astype(int)
tsref['Month'] = tsref['timepoint'].str[4:6].astype(int)
tsref['Hour'] = tsref['timepoint'].str[6:].astype(int)
tscut.timepoint = tscut.timepoint.astype(str)
tscut['Year'] = tscut['timepoint'].str[:4].astype(int)
tscut['Month'] = tscut['timepoint'].str[4:6].astype(int)
tscut['Hour'] = tscut['timepoint'].str[6:].astype(int)
tsref_pt = tsref.pivot(index='Month',columns='Hour',values='diff')*1000
tscut_pt = tscut.pivot(index='Month',columns='Hour',values='diff')*1000

ts3 = tscut_pt-tsref_pt
ts3df = [tsref_pt,tscut_pt,ts3]
hr, wr, ws, hs = [1,4.5], [8,1], 0.032, 0.045
monthly_lims,moticks = [(0,175),(0,700),(0,700)], [[0,175],[0,700],[0,700]]
day_lims, dayticks = [(0,100),(0,400),(0,400)], [[0,100],[0,400],[0,400]]
cb_mins, colorbar_maxes = [0,0,-5], [7.5,30,30]

cmap = plt.get_cmap('seismic')
new_cmap = truncate_colormap(cmap, .417, 1)

plt.rcParams.update(plt.rcParamsDefault)
font = FontProperties()
font.set_name('Open Sans')
plt.rcParams["font.family"] = "Arial"
fig = plt.figure(constrained_layout=False, figsize = (6,9.75))
gs = fig.add_gridspec(3,2, height_ratios=[1,1,1], width_ratios=[1,0.035], hspace=.24, wspace=0.08)
gs0 = gs[0,0].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gs1 = gs[1,0].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gs2 = gs[2,0].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
ax_cb1, ax_cb2, ax_cb3 = fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,1]), fig.add_subplot(gs[2,1])
gslist, cblist = [gs0,gs1,gs2], [ax_cb1,ax_cb2,ax_cb3]
panel_labels = ['a','b','c']
cbar_formats = ["{:.1f} GW","{:.1f} GW","{:.1f} GW"]
cbl = ['Increase in Net Electricity Trade\nReference (No $\mathrm{CO_{2}}$ Target)',
        'Increase in Net Electricity Trade\nMitigation (90% $\mathrm{CO_{2}}$ Cut)',
        'Difference between (b) and (a)']
ts, lp = [[0,2.5,5,0,7.5],[0,7.5,15,22.5,30],[-5,0,10,20,30]], [15.5,11,11]
clmp, barc, barec = ['RdPu','YlGnBu',new_cmap], ['#EC417F','#429DED','#000000'], ['#ff84b1','#0074DA','#000000']

for y in range(3):
    df = ts3df[y].copy(deep=True)
    hourly_values = df.sum(axis=0).reset_index().rename(columns={'Hour':'Hour',0:'Sum'})
    monthly_values = df.sum(axis=1).reset_index().rename(columns={'Month':'Month',0:'Sum'})
    ax_hm = fig.add_subplot(gslist[y][1,0])   #axes for heatmap
    ax_hr = fig.add_subplot(gslist[y][0,0])   #axes for hour bars
    ax_mo = fig.add_subplot(gslist[y][1,1])   #axes for month bars
    ax_lg = fig.add_subplot(gslist[y][0,1])
    ax_lg.axis('off')

    heatmap = sns.heatmap(df, ax=ax_hm, cbar_ax=cblist[y], annot=False, fmt='.0f', vmin=cb_mins[y], vmax=colorbar_maxes[y],\
                          xticklabels=np.arange(1,25),yticklabels=np.arange(1,13),cmap=clmp[y])
    hour_bars = sns.barplot(x='Hour', y='Sum',data=hourly_values, ax=ax_hr, color=barc[y], edgecolor=barec[y], orient='v',alpha=0.7)
    month_bars = sns.barplot(x='Sum', y='Month',data=monthly_values, ax=ax_mo, color=barc[y], edgecolor=barec[y], orient='h',alpha=0.7)
    
    heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=8)
    heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=8)
    heatmap.set_xlabel('Hour of Day', fontsize=9, labelpad=5)
    heatmap.set_ylabel('Month', fontsize=9, labelpad=1)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_yticks(ts[y],labels=[cbar_formats[y].format(i) for i in ts[y]],fontsize=9)
    cbar.ax.set_ylabel(cbl[y],fontsize=10,labelpad=lp[y])
    ax_hr.get_xaxis().set_visible(False)
    ax_hr.set_ylim(day_lims[y])
    ax_hr.set_yticks(dayticks[y])
    ax_hr.set_yticklabels([str(l) for l in dayticks[y]], fontsize=7)
    ax_hr.set_ylabel('Sum', fontsize=8, labelpad=2)
    ax_mo.get_yaxis().set_visible(False)
    ax_mo.set_xlim(monthly_lims[y])
    ax_mo.set_xticks(moticks[y])
    ax_mo.set_xticklabels([str(l) for l in moticks[y]], fontsize=7)
    ax_mo.set_xlabel('Sum', fontsize=8, labelpad=2)
    ax_mo.tick_params(axis='x', pad=2)
    ax_hr.tick_params(axis='y', pad=2)
    ax_lg.annotate(panel_labels[y],xy=(0.5,0.5), xycoords='axes fraction', fontsize=14, fontweight='bold',ha='center',va='center')

plt.savefig(figpath + 'figure_S9.png',facecolor='w',bbox_inches='tight',dpi=800)
plt.show()

#%%