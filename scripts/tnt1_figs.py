#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 07:10:08 2024

@author: jake
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
import mpltern

### GENERAL INFORMATION / VARIABLES ###
main_path = '../' # SET main_path TO NAVIGATE TO gplac/results/ DIRECTORY
respath = main_path + 'outputs/fix_tnt1_outputs/'
figpath = '../paper2_figures/paper_figures/'
datapath = 'figure_data/'
sc_names = pd.read_csv('processed_data/scenario_names.csv')
mercosur = ['Argentina','Brazil','Chile','Paraguay','Uruguay']
periods = [2020,2025,2030,2035,2040,2045,2050]

paired_colors = [[['#7DF5E0','#004D40'],['#FF669E','#D81B60']],[['#FFC107','#AD8200'],['#63B6FF','#1E88E5']]] # [[[green],[pink]],[[yellow],[blue]]]
my_colours = ['#7DF5E0','#008770','#FF96BC','#CA1658','#FFCB31','#AD8200','#84C5FF','#0074DA'] # [green,pink,yellow,blue]
med_colours = ['#008770','#7DF5E0','#CA1658','#FF96BC','#AD8200','#FFCB31','#0074DA','#84C5FF']
palette = {'000':'#FF669E','001':'#D81B60','010':'#10EFC9','011':'#004D40','100':'#63B6FF','101':'#0074DA','110':'#FFC107','111':'#AD8200'}
cap_tech = ['Wind','Solar','Hydro','Fossil']
panels = ['Region-wide'] + mercosur

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap=mcolors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'\
             .format(n=cmap.name,a=minval,b=maxval),cmap(np.linspace(minval,maxval,n)))
    return new_cmap
arr = np.linspace(0, 50, 100).reshape((10, 10))
fig, ax = plt.subplots(ncols=2)
cmap = plt.get_cmap('binary')
new_cmap = truncate_colormap(cmap, 0.1, .45)
ax[0].imshow(arr, interpolation='nearest', cmap=cmap)
ax[1].imshow(arr, interpolation='nearest', cmap=new_cmap)
plt.show()
matplotlib.cm.register_cmap("new_cmap", new_cmap)
cpal = sns.color_palette("new_cmap", n_colors=1000, desat=1)

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



### data files
cost_df = pd.read_csv(respath + 'costs_emissions_tnt1.csv')
hourly_disp = pd.read_csv(respath + 'hourly_dispatch_tnt1.csv')
hydcurt = pd.read_csv(respath + 'scheduled_hydro_curtailment_tnt1.csv')
vrecurt = pd.read_csv(respath + 'vre_curtailment_tnt1.csv')




cost_df = cost_df[['scs_id','period','load_zone','variable_om_cost_total','fuel_cost_total','startup_cost_total',
                'capacity_cost','transmission_capacity_cost','mwh','MtCO2','total_costs']].groupby('scs_id').sum().reset_index()
cost_df['total_costs'] /= 1000000000
cost_df['capacity_cost'] /= 1000000000
cost_df['variable_om_cost_total'] /= 1000000000
cost_df['fuel_cost_total'] /= 1000000000
cost_df['startup_cost_total'] /= 1000000000
cost_df['emis_policy'] = cost_df.scs_id.str.contains('cut90p')*1
cost_df['exist_trd'] = cost_df.scs_id.str.contains('t1')*1
cost_df['op_char'] = cost_df.scs_id.str.split('_', expand=True)[3].str[2:].astype(int)
cost_df['fixed_pv'] = (cost_df.op_char > 10)*1
cost_df['pol'] = cost_df.emis_policy.astype(str) + cost_df.exist_trd.astype(str)
cost_df['group'] = cost_df.emis_policy.astype(str) + cost_df.exist_trd.astype(str) + cost_df.fixed_pv.astype(str)
cost_df.pol.replace({'00':'NoCut,Full','01':'NoCut,Lim','10':'90%Cut,Full','11':'90%Cut,Lim'},inplace=True)
difforder=CategoricalDtype(['NoCut,Lim','NoCut,Full','90%Cut,Lim','90%Cut,Full'], ordered=True)
cost_df['pol'] = cost_df['pol'].astype(difforder)


split_tx_cost_df = pd.read_csv('figure_data/split_tx_cost_df.csv')
split_tx_cost_df = split_tx_cost_df[(split_tx_cost_df.load_zone=='Region-wide')&(split_tx_cost_df.exist_trd==0)]
split_tx_cost_df['total_costs'] /= 1000000000
split_tx_cost_df['variable_om_cost_total'] /= 1000000000
split_tx_cost_df['fuel_cost_total'] /= 1000000000
split_tx_cost_df['startup_cost_total'] /= 1000000000

fig, ax = plt.subplots(1,2, figsize= (7,4))
plt.subplots_adjust(wspace=0.3)
g = sns.boxplot(data=cost_df,x='emis_policy',y='total_costs',hue='fixed_pv', width=0.6, ax=ax[0],
                medianprops={"linewidth": 1,'color':'k','label':'_median_','solid_capstyle':'butt'},saturation=1,
                flierprops={"marker": "o",'markerfacecolor':'none','markeredgecolor':'k','markersize':2},
                boxprops={"linewidth": .6,'edgecolor':'k','facecolor':'white'})

g2 = sns.boxplot(data=split_tx_cost_df,x='emis_policy',y='total_costs',hue='fixed_pv', width=0.6, ax=ax[1],
                medianprops={"linewidth": 1,'color':'k','label':'_median_','solid_capstyle':'butt'},saturation=1,
                flierprops={"marker": "o",'markerfacecolor':'none','markeredgecolor':'k','markersize':2},
                boxprops={"linewidth": .6,'edgecolor':'k','facecolor':'white'})
ax[0].set_ylim(280,350)
ax[1].set_ylim(280,350)
#plt.show()
plt.savefig('total_costs.png',dpi=300, bbox_inches='tight')









df_tn = hourly_disp[['scs','timepoint','load_zone','Battery_discharge','Battery_charge','curt']].copy(deep=True)
df_tn.timepoint = df_tn.timepoint.astype(str)
df_tn['Year'] = df_tn['timepoint'].str[:4].astype(int)
df_tn['Month'] = df_tn['timepoint'].str[4:6].astype(int)
df_tn['Hour'] = df_tn['timepoint'].str[6:].astype(int)

df_tn_sw_2050 = df_tn[df_tn.Year==2050].groupby(['scs','timepoint','Year','Month','Hour']).sum().reset_index()
df_tn_sw_2050 = df_tn_sw_2050.drop(columns=['scs','load_zone']).groupby(['Year','Month','Hour']).mean().reset_index()

df_tn_sw_2050_dis_pt = df_tn_sw_2050.pivot(index='Month',columns='Hour',values='Battery_discharge')
df_tn_sw_2050_ch_pt = df_tn_sw_2050.pivot(index='Month',columns='Hour',values='Battery_charge')
df_tn_sw_2050_ch_pt *= -1
df_tn_sw_2050_curt_pt = df_tn_sw_2050.pivot(index='Month',columns='Hour',values='curt')

df_list = [df_tn_sw_2050_dis_pt,df_tn_sw_2050_ch_pt,df_tn_sw_2050_curt_pt]

df_dispatch_sw = hourly_disp.groupby(['timepoint','scs']).sum().reset_index().set_index('timepoint')
### averages for scenario groupings across all turbines: cut90p.t1.1axis, cut90p.t1.fixed, cut90p.tn.1axis, cut90p.tn.fixedyear,months,hours = '2050',['01','02','03','04','05','06','07','08','09','10','11','12'],['01','24']
df_dispatch_sw = hourly_disp.groupby(['timepoint','scs']).sum().reset_index()
df_dispatch_sw['op_char'] = df_dispatch_sw.scs.str.split('_', expand=True)[3].str[2:].astype(int)
df_dispatch_sw['month'] = df_dispatch_sw.timepoint.astype(str).str[4:6].astype(int)
group1 = df_dispatch_sw[(df_dispatch_sw.op_char<=10) & (df_dispatch_sw.month.isin([4,8,12]))]
group2 = df_dispatch_sw[(df_dispatch_sw.op_char>10) & (df_dispatch_sw.month.isin([4,8,12]))]
group1 = group1.drop(columns=['op_char','scs','load_zone']).groupby(['timepoint','month']).mean().reset_index()
group1['group'] = 1
group2 = group2.drop(columns=['op_char','scs','load_zone']).groupby(['timepoint','month']).mean().reset_index()
group2['group'] = 2
mean_disp = pd.concat([group1,group2])
mean_disp = mean_disp.set_index('timepoint')


#%% HOURLY DISPATCH PLOTS (SI)

nrow,ncol=2,3
yrs = ['2035','2050']
for yr in yrs:
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(ncol*3.4,nrow*2.2), facecolor='white')
    fig.subplots_adjust(hspace=0.09,wspace=0.17)
    dfl = [[np.nan for i in range(ncol)] for j in range(nrow)]
    year,months,hours = yr,['04','08','12'],['01','24']
    mo_labs = ['April','August','December']
    grp_labs = [r"$90\%\ CO_{2}\ Cut$" + '\n' + r'$\bf{Limited}$' + r'$\ Coordination$' + '\n' + r'$1-axis\ Solar$',
                r"$90\%\ CO_{2}\ Cut$" + '\n' + r'$\bf{Limited}$' + r'$\ Coordination$' + '\n' + r'$Fixed\ Solar$',
                r"$90\%\ CO_{2}\ Cut$" + '\n' + r'$\bf{Full}$' + r'$\ Coordination$' + '\n' + r'$1-axis\ Solar$',
                r"$90\%\ CO_{2}\ Cut$" + '\n' + r'$\bf{Full}$' + r'$\ Coordination$' + '\n' + r'$Fixed\ Solar$',]
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
    axs[nrow-1,(ncol//2)-1].annotate('Year: '+yr,xy=(2.97,-0.44),xycoords='axes fraction',annotation_clip=False,fontsize=16,fontweight='bold')
    
    plt.show()
    #fig.savefig(figpath + 'hourly_mean_dispatch_tnt1_{}.png'.format(yr),bbox_inches='tight',dpi=600)






#%%

# hard-coded limits for plotting
monthly_lims, moticks, day_lims, dayticks, colorbar_maxes = [20,20,600], [20,20,600], [20,20,250], [20,20,250], [2,2,50]
hr, wr, ws, hs = [1,4.5], [8,1], 0.032, 0.045
labs = ['Battery Discharge (MW)','Battery Charge (MW)','Curtailment (MW)']

plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-v0_8-white')
fig = plt.figure(constrained_layout=False, figsize = (7,9))
gs = fig.add_gridspec(3,2, width_ratios=[1,0.05], height_ratios=[1,1,1], wspace = .13, hspace= .35)
gs00 = gs[0,0].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gs10 = gs[1,0].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gs20 = gs[2,0].subgridspec(2,2, height_ratios=hr, width_ratios=wr, wspace=ws,hspace=hs)
gslist = [gs00,gs10,gs20]
ax_cb1,ax_cb2,ax_cb3=fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[2,1])
cb_list = [ax_cb1,ax_cb2,ax_cb3]
panel_labels = ['(a)','(b)','(c)','(d)','(e)','(f)']
for y in range(3):
    df = df_list[y]
    hourly_values = df.sum(axis=0).reset_index().rename(columns={'Hour':'Hour',0:'Sum'})
    monthly_values = df.sum(axis=1).reset_index().rename(columns={'Month':'Month',0:'Sum'})
    ax_hm = fig.add_subplot(gslist[y][1,0])   #axes for heatmap
    ax_hr = fig.add_subplot(gslist[y][0,0])   #axes for hour bars
    ax_mo = fig.add_subplot(gslist[y][1,1])   #axes for month bars
    ax_lg = fig.add_subplot(gslist[y][0,1])
    ax_lg.axis('off')
    heatmap = sns.heatmap(df, ax=ax_hm, cbar_ax=cb_list[y], annot=False, vmin=0, vmax=colorbar_maxes[y], fmt='.0f',\
                          xticklabels=np.arange(1,25),yticklabels=np.arange(1,13))
    hour_bars = sns.barplot(x='Hour', y='Sum',data=hourly_values, ax=ax_hr, color='#88CCEE', edgecolor='k', orient='v')
    month_bars = sns.barplot(x='Sum', y='Month',data=monthly_values, ax=ax_mo, color='#88CCEE', edgecolor='k', orient='h')
    
    heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize = 8, fontweight='bold')
    heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 8, fontweight='bold')
    heatmap.set_xlabel('Hour of Day', fontsize=10, fontweight='bold',labelpad=6)
    heatmap.set_ylabel('Month', fontsize=10, fontweight='bold',labelpad=3)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    t = cbar.get_ticks().tolist()
    if y<2:
        cbar.ax.set_yticks(t,labels=["{:.1f} MW".format(i) for i in t],fontweight='bold')
    else:
        cbar.ax.set_yticks(t,labels=["{:.0f} MW".format(i) for i in t],fontweight='bold')
    ax_hr.get_xaxis().set_visible(False)
    ax_hr.set_ylim(0,day_lims[y])
    ax_hr.set_yticks([dayticks[y]])
    ax_hr.set_yticklabels([str(dayticks[y])], fontsize=8, fontweight='bold')
    ax_hr.set_ylabel('Sum', fontsize=8, fontweight='bold',labelpad=2)
    ax_mo.get_yaxis().set_visible(False)
    ax_mo.set_xlim(0,monthly_lims[y])
    ax_mo.set_xticks([moticks[y]])
    ax_mo.set_xticklabels([str(moticks[y])], fontsize=8, fontweight='bold')
    ax_mo.set_xlabel('Sum', fontsize=8, fontweight='bold',labelpad=2)
    ax_hr.annotate(str(labs[y]),(.02,.74),xycoords='axes fraction',annotation_clip=False,
                        fontweight='bold',ha='left',va='center',fontsize=8)
    ax_lg.annotate(panel_labels[y],(.5,.5),xycoords='axes fraction',annotation_clip=False,
                        fontweight='bold',ha='center',va='center',fontsize=16)
    if y==0:
        ax_hr.set_title('          Average of Fixed Coordination Scenarios (2050)',fontweight='bold',fontsize=12,y=1.14)

plt.show()
#plt.savefig(figpath + 'heatmap_battery_curt_tnt1.png',facecolor='w',bbox_inches='tight',dpi=600)





f = hydcurt.drop(columns=['technology','load_zone']).groupby(['scs','period']).sum().reset_index()
fig, ax = plt.subplots()
sns.lineplot(f,x='period',y='twh', units='scs', estimator=None)
plt.title('Hydro curtailment - fix capacity')
plt.show()
#plt.savefig(figpath + 'hyd_curt_line_tnt1.png',facecolor='w',bbox_inches='tight',dpi=600)


v = vrecurt.drop(columns=['technology','load_zone']).groupby(['scs','period']).sum().reset_index()
fig, ax = plt.subplots()
sns.lineplot(v,x='period',y='twh', units='scs', estimator=None)
plt.title('VRE curtailment - fix capacity')
plt.show()
#plt.savefig(figpath + 'vre_curt_line_tnt1.png',facecolor='w',bbox_inches='tight',dpi=600)









ss = ['prm15_fix_wcost_re1_c2p1_tnt1','prm15_fix_wcost_re17_c2p1_tnt1','prm15_fix_wcost_re17_c2p1_co2_cut90p_tnt1']
load_bal = pd.read_csv(respath+'prm15_fix_wcost_re1_c2p1_co2_cut90p_tnt1/results/load_balance.csv')
load_bal['month'] = load_bal['timepoint'].astype(str).str[4:6]
load_bal = load_bal.set_index(['zone','period','month','timepoint'])
load_bal['prm15_fix_wcost_re1_c2p1_co2_cut90p_tnt1'] = load_bal['unserved_energy_mw']*load_bal['timepoint_weight']/(10**6)
load_bal = pd.DataFrame(load_bal['prm15_fix_wcost_re1_c2p1_co2_cut90p_tnt1'])
for i in range(len(ss)):
    temp_lb=pd.read_csv(respath+ss[i]+'/results/load_balance.csv')
    temp_lb['month'] = temp_lb['timepoint'].astype(str).str[4:6]
    temp_lb = temp_lb.set_index(['zone','period','month','timepoint'])
    temp_lb[ss[i]] = temp_lb['unserved_energy_mw']*temp_lb['timepoint_weight']/(10**6)
    temp_lb = pd.DataFrame(temp_lb[ss[i]])
    load_bal = load_bal.merge(temp_lb, on=['zone','period','month','timepoint'])
load_bal = load_bal.groupby(['period','month','zone']).sum().reset_index()
load_bal.reset_index().to_csv(respath + 'unmet_demand.csv',index=False)



untn = pd.read_csv('unmet_demand_40.csv')
unt1 = pd.read_csv('unmet_demand.csv')
unt = untn.merge(unt1, on=['zone','period','timepoint'])
unt['month'] = unt['timepoint'].astype(str).str[4:6]
unt = unt.groupby(['period','month','zone']).sum().reset_index().drop(columns='timepoint')

df1 = load_bal.groupby(['period','month']).sum().reset_index().iloc[:,3:]
df2 = unt.groupby(['period','month']).sum().reset_index().iloc[:,3:]
df2 /= (10**6)

fig, ax = plt.subplots()
g = sns.lineplot(df1, palette=['k']*4, ax=ax, linewidth=0.6)
g2 = sns.lineplot(df2, ax=ax, palette=['r']*80, linewidth=0.6, zorder=0)
ax.legend([],frameon=False)
plt.title('unmet demand for 4 sample scenarios compared to other 80 in red')
#plt.show()
plt.savefig('unmet_demand_tnt1.png', dpi=400, bbox_inches='tight')






