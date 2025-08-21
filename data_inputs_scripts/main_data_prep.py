# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:43:04 2024

@author: jwesse03
"""

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import pyarrow

### GENERAL INFORMATION / VARIABLES ###
respath = 'outputs/lim_coord/' # ASSUMES DATA CAN BE FOUND IN outputs/ BY SCENARIO NAME
respath1 = 'outputs/full_coord/'
datapath = 'figure_data/'
sc_names = pd.read_csv(datapath + 'scenario_names.csv')
mercosur = ['Argentina','Brazil','Chile','Paraguay','Uruguay']
periods = [2020,2025,2030,2035,2040,2045,2050]

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

real_names=[sc_names['full_name'][i] for i in range(len(sc_names)) if sc_names['short_name'][i] in scs80]
style_df = pd.DataFrame({'scenario':scs80,'real_name':real_names}).set_index('scenario')
tech_mapper = {'Coal_Sub':'Coal','Coal_Sup':'Coal','Coal_IGCC':'Coal','Gas_CCGT':'Gas','Gas_OCGT':'Gas','Gas_ICE':'Gas',
               'Oil':'Diesel','SolarPV':'Solar','Wind':'Wind','Geothermal':'Geothermal','Biomass':'Biomass','Diesel':'Diesel',
               'Nuclear':'Nuclear','Hydro':'Hydro','Battery':'Battery'}


#%% CAPACITY MIX

# list_pt_all_cap_tech = []
# for i in range(len(scs80)):
#     if i < 40:
#         all_cap = pd.read_csv(respath+str(scs80[i])+'/results/capacity_all.csv')
#     else:
#         all_cap = pd.read_csv(respath1+str(scs80[i])+'/results/capacity_all.csv')
#     ### rename tech of existing and new capacities for plotting 
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_new_lin') & (all_cap['technology'] == 'Coal_Sub'),'technology'] = 'Coal_new'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_new_lin') & (all_cap['technology'] == 'Coal_Sup'),'technology'] = 'Coal_new'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_new_lin') & (all_cap['technology'] == 'Coal_IGCC'),'technology'] = 'Coal_new'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_new_lin') & (all_cap['technology'] == 'Gas_CCGT'),'technology'] = 'Gas_new'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_new_lin') & (all_cap['technology'] == 'Gas_ICE'),'technology'] = 'Gas_new'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_new_lin') & (all_cap['technology'] == 'Gas_OCGT'),'technology'] = 'Gas_new'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_new_lin') & (all_cap['technology'] == 'SolarPV'),'technology'] = 'Solar_new'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_new_lin') & (all_cap['technology'] == 'SolarCSP'),'technology'] = 'Solar_new'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_new_lin') & (all_cap['technology'] == 'Wind'),'technology'] = 'Wind_new'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_new_bin') & (all_cap['technology'] == 'Hydro'),'technology'] = 'Hydro_new'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_new_lin') & (all_cap['technology'] == 'Nuclear'),'technology'] = 'Nuclear_new'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_new_lin') & (all_cap['technology'] == 'Diesel'),'technology'] = 'Diesel_new'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_new_lin') & (all_cap['technology'] == 'Geothermal'),'technology'] = 'Geothermal_new'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_new_lin') & (all_cap['technology'] == 'Biomass'),'technology'] = 'Biomass_new'
#     all_cap.loc[(all_cap['capacity_type'] == 'stor_new_lin') & (all_cap['technology'] == 'Battery'),'technology'] = 'Battery_new'
    
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_spec') & (all_cap['technology'] == 'Coal_Sub'),'technology'] = 'Coal'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_spec') & (all_cap['technology'] == 'Coal_Sup'),'technology'] = 'Coal'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_spec') & (all_cap['technology'] == 'Coal_IGCC'),'technology'] = 'Coal'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_spec') & (all_cap['technology'] == 'Gas_CCGT'),'technology'] = 'Gas'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_spec') & (all_cap['technology'] == 'Gas_ICE'),'technology'] = 'Gas'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_spec') & (all_cap['technology'] == 'Gas_OCGT'),'technology'] = 'Gas'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_spec') & (all_cap['technology'] == 'SolarPV'),'technology'] = 'Solar'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_spec') & (all_cap['technology'] == 'SolarCSP'),'technology'] = 'Solar'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_spec') & (all_cap['technology'] == 'Wind'),'technology'] = 'Wind'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_spec') & (all_cap['technology'] == 'Hydro'),'technology'] = 'Hydro'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_spec') & (all_cap['technology'] == 'Nuclear'),'technology'] = 'Nuclear'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_spec') & (all_cap['technology'] == 'Diesel'),'technology'] = 'Diesel'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_spec') & (all_cap['technology'] == 'Geothermal'),'technology'] = 'Geothermal'
#     all_cap.loc[(all_cap['capacity_type'] == 'gen_spec') & (all_cap['technology'] == 'Biomass'),'technology'] = 'Biomass'
#     ### total capacity by period and renamed tech
#     all_cap_tech = all_cap.groupby(['period','load_zone','technology'])['capacity_mw'].sum()/1000 
#     all_cap_tech = all_cap_tech.reset_index()
#     ### create pivot table
#     pt_all_cap_tech = all_cap_tech.pivot(index=['period','load_zone'], columns='technology', values='capacity_mw')
#     ### assign scenario name in a column
#     pt_all_cap_tech['scs'] = scs80[i]
#     ### append to the pivot table to list
#     list_pt_all_cap_tech.append(pt_all_cap_tech)
#     print(str(i))

# df_pt_all_cap_tech_allscs = pd.concat([list_pt_all_cap_tech[i] for i in range(len(scs80))])

# capacity_mix = df_pt_all_cap_tech_allscs.copy(deep=True).reset_index().fillna(0)
# sys_wide = capacity_mix.groupby(['period','scs']).sum().reset_index()
# sys_wide['load_zone'] = 'Region-wide'
# capacity_mix = pd.concat([capacity_mix,sys_wide])
# capacity_mix.to_csv(datapath+'capacity_mix.csv',index=False)

#%% CAPACITY MIX checking

# all_cap = pd.read_csv(respath+str(scs80[i])+'/results/capacity_all.csv')
# cap = all_cap[(all_cap['capacity_type']=='gen_new_lin')&(all_cap['load_zone']=='Brazil')&(all_cap.technology=='Wind_new')&(all_cap.period==2050)].drop(columns=['capacity_mwh'])
# capacity_gen_new_lin = pd.read_csv(respath+scs80[35]+'/results/capacity_gen_new_lin.csv')
# capacity_gen_new_lin = capacity_gen_new_lin[(capacity_gen_new_lin['technology'] == 'Wind') |\
#                                             (capacity_gen_new_lin['technology'] == 'SolarPV')]
# capg = capacity_gen_new_lin[(capacity_gen_new_lin['load_zone']=='Brazil')&(capacity_gen_new_lin['technology']=='Wind')]
# capgs = capg.groupby(['project','technology','load_zone']).sum().reset_index().drop(columns=['vintage','technology','load_zone'])
# cap_gs = cap.merge(capgs, on='project')
# cap_gs['diff'] = cap_gs['capacity_mw']-cap_gs['new_build_mw']

# proj_diff =    ['p_wind_Brazil_BR2823',
#                 'p_wind_Brazil_BR2824',
#                 'p_wind_Brazil_BR2814',
#                 'p_wind_Brazil_BR2803',
#                 'p_wind_Brazil_BR2812',
#                 'p_wind_Brazil_BR2801',
#                 'p_wind_Brazil_BR2673',
#                 'p_wind_Brazil_BR2772',
#                 'p_wind_Brazil_BR2813',
#                 'p_wind_Brazil_BR2832',
#                 'p_wind_Brazil_BR2787',
#                 'p_wind_Brazil_BR2811',
#                 'p_wind_Brazil_BR2800',
#                 'p_wind_Brazil_BR2831',
#                 'p_wind_Brazil_BR2706',
#                 'p_wind_Brazil_BR2802']

#%% GENERATION MIX AND CURTAILMENT AND TRANSMISSION LOSSES

# list_pt_gen_loss_curt = []
# for i in range(len(scs80)):
#     if i < 40:
#         path = respath
#     else:
#         path = respath1
#     ### Generation-TWh 2020-2050
#     alldisp1 = pd.read_csv(path+''+str(scs80[i])+'/results/dispatch_all.csv')    
#     alldisp1['twh'] = alldisp1['timepoint_weight'] * alldisp1['power_mw']/(10**6)    
#     alldisp1_tec_twh_nstor=alldisp1[(alldisp1.operational_type!='stor')].groupby(['period','load_zone','technology'])['twh'].sum()
#     alldisp1_twh_stor=alldisp1[(alldisp1.operational_type=='stor')&(alldisp1.twh<=0)].groupby(['period','load_zone','technology'])['twh'].sum()
#     alldisp1_tec_twh_ns = pd.concat([alldisp1_tec_twh_nstor,alldisp1_twh_stor]).reset_index()
#     ### Curtailment-VRE 2020-2050
#     disp_re = pd.read_csv(path+''+str(scs80[i])+'/results/dispatch_variable.csv')
#     disp_re['twh'] = disp_re['total_curtailment_mw']*disp_re['timepoint_weight']/(10**6)
#     curt_re = disp_re.groupby(['period','load_zone'])['twh'].sum() #[(disp_re.period!=2045)]
#     curt_re = curt_re.reset_index()
#     curt_re['technology']='curt_vre'
#     ### Curtailment-Other 2020-2050
#     disp_h = pd.read_csv(path+''+str(scs80[i])+'/results/dispatch_gen_hydro.csv')
#     disp_h['twh'] = disp_h['scheduled_curtailment_mw']*disp_h['timepoint_weight']/(10**6)
#     curt_oth = disp_h.groupby(['period','load_zone'])['twh'].sum() #[(disp_h.period!=2045)]
#     curt_oth = curt_oth.reset_index()
#     curt_oth['technology']='curt_other'
#     ### concat Gen_TWh, Trans_loss, and Curtailments
#     df_gen_loss_curt = pd.concat([alldisp1_tec_twh_ns, curt_re, curt_oth],axis=0)
#     df_gen_loss_curt.set_index('period').sort_index()
#     ### rename tech for plotting
#     df_gen_loss_curt.loc[df_gen_loss_curt['technology'] == 'Coal_Sub','technology'] = 'Coal'
#     df_gen_loss_curt.loc[df_gen_loss_curt['technology'] == 'Coal_Sup','technology'] = 'Coal'
#     df_gen_loss_curt.loc[df_gen_loss_curt['technology'] == 'Coal_IGCC','technology'] = 'Coal'
#     df_gen_loss_curt.loc[df_gen_loss_curt['technology'] == 'Gas_CCGT','technology'] = 'Gas'
#     df_gen_loss_curt.loc[df_gen_loss_curt['technology'] == 'Gas_OCGT','technology'] = 'Gas'
#     df_gen_loss_curt.loc[df_gen_loss_curt['technology'] == 'Gas_ICE','technology'] = 'Gas'
#     df_gen_loss_curt.loc[df_gen_loss_curt['technology'] == 'Diesel','technology'] = 'Diesel'
#     df_gen_loss_curt.loc[df_gen_loss_curt['technology'] == 'Oil','technology'] = 'Diesel'
#     df_gen_loss_curt.loc[df_gen_loss_curt['technology'] == 'Geothermal','technology'] = 'Geothermal'
#     df_gen_loss_curt.loc[df_gen_loss_curt['technology'] == 'Biomass','technology'] = 'Biomass'
#     df_gen_loss_curt.loc[df_gen_loss_curt['technology'] == 'SolarPV','technology'] = 'Solar'
#     df_gen_loss_curt.loc[df_gen_loss_curt['technology'] == 'curt_vre','technology'] = 'Curtailment'
#     df_gen_loss_curt.loc[df_gen_loss_curt['technology'] == 'curt_other','technology'] = 'Curtailment'
#     ### sum of twh by renamed tech and period
#     df_gen_loss_curt_reclass = df_gen_loss_curt.groupby(['period','load_zone','technology'])['twh'].sum().reset_index()
#     ### create pivot table
#     pt_gen_loss_curt = df_gen_loss_curt_reclass.pivot(index=['period','load_zone'], columns='technology', values='twh')
#     ### assign scenario name in a column
#     pt_gen_loss_curt['scs'] = scs80[i]
#     ### append to the pivot table to list
#     list_pt_gen_loss_curt.append(pt_gen_loss_curt)
#     print(str(i))
    
# df_pt_gen_loss_curt_allscs = pd.concat([list_pt_gen_loss_curt[i] for i in range(len(scs80))])

# generation_mix = df_pt_gen_loss_curt_allscs.copy(deep=True).reset_index().fillna(0)
# sys_wide_gen = generation_mix.groupby(['period','scs']).sum().reset_index()
# sys_wide_gen['load_zone'] = 'Region-wide'
# generation_mix = pd.concat([generation_mix,sys_wide_gen])
# generation_mix.to_csv(datapath+'generation_mix.csv',index=False)

# # transmission losses
# list_tx_loss = []
# for i in range(len(scs80)):
#     if i < 40:
#         path = respath
#     else:
#         path = respath1
#     ### Trans_Loss 2020-2050
#     trans = pd.read_csv(path+''+str(scs80[i])+'/results/transmission_operations.csv')
#     trans['twh'] = (trans['transmission_losses_lz_from']+trans['transmission_losses_lz_to'])*trans['timepoint_weight']/(10**6)
#     df_trans_loss = pd.DataFrame([trans.groupby(['period','tx_line'])['twh'].sum()*(-1)]).transpose()
#     df_trans_loss['technology']='Trans_loss'
#     df_trans_loss['scs'] = scs_short[i]
#     df_trans_loss = df_trans_loss.reset_index()
#     list_tx_loss.append(df_trans_loss)
#     print(str(i))
# df_tx_loss_allscs = pd.concat([list_tx_loss[i] for i in range(len(scs80))])
# df_tx_loss_allscs.to_csv(datapath+'transmission_losses.csv',index=False)

#%% COST COMPONENTS AND EMISSIONS

# all_costs_emissions = pd.DataFrame()
# for i in range(len(scs80)):
#     if i < 40:
#         path = respath
#     else:
#         path = respath1
#     # operations costs 2020-2050
#     op_costs = pd.read_csv(path+''+str(scs80[i])+'/results/costs_operations.csv')    
#     # multiply by timepoint_weight
#     op_costs['variable_om_cost_total'] = op_costs['variable_om_cost']*op_costs['timepoint_weight']*op_costs['number_of_hours_in_timepoint']
#     op_costs['fuel_cost_total'] = op_costs['fuel_cost'] * op_costs['timepoint_weight'] * op_costs['number_of_hours_in_timepoint']
#     op_costs['startup_cost_total'] = op_costs['startup_cost'] * op_costs['timepoint_weight'] * op_costs['number_of_hours_in_timepoint']
#     # check whether timepoint should be multiplied
#     op_costs_summary = op_costs.groupby(['period','load_zone'])[['variable_om_cost_total', 'fuel_cost_total', 'startup_cost_total']].sum()
#     # capacity costs 2020-2050
#     capacity_costs = pd.read_csv(path+''+str(scs80[i])+'/results/costs_capacity_all_projects.csv')    
#     capacity_costs_summary = capacity_costs.groupby(['period','load_zone'])[['capacity_cost']].sum()
#     # transmission capacity costs 2020-2050
#     transmission_costs = pd.read_csv(path+''+str(scs80[i])+'/results/costs_transmission_capacity.csv')
#     transmission_costs = transmission_costs.rename(columns={"load_zone_from": "load_zone"})
#     transmission_costs_summary = transmission_costs.groupby(['period','load_zone'])[['capacity_cost']].sum()
#     transmission_costs_summary.rename(columns = {'capacity_cost': 'transmission_capacity_cost'}, inplace=True)
#     # existing capital capacity costs
#     all_cap = pd.read_csv(path+''+str(scs80[i])+'/results/capacity_all.csv')
#     ecap = all_cap[all_cap.capacity_type == 'gen_spec']
#     ecap_tech = ecap.groupby(['period','technology','load_zone'])[['capacity_mw']].sum()
#     ecap_tech.reset_index(inplace=True)
#     # energy demand 
#     load = pd.read_csv(path+''+str(scs80[i])+'/results/load_balance.csv')
#     load['mwh'] = load['timepoint_weight'] * load['load_mw']
#     load = load.rename(columns={"zone": "load_zone"})
#     load_summary = load.groupby(['period','load_zone'])['mwh'].sum()
#     # emissions
#     emissions = pd.read_csv(path+''+str(scs80[i])+'/results/carbon_emissions_by_project.csv')
#     emissions['MtCO2'] = emissions['timepoint_weight'] * emissions['carbon_emissions_tons'] / 10**6  
#     emissions_summary = emissions.groupby(['period','load_zone'])['MtCO2'].sum()
#     # concatenate costs and sum
#     ce_df = pd.concat([op_costs_summary, capacity_costs_summary, transmission_costs_summary, load_summary, emissions_summary], axis=1)
#     cols_to_include=['variable_om_cost_total','fuel_cost_total','startup_cost_total','capacity_cost','transmission_capacity_cost']
    
#     ce_df['tCO2_per_mwh'] = ce_df['MtCO2']/ce_df['mwh']*10**6
#     ce_df['total_costs'] = ce_df.loc[:,ce_df.columns.isin(cols_to_include)].sum(axis=1)
#     ce_df['costs_per_mwh'] = ce_df['total_costs']/ce_df['mwh']
#     ce_df['var_om_cost_per_mwh'] = ce_df['variable_om_cost_total']/ce_df['mwh']
#     ce_df['fuel_cost_per_mwh'] = ce_df['fuel_cost_total']/ce_df['mwh']
#     ce_df['startup_cost_per_mwh'] = ce_df['startup_cost_total']/ce_df['mwh']
#     ce_df['capacity_cost_per_mwh'] = ce_df['capacity_cost']/ce_df['mwh']
#     ce_df['tx_capacity_cost_per_mwh'] = ce_df['transmission_capacity_cost']/ce_df['mwh']
#     # add scenario
#     ce_df['scs_id'] = scs80[i]
#     ce_df.reset_index()
#     all_costs_emissions = pd.concat([all_costs_emissions, ce_df])
#     print(str(i))
# cost_emis_df = all_costs_emissions.reset_index()
# cost_emis_df = cost_emis_df.fillna(0)
# cost_emis_df.to_csv(datapath + 'costs_emissions.csv',index=False)

#%% CAPACITY COSTS BY TECHNOLOGY

# all_costs_scs = pd.DataFrame()
# for i in range(len(scs80)):
#     if i < 40:
#         path = respath
#     else:
#         path = respath1
#     # capacity costs 2020-2050
#     capacity_costs = pd.read_csv(path+''+str(scs80[i])+'/results/costs_capacity_all_projects.csv')    
#     capacity_costs['tech'] = capacity_costs['technology'].map({'Biomass':'Biomass','Coal_Sub':'Fossil','Coal_IGCC':'Fossil','Battery':'Battery',
#                                                'Diesel':'Fossil','Coal_Sup':'Fossil','Gas_CCGT':'Fossil','Gas_OCGT':'Fossil','Gas_ICE':'Fossil',
#                                                'Geothermal':'Geothermal','Hydro':'Hydro','Nuclear':'Nuclear','SolarPV':'Solar','Wind':'Wind'})
#     capacity_costs_summary = capacity_costs[capacity_costs.period>=2050].groupby(['period','tech'])[['capacity_cost']].sum()
#     # add scenario
#     capacity_costs_summary['scs_id'] = scs80[i]
#     capacity_costs_summary.reset_index()
#     all_costs_scs = pd.concat([all_costs_scs, capacity_costs_summary])
#     print(str(i))
# costt_df = all_costs_scs.reset_index()
# costt_df = costt_df.fillna(0).drop(columns='period')[['scs_id','tech','capacity_cost']]
# costt_df.to_csv(datapath + 'costs_by_tech.csv',index=False)


#%% VRE PREFERENCE

# # read in list of possible projects
# solar_projects = pd.read_csv(datapath+'projdata_mercosur_solar_planned.csv').drop(columns=['index'])
# wind_projects = pd.read_csv(datapath+'projdata_mercosur_wind_planned.csv').drop(columns=['index'])
# #update capacity factors to specific data used
# solar_projects = solar_projects.rename(columns={'ann_cfs_ninja_solar_1axis':'cf'})
# wind_projects = wind_projects.rename(columns={'ann_cfs_ninja_wind_vestas_v90_2mw_100m':'cf'})
# re_proj = pd.concat([solar_projects, wind_projects])
# # initialize dataframe to store model outputs as a single tidy dataset
# new_lin = pd.DataFrame(columns=['project','vintage','technology','load_zone','new_build_mw','scenario'])
# # loop through each scenario named in 'scenarios' to get solar and wind builds
# for i in range(len(scs80)):
#     if i < 40:
#         path = respath
#     else:
#         path = respath1
#     capacity_gen_new_lin = pd.read_csv(path+scs80[i]+'/results/capacity_gen_new_lin.csv')
#     capacity_gen_new_lin = capacity_gen_new_lin[(capacity_gen_new_lin['technology'] == 'Wind') |\
#                                                 (capacity_gen_new_lin['technology'] == 'SolarPV')]
#     capacity_gen_new_lin['scenario'] = scs80[i]
#     new_lin = pd.concat([new_lin, capacity_gen_new_lin])
#     print(str(i))
# # re-index for organization and performance
# new_lin.set_index(['project','vintage','load_zone'],inplace=True)
# new_lin.sort_index(inplace=True)
# # add latitude, longitude, potential capacity, capacity factor, transmission distance, and build binary
# new_lin[['lat','lon','cap_mw','annual_cf','tx_dist']] = np.nan
# for project in re_proj['project'].unique():
#     new_lin.loc[project,['lat','lon','cap_mw','annual_cf','tx_dist']] =\
#                 re_proj[re_proj['project'] == project][['Latitude','Longitude','cap_mw','cf','d_trans_km']].values
# new_lin['new_build_binary'] = [new_lin.iloc[i]['new_build_mw'] > 0 for i in range(len(new_lin))]
# new_lin.to_csv(datapath + 'new_lin.csv')

# new_lin = pd.read_csv(datapath + 'new_lin.csv')
# caps = new_lin[['project','cap_mw']].groupby('project').mean().reset_index()
# new_lin_tot = new_lin[['project','load_zone','technology','scenario','new_build_mw']].copy(deep=True).reset_index().drop(columns='index')
# new_lin_totals = new_lin_tot.groupby(['project','load_zone','technology','scenario']).sum().reset_index()
# new_lin_totals['cap_mw'] = new_lin_totals['project'].map(caps.set_index('project')['cap_mw'])
# new_lin_totals['new_cap_mw'] = new_lin_totals['cap_mw']*(new_lin_totals['new_build_mw'] >= new_lin_totals['cap_mw']) + \
#                                       new_lin_totals['new_build_mw']*(new_lin_totals['new_build_mw'] < new_lin_totals['cap_mw'])
# new_lin_totals['new_build_mw'] = new_lin_totals['new_cap_mw']
# new_lin_systot = new_lin_totals.groupby(['scenario','technology','load_zone']).sum().reset_index()
# scaled_df = new_lin_systot.copy(deep=True).reset_index().drop(columns='index')
# scaled_df['emis_policy'] = scaled_df.scenario.str.contains('cut90p')*1
# scaled_df['exist_trd'] = scaled_df.scenario.str.contains('t1')*1
# scaled_df['op_char'] = scaled_df.scenario.str.split('_', expand=True)[2].str[2:].astype(int)
# scaled_df['fixed_pv'] = (scaled_df.op_char > 10)*1
# scaled_df['pol'] = scaled_df.emis_policy.astype(str) + scaled_df.exist_trd.astype(str)
# scaled_df['group'] = scaled_df.emis_policy.astype(str) + scaled_df.exist_trd.astype(str) + scaled_df.fixed_pv.astype(str)
# scaled_df.pol.replace({'00':'NoCut,Full','01':'NoCut,Lim','10':'90%Cut,Full','11':'90%Cut,Lim'},inplace=True)
# difforder=CategoricalDtype(['NoCut,Lim','NoCut,Full','90%Cut,Lim','90%Cut,Full'], ordered=True)
# scaled_df['pol'] = scaled_df['pol'].astype(difforder)
# scaled_df.sort_values('pol',inplace=True)
# scaled_df.set_index('scenario', inplace=True)
# scaled_df.to_csv(datapath + 'parallel_axis_vre_df_country.csv')

#%% PARALLEL AXIS VRE MAP DATA

# sol_new_biome = pd.read_csv(datapath + 'solar_new_biomes.csv')
# wind_new_biome = pd.read_csv(datapath + 'wind_new_biomes.csv')
# projects = pd.concat([sol_new_biome,wind_new_biome])
# new_lin = pd.read_csv(datapath + 'new_lin.csv')
# caps = new_lin[['project','cap_mw']].groupby('project').mean().reset_index()
# new_lin_bio = new_lin.copy(deep=True).reset_index().drop(columns='index')
# new_lin_bio['biome'] = new_lin_bio['project'].map(projects.set_index('project')['biome'])
# new_lin_bio_totals = new_lin_bio.drop(columns='vintage').groupby(['project','load_zone','technology','scenario','biome']).sum().reset_index()
# new_lin_bio_totals = new_lin_bio_totals[['project','load_zone','technology','scenario','biome','new_build_mw']]
# new_lin_bio_totals['cap_mw'] = new_lin_bio_totals['project'].map(caps.set_index('project')['cap_mw'])
# new_lin_bio_totals['new_cap_mw'] = new_lin_bio_totals['cap_mw']*(new_lin_bio_totals['new_build_mw'] >= new_lin_bio_totals['cap_mw']) + \
#                                      new_lin_bio_totals['new_build_mw']*(new_lin_bio_totals['new_build_mw'] < new_lin_bio_totals['cap_mw'])
# new_lin_bio_totals['new_build_mw'] = new_lin_bio_totals['new_cap_mw']
# new_lin_bio_grp = new_lin_bio_totals.groupby(['scenario','technology','biome']).sum().reset_index()
# new_lin_bio_grp['region'] = new_lin_bio_grp['technology'] + ' ' + new_lin_bio_grp['biome']
# new_vre_pt = pd.pivot_table(new_lin_bio_grp,values='new_build_mw',index='scenario',columns='region')
# scaled_df = new_vre_pt.copy(deep=True).reset_index()
# scaled_df['emis_policy'] = scaled_df.scenario.str.contains('cut90p')*1
# scaled_df['exist_trd'] = scaled_df.scenario.str.contains('t1')*1
# scaled_df['op_char'] = scaled_df.scenario.str.split('_', expand=True)[2].str[2:].astype(int)
# scaled_df['fixed_pv'] = (scaled_df.op_char > 10)*1
# scaled_df['pol'] = scaled_df.emis_policy.astype(str) + scaled_df.exist_trd.astype(str)
# scaled_df['group'] = scaled_df.emis_policy.astype(str) + scaled_df.exist_trd.astype(str) + scaled_df.fixed_pv.astype(str)
# scaled_df.pol.replace({'00':'NoCut,Full','01':'NoCut,Lim','10':'90%Cut,Full','11':'90%Cut,Lim'},inplace=True)
# difforder=CategoricalDtype(['NoCut,Lim','NoCut,Full','90%Cut,Lim','90%Cut,Full'], ordered=True)
# scaled_df['pol'] = scaled_df['pol'].astype(difforder)
# scaled_df.sort_values('pol',inplace=True)
# scaled_df.set_index('scenario', inplace=True)
# scaled_df.to_csv(datapath + 'parallel_axis_vre_df.csv')

#%% ADDING SPLIT TRANSMISSION COSTS TO COST DF

# ce_df = pd.read_csv(datapath + 'costs_emissions.csv')
# ce_df.sort_values(by='scs_id', key=lambda column: column.map(lambda e: scs80.index(e)), inplace=True)
# ce_df = ce_df[['period','load_zone','variable_om_cost_total','fuel_cost_total','startup_cost_total','capacity_cost',
#                'transmission_capacity_cost','mwh','MtCO2','total_costs','scs_id']].reset_index(drop=True)
# ce_df_2050 = ce_df.groupby(['load_zone','scs_id']).sum().reset_index()
# ce_sw_2050 = ce_df.groupby(['scs_id']).sum().reset_index()
# ce_sw_2050['load_zone'] = 'Region-wide'
# cost_df = pd.concat([ce_df_2050,ce_sw_2050])
# cost_df['operations_cost_total'] = cost_df['variable_om_cost_total'] + cost_df['fuel_cost_total'] + cost_df['startup_cost_total']

# cost_df['emis_policy'] = cost_df.scs_id.str.contains('cut90p')*1
# cost_df['exist_trd'] = cost_df.scs_id.str.contains('t1')*1
# cost_df['op_char'] = cost_df.scs_id.str.split('_', expand=True)[2].str[2:].astype(int)
# cost_df['fixed_pv'] = (cost_df.op_char > 10)*1
# cost_df['pol'] = cost_df.emis_policy.astype(str) + cost_df.exist_trd.astype(str)
# cost_df['group'] = cost_df.emis_policy.astype(str) + cost_df.exist_trd.astype(str) + cost_df.fixed_pv.astype(str)
# cost_df.pol.replace({'00':'NoCut,Full','01':'NoCut,Lim','10':'90%Cut,Full','11':'90%Cut,Lim'},inplace=True)
# difforder=CategoricalDtype(['NoCut,Lim','NoCut,Full','90%Cut,Lim','90%Cut,Full'], ordered=True)
# cost_df['pol'] = cost_df['pol'].astype(difforder)
# cost_df.sort_values('pol',inplace=True)

# new_tx_costs = pd.DataFrame()
# for i in range(len(scs80)):
#     if i < 40:
#         path = respath
#     else:
#         path = respath1
#     tx_cost = pd.read_csv(path + scs80[i] + '/results/costs_transmission_capacity.csv')
#     txdf = tx_cost[tx_cost.tx_line.str.contains('New')][['period','load_zone_from','load_zone_to','capacity_cost']]\
#             .groupby(['load_zone_from','load_zone_to']).sum().reset_index().drop(columns=['period'])
#     tx_split = pd.concat([txdf[['load_zone_from','capacity_cost']].rename(columns={'load_zone_from':'load_zone'}),
#                           txdf[['load_zone_to','capacity_cost']].rename(columns={'load_zone_to':'load_zone'})])
#     tx_split['capacity_cost'] /= 2
#     tx_tot = tx_split.groupby(['load_zone']).sum().reset_index()
#     tx_tot['scs_id'] = scs80[i]
#     new_tx_costs = pd.concat([new_tx_costs,tx_tot])
#     print(str(i))
# tx_sw = new_tx_costs.groupby('scs_id').sum().reset_index()
# tx_sw['load_zone'] = 'Region-wide'
# new_tx_costs = pd.concat([new_tx_costs,tx_sw]).reset_index(drop=True).rename(columns={'capacity_cost':'split_tx_cost'})

# split_tx_cost_df = cost_df.merge(new_tx_costs,on=['load_zone','scs_id'],how='outer')
# split_tx_cost_df.split_tx_cost.fillna(0,inplace=True)
# split_tx_cost_df['new_total_cost']=split_tx_cost_df['operations_cost_total']+split_tx_cost_df['capacity_cost']+split_tx_cost_df['split_tx_cost']
# location_legend = [-1.15,-0.42]
# cost_cols = ['capacity_cost','operations_cost_total','split_tx_cost','new_total_cost']
# cost_labels = ['Gen. Capacity Costs ($B)','Grid Operations Costs ($B)','Transmission Costs ($B)','Total Costs ($B)']
# split_tx_cost_df[cost_cols] /= 1000000000
# split_tx_cost_df = split_tx_cost_df.drop(columns='period')
# split_tx_cost_df.to_csv(datapath + 'split_tx_cost_df.csv',index=False)

#%% NEW TRANSMISSION CAPACITY

# tx_df = pd.DataFrame()
# for i in range(len(scs80)):
#     if i < 40:
#         path = respath
#     else:
#         path = respath1
#     try:
#         tx_new = pd.read_csv(path+scs80[i]+'/results/transmission_new_capacity.csv')
#         tx_new['scs'] = scs80[i]
#         tx_df = pd.concat([tx_df,tx_new])
#     except:
#         pass
#     print(str(i))
# tx_df = tx_df[(tx_df.transmission_line.isin(['Argentina_Brazil_New','Argentina_Chile_New','Argentina_Uruguay_New']))]
# tx_df = tx_df.groupby(['scs','transmission_line']).sum().reset_index()
# tx_df['new_build_transmission_capacity_mw']=tx_df['new_build_transmission_capacity_mw']/1000
# tx_tidy = tx_df.copy(deep=True)#pd.melt(tx_df,id_vars=['transmission_line'],value_vars=scs1,var_name='scenario')
# tx_tidy['emis_policy'] = tx_tidy.scs.str.contains('cut90p')*1
# tx_tidy['exist_trd'] = tx_tidy.scs.str.contains('t1')*1
# tx_tidy['op_char'] = tx_tidy.scs.str.split('_', expand=True)[2].str[2:].astype(int)
# tx_tidy['fixed_pv'] = (tx_tidy.op_char > 10)*1
# tx_tidy['pol'] = tx_tidy.emis_policy.astype(str) + tx_tidy.exist_trd.astype(str)
# tx_tidy['group'] = tx_tidy.emis_policy.astype(str) + tx_tidy.exist_trd.astype(str) + tx_tidy.fixed_pv.astype(str)
# tx_tidy.pol.replace({'00':'NoCut,Full','01':'NoCut,Lim','10':'90%Cut,Full','11':'90%Cut,Lim'},inplace=True)
# difforder=CategoricalDtype(['Argentina_Uruguay_New','Argentina_Brazil_New','Argentina_Chile_New'], ordered=True)
# tx_tidy['transmission_line'] = tx_tidy['transmission_line'].astype(difforder)
# tx_tidy.sort_values('transmission_line',inplace=True)
# tx_tidy.to_csv(datapath + 'new_tx_cap.csv', index = False)

#%% HOURLY DISPATCH

# list_hourly_disp_load = []
# for i in range(len(scs)):
#     ###Generation-TWh 2020-2050
#     alldisp1 = pd.read_csv(respath+scs[i]+'/results/dispatch_all.csv')
#     ###rename tech for plotting
#     alldisp1['technology'] = alldisp1.technology.map(tech_mapper)
#     alldisp1.loc[(alldisp1['technology'] == 'Battery')&(alldisp1['power_mw'] < 0),'technology'] = 'Battery_charge'
#     alldisp1.loc[(alldisp1['technology'] == 'Battery')&(alldisp1['power_mw'] >= 0),'technology'] = 'Battery_discharge'
#     alldisp1 = alldisp1.fillna(0)
#     ###system-wide dispatch by timepoints and tech
#     df_hourly_disp = alldisp1.groupby(['timepoint','technology','load_zone'])['power_mw'].sum()/1000
#     df_hourly_disp = df_hourly_disp.reset_index()
#     ###create pivot table
#     pt_hourly_disp = df_hourly_disp.pivot(index=['timepoint','load_zone'], columns='technology', values='power_mw')
#     ###calculate load by timepoints
#     load_bal = pd.read_csv(respath+scs[i]+'/results/load_balance.csv')
#     load_bal = load_bal.groupby(['timepoint','zone'])['load_mw'].sum()/1000
#     pt_hourly_disp['load'] = load_bal
#     ###calculate curtailment by timepoints
#     curt_df = pd.read_csv(respath+scs[i]+'/results/dispatch_variable.csv')
#     curt_df = curt_df.groupby(['timepoint','load_zone'])['total_curtailment_mw'].sum()/1000
#     pt_hourly_disp['curt'] = curt_df
#     ###include scenario name and append to list
#     pt_hourly_disp['scs'] = scs[i]
#     list_hourly_disp_load.append(pt_hourly_disp)
#     print(i)

# #list_hourly_disp_load1 = []
# for i in range(len(scs1)):
#     ###Generation-TWh 2020-2050
#     alldisp1 = pd.read_csv(respath1+scs1[i]+'/results/dispatch_all.csv')
#     ###rename tech for plotting
#     alldisp1['technology'] = alldisp1.technology.map(tech_mapper)
#     alldisp1.loc[(alldisp1['technology'] == 'Battery')&(alldisp1['power_mw'] < 0),'technology'] = 'Battery_charge'
#     alldisp1.loc[(alldisp1['technology'] == 'Battery')&(alldisp1['power_mw'] >= 0),'technology'] = 'Battery_discharge'
#     alldisp1 = alldisp1.fillna(0)
#     ###system-wide dispatch by timepoints and tech
#     df_hourly_disp = alldisp1.groupby(['timepoint','technology','load_zone'])['power_mw'].sum()/1000
#     df_hourly_disp = df_hourly_disp.reset_index()
#     ###create pivot table
#     pt_hourly_disp = df_hourly_disp.pivot(index=['timepoint','load_zone'], columns='technology', values='power_mw')
#     ###calculate load by timepoints
#     load_bal = pd.read_csv(respath1+scs1[i]+'/results/load_balance.csv')
#     load_bal = load_bal.groupby(['timepoint','zone'])['load_mw'].sum()/1000
#     pt_hourly_disp['load'] = load_bal
#     ###calculate curtailment by timepoints
#     curt_df = pd.read_csv(respath1+scs1[i]+'/results/dispatch_variable.csv')
#     curt_df = curt_df.groupby(['timepoint','load_zone'])['total_curtailment_mw'].sum()/1000
#     pt_hourly_disp['curt'] = curt_df
#     ###include scenario name and append to list
#     pt_hourly_disp['scs'] = scs1[i]
#     list_hourly_disp_load1.append(pt_hourly_disp)
#     print(i)

# ### Concat pivot tables of all scenarios
# df_hourly_disp_load_allscs = pd.concat([list_hourly_disp_load[i] for i in range(len(scs))])
# df_hourly_disp_load_allscs = df_hourly_disp_load_allscs.fillna(0)

# df_hourly_disp_load_allscs1 = pd.concat([list_hourly_disp_load1[i] for i in range(len(scs1))])
# df_hourly_disp_load_allscs1 = df_hourly_disp_load_allscs1.fillna(0)

# df_hourly_disp_load_allscs.to_csv(data_path+'hourly_dispatch_2.csv')
# df_hourly_disp_load_allscs1.to_csv(data_path+'hourly_dispatch_1.csv')

# df_disp1 = pd.read_csv(data_path+'hourly_dispatch_1.csv')
# df_disp2 = pd.read_csv(data_path+'hourly_dispatch_2.csv')

# df_dispatch = pd.concat([df_disp1,df_disp2])
# df_dispatch.to_csv(datapath+'hourly_dispatch_all80.csv',index=False)

# # Curtailment-VRE 2020-2050
# vre_curt = pd.DataFrame()
# for i in range(len(scs80)):
#     if i < 40:
#         path = respath
#     else:
#         path = respath1
#     disp_re = pd.read_csv(path+scs80[i]+'/results/dispatch_variable.csv')
#     disp_re['twh'] = disp_re['total_curtailment_mw']*disp_re['timepoint_weight']/(10**6)
#     curt_re = disp_re.groupby(['period','load_zone'])['twh'].sum()
#     curt_re = curt_re.reset_index()
#     curt_re['technology']='curt_vre'
#     curt_re['scs'] = scs80[i]
#     vre_curt = pd.concat([vre_curt,curt_re])
#     print(str(i))
# vre_curt.to_csv(datapath + 'vre_curtailment.csv', index=False)


#%% TRADE DATA

# list_exim_lz_allscs = []
# list_tot_exp_imp_allscs = []
# for i in range(len(scs80)):
#     if i < 40:
#         path = respath
#     else:
#         path = respath1
#     ## estimate total export/import where exports include transmission losses
#     exim = pd.read_csv(path+scs80[i]+'/results/imports_exports.csv')
#     exim['imports_twh'] = exim['imports_mw'] * exim['timepoint_weight']/(10**6)
#     exim['exports_twh'] = exim['exports_mw'] * exim['timepoint_weight']/(10**6)
#     import_lz_as_neg_exp = exim[(exim.exports_twh < 0)].groupby(['period','load_zone'])['exports_twh'].sum()*(-1)
#     import_lz_as_pos_imp = exim[(exim.imports_twh > 0)].groupby(['period','load_zone'])['imports_twh'].sum()
#     export_lz_as_pos_exp = exim[(exim.exports_twh > 0)].groupby(['period','load_zone'])['exports_twh'].sum()
#     export_lz_as_neg_imp = exim[(exim.imports_twh < 0)].groupby(['period','load_zone'])['imports_twh'].sum()*(-1)
#     df_exim_lz=pd.concat([import_lz_as_neg_exp,import_lz_as_pos_imp,export_lz_as_pos_exp, export_lz_as_neg_imp],axis=1)
#     df_exim_lz.columns = ['import_lz_as_neg_exp','import_lz_as_pos_imp','export_lz_as_pos_exp', 'export_lz_as_neg_imp']
#     df_exim_lz = df_exim_lz.fillna(0)
#     df_exim_lz['tot_imp_twh'] = df_exim_lz[['import_lz_as_neg_exp','import_lz_as_pos_imp']].sum(axis=1)
#     df_exim_lz['tot_exp_twh'] = df_exim_lz[['export_lz_as_pos_exp', 'export_lz_as_neg_imp']].sum(axis=1)
#     df_exim_lz['net_imp_twh'] = df_exim_lz['tot_imp_twh'] - df_exim_lz['tot_exp_twh'] ##matches with net_exim below
#     df_exim_lz['tot_trade_twh'] = df_exim_lz['tot_imp_twh'] + df_exim_lz['tot_exp_twh']
#     df_exim_lz['Scenario'] = scs_tick[i]
#     ## estimate transmission losses and exclude from exports   
#     trans_op = pd.read_csv(path+scs80[i]+'/results/transmission_operations.csv')
#     trans_op['tx_loss_from_twh'] = trans_op['transmission_losses_lz_from']*trans_op['timepoint_weight']/(10**6)
#     trans_op['tx_loss_to_twh'] = trans_op['transmission_losses_lz_to']*trans_op['timepoint_weight']/(10**6)
#     from_to_losses = trans_op.groupby(['period', 'lz_from'])['tx_loss_to_twh'].sum()
#     to_from_losses = trans_op.groupby(['period', 'lz_to'])['tx_loss_from_twh'].sum()
#     df_exim_lz_with_losses = pd.concat([df_exim_lz, from_to_losses, to_from_losses],axis=1).fillna(0)
#     df_exim_lz_with_losses['tot_loss'] = df_exim_lz_with_losses[['tx_loss_to_twh','tx_loss_from_twh']].sum(axis=1)
#     df_exim_lz_with_losses['tot_exp_twh_excl_loss'] = df_exim_lz_with_losses['tot_exp_twh'] - df_exim_lz_with_losses['tot_loss']
#     df_exim_lz_with_losses = df_exim_lz_with_losses.reset_index()
#     ## separate columns with final export and import data
#     df_exim_lz1 = df_exim_lz_with_losses[['period','level_1','tot_imp_twh',\
#                                           'tot_exp_twh_excl_loss','net_imp_twh','Scenario']].copy()
#     df_exim_lz1.columns = ['period','load_zone','tot_imp_twh','tot_exp_twh_excl_loss','net_imp_twh','Scenario']
#     list_exim_lz_allscs.append(df_exim_lz1)
#     ## total export import for px plot
#     df_imp = df_exim_lz1[['period','load_zone']].copy()
#     df_exp = df_exim_lz1[['period','load_zone']].copy()
#     df_imp['trade'] = df_exim_lz1['tot_imp_twh']
#     df_imp['type'] = 'import'
#     df_exp['trade'] = df_exim_lz1['tot_exp_twh_excl_loss']
#     df_exp['type'] = np.where(df_exim_lz1["tot_exp_twh_excl_loss"] < 0, 'import', 'export')
#     df_exp["trade"] = df_exp["trade"].abs()
#     df_tot_exp_imp = pd.concat([df_imp, df_exp])
#     df_tot_exp_imp = df_tot_exp_imp.groupby(['type', 'period','load_zone'])['trade'].sum().reset_index()
#     df_tot_exp_imp['Scenario'] = scs_tick[i]
#     list_tot_exp_imp_allscs.append(df_tot_exp_imp)
#     print(str(i))
    
# ## concat all scenarios
# df_exim_lz_allscs = pd.concat([list_exim_lz_allscs[i] for i in range(len(scs80))])
# df_tot_exp_imp_allscs = pd.concat([list_tot_exp_imp_allscs[i] for i in range(len(scs80))]).reset_index().drop(columns='index')
# df_tot_exp_imp_allscs['multiplier'] = np.where(df_tot_exp_imp_allscs["type"] == 'import', 1, -1)
# df_tot_exp_imp_allscs['trade_1'] = df_tot_exp_imp_allscs['trade'].multiply(df_tot_exp_imp_allscs['multiplier'])
# df_tot_exp_imp_allscs.round(decimals=5).to_csv(datapath + 'all_trade_tot_exim_by_lzyr.csv', index=False)

# # country-level trade
# trade_list = []
# for i in range(len(scs80)):
#     if i < 40:
#         path = respath
#     else:
#         path = respath1
#     trans_op = pd.read_csv(path+scs80[i]+'/results/transmission_operations.csv')
#     trans_op['tx_flow_twh'] = trans_op['transmission_flow_mw']*trans_op['timepoint_weight']/(10**6)
#     trans_op['tx_loss_from_twh'] = trans_op['transmission_losses_lz_from']*trans_op['timepoint_weight']/(10**6)
#     trans_op['tx_loss_to_twh'] = trans_op['transmission_losses_lz_to']*trans_op['timepoint_weight']/(10**6)
#     trans_op['net_flow_twh'] = trans_op['tx_flow_twh'] + trans_op['tx_loss_from_twh'] - trans_op['tx_loss_to_twh']
#     trans_op['net_flow_twh'] = np.abs(trans_op['net_flow_twh'])
#     grp_trans = trans_op.groupby(['lz_from','lz_to','period']).sum().reset_index()\
#                         .drop(columns=['timepoint','timepoint_weight','number_of_hours_in_timepoint'])
#     grp_trans['scenario'] = scs80[i]
#     trade_list.append(grp_trans[['lz_from','lz_to','period','net_flow_twh','scenario']])
    
# df_trade_list = pd.concat([trade_list[i] for i in range(len(scs80))])
# df_trade_list.to_csv(datapath + 'total_trade_data.csv', index=False)

#%% TRADE BY TIMEPOINT

# list_exim_lz_allscs = []
# list_tot_exp_imp_allscs = []
# for i in range(79,len(scs80)):
#     if i < 40:
#         path = respath
#     else:
#         path = respath1
#     ## estimate total export/import where exports include transmission losses
#     exim = pd.read_csv(path+scs80[i]+'/results/imports_exports.csv')
#     exim['imports_twh'] = exim['imports_mw']/(10**6)
#     exim['exports_twh'] = exim['exports_mw']/(10**6)
#     import_lz_as_neg_exp = exim[(exim.exports_twh < 0)].groupby(['period','load_zone'])['exports_twh'].sum()*(-1)
#     import_lz_as_pos_imp = exim[(exim.imports_twh > 0)].groupby(['period','load_zone'])['imports_twh'].sum()
#     export_lz_as_pos_exp = exim[(exim.exports_twh > 0)].groupby(['period','load_zone'])['exports_twh'].sum()
#     export_lz_as_neg_imp = exim[(exim.imports_twh < 0)].groupby(['period','load_zone'])['imports_twh'].sum()*(-1)
#     df_exim_lz=pd.concat([import_lz_as_neg_exp,import_lz_as_pos_imp,export_lz_as_pos_exp, export_lz_as_neg_imp],axis=1)
#     df_exim_lz.columns = ['import_lz_as_neg_exp','import_lz_as_pos_imp','export_lz_as_pos_exp', 'export_lz_as_neg_imp']
#     df_exim_lz = df_exim_lz.fillna(0)
#     df_exim_lz['tot_imp_twh'] = df_exim_lz[['import_lz_as_neg_exp','import_lz_as_pos_imp']].sum(axis=1)
#     df_exim_lz['tot_exp_twh'] = df_exim_lz[['export_lz_as_pos_exp', 'export_lz_as_neg_imp']].sum(axis=1)
#     df_exim_lz['net_imp_twh'] = df_exim_lz['tot_imp_twh'] - df_exim_lz['tot_exp_twh'] ##matches with net_exim below
#     df_exim_lz['tot_trade_twh'] = df_exim_lz['tot_imp_twh'] + df_exim_lz['tot_exp_twh']
#     df_exim_lz['Scenario'] = scs_tick[i]
#     ## estimate transmission losses and exclude from exports   
#     trans_op = pd.read_csv(path+scs80[i]+'/results/transmission_operations.csv')
#     trans_op['tx_loss_from_twh'] = trans_op['transmission_losses_lz_from']/(10**6)
#     trans_op['tx_loss_to_twh'] = trans_op['transmission_losses_lz_to']/(10**6)
#     from_to_losses = trans_op.groupby(['period', 'lz_from'])['tx_loss_to_twh'].sum()
#     to_from_losses = trans_op.groupby(['period', 'lz_to'])['tx_loss_from_twh'].sum()
#     df_exim_lz_with_losses = pd.concat([df_exim_lz, from_to_losses, to_from_losses],axis=1).fillna(0)
#     df_exim_lz_with_losses['tot_loss'] = df_exim_lz_with_losses[['tx_loss_to_twh','tx_loss_from_twh']].sum(axis=1)
#     df_exim_lz_with_losses['tot_exp_twh_excl_loss'] = df_exim_lz_with_losses['tot_exp_twh'] - df_exim_lz_with_losses['tot_loss']
#     df_exim_lz_with_losses = df_exim_lz_with_losses.reset_index()
#     ## separate columns with final export and import data
#     df_exim_lz1 = df_exim_lz_with_losses[['period','level_1','tot_imp_twh',\
#                                           'tot_exp_twh_excl_loss','net_imp_twh','Scenario']].copy()
#     df_exim_lz1.columns = ['period','load_zone','tot_imp_twh','tot_exp_twh_excl_loss','net_imp_twh','Scenario']
#     list_exim_lz_allscs.append(df_exim_lz1)
#     ## total export import for px plot
#     df_imp = df_exim_lz1[['period','load_zone']].copy()
#     df_exp = df_exim_lz1[['period','load_zone']].copy()
#     df_imp['trade'] = df_exim_lz1['tot_imp_twh']
#     df_imp['type'] = 'import'
#     df_exp['trade'] = df_exim_lz1['tot_exp_twh_excl_loss']
#     df_exp['type'] = np.where(df_exim_lz1["tot_exp_twh_excl_loss"] < 0, 'import', 'export')
#     df_exp["trade"] = df_exp["trade"].abs()
#     df_tot_exp_imp = pd.concat([df_imp, df_exp])
#     df_tot_exp_imp = df_tot_exp_imp.groupby(['type', 'period','load_zone'])['trade'].sum().reset_index()
#     df_tot_exp_imp['Scenario'] = scs_tick[i]
#     list_tot_exp_imp_allscs.append(df_tot_exp_imp)
#     print(str(i))
    
# ## concat all scenarios
# df_exim_lz_allscs = pd.concat([list_exim_lz_allscs[i] for i in range(len(scs80))])
# df_tot_exp_imp_allscs = pd.concat([list_tot_exp_imp_allscs[i] for i in range(len(scs80))]).reset_index().drop(columns='index')
# df_tot_exp_imp_allscs['multiplier'] = np.where(df_tot_exp_imp_allscs["type"] == 'import', 1, -1)
# df_tot_exp_imp_allscs['trade_1'] = df_tot_exp_imp_allscs['trade'].multiply(df_tot_exp_imp_allscs['multiplier'])
# #df_tot_exp_imp_allscs.round(decimals=5).to_csv(datapath + 'all_trade_tot_exim_by_lzyr.csv', index=False)





# # country-level trade
# trade_list = []
# for i in range(len(scs80)):
#     if i < 40:
#         path = respath
#     else:
#         path = respath1
#     trans_op = pd.read_csv(path+scs80[i]+'/results/transmission_operations.csv')
#     trans_op['tx_flow_twh'] = trans_op['transmission_flow_mw']/(10**6)
#     trans_op['tx_loss_from_twh'] = trans_op['transmission_losses_lz_from']/(10**6)
#     trans_op['tx_loss_to_twh'] = trans_op['transmission_losses_lz_to']/(10**6)
#     trans_op['net_flow_twh'] = trans_op['tx_flow_twh'] + trans_op['tx_loss_from_twh'] - trans_op['tx_loss_to_twh']
#     trans_op['tx_line'] = trans_op['tx_line'].map({'Brazil_Uruguay_New':'Brazil_Uruguay','Brazil_Paraguay_New':'Brazil_Paraguay',
#                                                    'Argentina_Uruguay_New':'Argentina_Uruguay','Argentina_Paraguay_New':'Argentina_Paraguay',
#                                                    'Argentina_Chile_New':'Argentina_Chile','Argentina_Brazil_New':'Argentina_Brazil',
#                                                    'Brazil_Uruguay':'Brazil_Uruguay','Brazil_Paraguay':'Brazil_Paraguay',
#                                                    'Argentina_Uruguay':'Argentina_Uruguay','Argentina_Paraguay':'Argentina_Paraguay',
#                                                    'Argentina_Chile':'Argentina_Chile','Argentina_Brazil':'Argentina_Brazil'})
#     #trans_op['net_flow_twh'] = np.abs(trans_op['net_flow_twh'])
#     grp_trans = trans_op.groupby(['timepoint','tx_line','lz_from','lz_to','period']).sum().reset_index()\
#                         .drop(columns=['timepoint_weight','number_of_hours_in_timepoint'])
#     grp_trans['scenario'] = scs80[i]
#     trade_list.append(grp_trans[['timepoint','tx_line','lz_from','lz_to','period','net_flow_twh','scenario']])
#     print(str(i))
    
# df_trade_list = pd.concat([trade_list[i] for i in range(len(scs80))])
# df_trade_list.to_csv(datapath + 'total_trade_by_timepoint_net.csv', index=False)


#%%





