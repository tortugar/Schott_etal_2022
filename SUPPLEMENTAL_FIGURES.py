#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 18:30:46 2021

@author: fearthekraken
"""
import AS
import pwaves
import sleepy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import MultiComparison

#%%
###   Supp. FIGURE 1D - FISH quantification   ###
df = pd.read_csv('/home/fearthekraken/Documents/Data/sleepRec_processed/FISH_counts.csv')
plt.figure()
sns.boxplot(x='MARKER LABEL', y='%CRH + MARKER', order=['VGLUT1','VGLUT2','GAD2'], data=df, whis=np.inf, color='white', fliersize=0)
sns.stripplot(x='MARKER LABEL', y='%CRH + MARKER', hue='Mouse', order=['VGLUT1','VGLUT2','GAD2'], data=df, 
              palette={'Marlin':'lightgreen', 'SERT1':'lightblue', 'Nemo':'lightgray'}, size=10, linewidth=1, edgecolor='black')
plt.show()
print('')
for marker_label in ['VGLUT1', 'VGLUT2', 'GAD2']:
    p = df['%CRH + MARKER'].iloc[np.where(df['MARKER LABEL']==marker_label)[0]]
    print(f'{round(p.mean(),2)}% of CRH+ neurons co-express {marker_label} (+/-{round(p.std(),2)}%)')

#%%
###   Supp. FIGURE 2B - time-normalized DF/F activity across brain state transitions   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'crh_photometry.txt')[1]
sequence=[3,4,1,2]; state_thres=[(0,10000)]*len(sequence); nstates=[20,20,20,20]; vm=[0.2, 1.9]  # NREM --> IS --> REM --> WAKE

_, mx_pwave, _ = pwaves.stateseq(ppath, recordings, sequence=sequence, nstates=nstates, state_thres=state_thres, ma_thr=20, ma_state=3, 
                                       flatten_tnrem=4, fmax=25, pnorm=1, vm=vm, psmooth=[2,2], mode='dff', mouse_avg='mouse', print_stats=False)

#%%
###   Supp. FIGURE 2C,D,E - DF/F activity at brain state transitions   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'crh_photometry.txt')[1]
transitions = [(3,4)]; pre=40; post=15; vm=[0.3, 1.9]; tr_label = 'NtN'  # NREM --> IS
#transitions = [(4,1)]; pre=15; post=40; vm=[0.1, 2.1]; tr_label = 'tNR'  # IS --> REM
#transitions = [(1,2)]; pre=40; post=15; vm=[0.1, 2.1]; tr_label = 'RW'   # REM --> WAKE

si_threshold = [pre]*6; sj_threshold = [post]*6
mice, tr_act, tr_spe = pwaves.activity_transitions(ppath, recordings, transitions=transitions, pre=pre, post=post, si_threshold=si_threshold, 
                                                   sj_threshold=sj_threshold, ma_thr=20, ma_state=3, flatten_tnrem=4, vm=vm, fmax=25, pnorm=1, 
                                                   psmooth=[3,3], mode='dff', mouse_avg='trials', base_int=5, print_stats=True)

#%%
###   Supp. FIGURE 2F - DF/F activity following single & cluster P-waves   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'pwaves_photometry.txt')[1]

# get DF/F timecourse data, store in dataframe
pzscore=[0,0,0]; p_iso=0.8; pcluster=0
mice, iso_mx = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='', dff_win=[0,2], pzscore=pzscore, mouse_avg='mouse',  # single P-waves
                                     p_iso=p_iso, pcluster=pcluster, clus_event='waves', psmooth=(8,15), print_stats=False)
pzscore=[0,0,0]; p_iso=0; pcluster=0.5
mice, clus_mx = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='', dff_win=[0,2], pzscore=pzscore, mouse_avg='mouse',  # clustered P-waves
                                     p_iso=p_iso, pcluster=pcluster, clus_event='waves', psmooth=(8,15), print_stats=False)
df = pd.DataFrame({'Mouse' : np.tile(mice,2),
                   'Event' : np.repeat(['single', 'cluster'], len(mice)),
                   'DFF' : np.concatenate((iso_mx[2].mean(axis=1), clus_mx[2].mean(axis=1))) })

# bar plot
plt.figure(); sns.barplot(x='Event', y='DFF', data=df, ci=68, palette={'single':'salmon', 'cluster':'mediumslateblue'})
sns.pointplot(x='Event', y='DFF', hue='Mouse', data=df, ci=None, markers='', color='black'); plt.gca().get_legend().remove(); plt.show()

# stats
p = stats.ttest_rel(df['DFF'].iloc[np.where(df['Event'] == 'single')[0]], df['DFF'].iloc[np.where(df['Event'] == 'cluster')[0]])
print(f'single vs cluster P-waves -- T={round(p.statistic,3)}, p-value={round(p.pvalue,5)}')

#%%
###   Supp. FIGURE 2G -averaged DF/F surrounding P-waves in each brain state  ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'pwaves_photometry.txt')[1]
for s in [1,2,3,4]:
    pwaves.dff_timecourse(ppath, recordings, istate=s, dff_win=[2,2], plotMode='03', pzscore=[0,0,0], mouse_avg='mouse', 
                          ma_thr=20, ma_state=3, flatten_tnrem=4, p_iso=0, pcluster=0)

#%%
###   Supp. FIGURE 3B,C,D - P-wave frequency at brain state transitions   ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]
transitions = [(3,4)]; pre=40; post=15; vm=[0.4,2.0]; tr_label = 'NtN'  # NREM --> IS
#transitions = [(4,1)]; pre=15; post=40; vm=[0.1, 2.0]; tr_label = 'tNR'  # IS --> REM
#transitions = [(1,2)]; pre=40; post=15; vm=[0.1, 2.0]; tr_label = 'RW'   # REM --> WAKE

si_threshold = [pre]*6; sj_threshold = [post]*6
mice, tr_act, tr_spe = pwaves.activity_transitions(ppath, recordings, transitions=transitions, pre=pre, post=post, si_threshold=si_threshold, 
                                                   sj_threshold=sj_threshold, ma_thr=20, ma_state=3, flatten_tnrem=4, vm=vm, fmax=25, pnorm=1, 
                                                   psmooth=[3,3], mode='pwaves', mouse_avg='trials', base_int=5, print_stats=True)

#%%
###   Supp. FIGURE 3E - time-normalized frequency of single & clustered P-waves across brain state transitions   ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]
sequence=[3,4,1,2]; state_thres=[(0,10000)]*len(sequence); nstates=[20,20,20,20]; vm=[0.2,2.0]  # NREM --> IS --> REM --> WAKE

mice,smx,sspe = pwaves.stateseq(ppath, recordings, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25,   # single P-waves
                                 pnorm=1, vm=vm, psmooth=[2,2], mode='pwaves', mouse_avg='mouse', p_iso=0.8, pcluster=0, 
                                 clus_event='waves', pplot=False, print_stats=False)
mice,cmx,cspe = pwaves.stateseq(ppath, recordings, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25,   # clustered P-waves
                                 pnorm=1, vm=vm, psmooth=[2,2], mode='pwaves', mouse_avg='mouse', p_iso=0, pcluster=0.5, 
                                 clus_event='waves', pplot=False, print_stats=False)
# plot timecourses
pwaves.plot_activity_transitions([smx, cmx], [mice, mice], plot_id=['salmon', 'mediumslateblue'], group_labels=['single', 'cluster'], 
                                 xlim=nstates, xlabel='Time (normalized)', ylabel='P-waves/s', title='NREM-->tNREM-->REM-->Wake')


#%%
###   Supp. FIGURE 3F - average single & cluster P-wave frequency in each brain state   ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]

mice,x,sfreq,swf = pwaves.state_freq(ppath, recordings, istate=[1,2,3,4], p_iso=0.8, pcluster=0,  # single P-waves
                                      clus_event='waves', pplot=False, print_stats=False)
mice,x,cfreq,cwf = pwaves.state_freq(ppath, recordings, istate=[1,2,3,4], p_iso=0, pcluster=0.5,  # clustered P-waves
                                      clus_event='waves', pplot=False, print_stats=False)
# bar plot
pwaves.plot_state_freq(x, [mice,mice], [sfreq,cfreq], [swf,cwf], group_colors=['salmon', 'mediumslateblue'], group_labels=['single','cluster'], 
                       legend='groups', title='Avg. P-wave frequency - single vs clustered waves')

# stats
df = pd.DataFrame(columns=['Mouse', 'State', 'Event', 'Freq'])
for i,s in enumerate(['REM', 'Wake', 'NREM', 'IS']):
    df = df.append(pd.DataFrame({'Mouse' : np.tile(mice,2),
                                 'State' : [s]*len(mice)*2,
                                 'Event' : np.repeat(['single', 'cluster'], len(mice)),
                                 'Freq' : np.concatenate((sfreq[:,i], cfreq[:,i])) }))
# two-way repeated measures ANOVA
res_anova = AnovaRM(data=df, depvar='Freq', subject='Mouse', within=['Event', 'State']).fit()
print(res_anova); print('   ###   P-values   ###'); print(res_anova.anova_table['Pr > F'])
# post hocs - single and cluster P-wave frequency compared between each pair of brain states
single_df = df.iloc[np.where(df['Event'] == 'single')[0], :]; clus_df = df.iloc[np.where(df['Event'] == 'cluster')[0], :]
mc_single = MultiComparison(single_df['Freq'], single_df['State']).allpairtest(stats.ttest_rel, method='bonf'); mc_clus = MultiComparison(clus_df['Freq'], clus_df['State']).allpairtest(stats.ttest_rel, method='bonf')
print('\nSingle P-waves\n'); print(mc_single[0]); print('\nCluster P-waves\n'); print(mc_clus[0]);
# post hocs - for each brain state, single compared to cluster P-wave frequency
print('\nSingle vs cluster P-waves\n')
for s in ['REM', 'Wake', 'NREM', 'IS']:
    p = stats.ttest_rel(single_df['Freq'].iloc[np.where(single_df['State'] == s)[0]], clus_df['Freq'].iloc[np.where(clus_df['State'] == s)[0]])
    print(f'{s}  --  T={round(p.statistic,3)}, p-value={round(p.pvalue,5)}, sig={"yes" if p.pvalue < 0.05 else "no"}')

#%%
###   Supp. FIGURE 3G - average spectral power surrounding single & cluster P-waves   ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]

# top - averaged spectrograms
filename = 'sp_win3_single'; win=[-3,3]; pnorm=2; p_iso=0.8; pcluster=0
pwaves.avg_SP(ppath, recordings, istate=[1], win=win, mouse_avg='mouse', plaser=False, pnorm=pnorm, psmooth=[2,2],  # single P-waves
              fmax=25, vm=[0.6,2.0], p_iso=p_iso, pcluster=pcluster, clus_event='waves', pload=filename, psave=filename)
filename = 'sp_win3_cluster'; win=[-3,3]; pnorm=2; p_iso=0; pcluster=0.5
pwaves.avg_SP(ppath, recordings, istate=[1], win=win, mouse_avg='mouse', plaser=False, pnorm=pnorm, psmooth=[2,2],  # clustered P-waves
              fmax=25, vm=[0.6,2.0], p_iso=p_iso, pcluster=pcluster, clus_event='waves', pload=filename, psave=filename)

# bottom - average high theta power
filename = 'sp_win3_single'; win=[-3,3]; pnorm=2; p_iso=0.8; pcluster=0
mice, sdict, t = pwaves.avg_band_power(ppath, recordings, istate=[1], win=win, mouse_avg='mouse', plaser=False, pnorm=pnorm,  # single P-waves
                                       psmooth=0, bands=[(8,15)], band_colors=['green'], p_iso=p_iso, pcluster=pcluster, 
                                       clus_event='waves', ylim=[0.6,1.8], pload=filename, psave=filename)
filename = 'sp_win3_cluster'; win=[-3,3]; pnorm=2; p_iso=0; pcluster=0.5
mice, cdict, t = pwaves.avg_band_power(ppath, recordings, istate=[1], win=win, mouse_avg='mouse', plaser=False, pnorm=pnorm,  # clustered P-waves
                                       psmooth=0, bands=[(8,15)], band_colors=['green'], p_iso=p_iso, pcluster=pcluster, 
                                       clus_event='waves', ylim=[0.6,1.8], pload=filename, psave=filename)

# right - mean power in 1 s time window
x = np.intersect1d(np.where(t>=-0.5)[0], np.where(t<=0.5)[0])  # get columns between -0.5 s and +0.5 s
df = pd.DataFrame({'Mouse' : np.tile(mice,2),
                   'Event' : np.repeat(['single', 'cluster'], len(mice)),
                   'Pwr'   : np.concatenate((sdict[(8,15)][:,x].mean(axis=1), cdict[(8,15)][:,x].mean(axis=1))) })
fig = plt.figure(); sns.barplot(x='Event', y='Pwr', data=df, ci=68, palette={'single':'salmon', 'cluster':'mediumslateblue'})
sns.pointplot(x='Event', y='Pwr', hue='Mouse', data=df, ci=None, markers='', color='black'); plt.gca().get_legend().remove()
plt.title('Single vs Clustered P-waves'); plt.show()

# stats
p = stats.ttest_rel(df['Pwr'].iloc[np.where(df['Event'] == 'single')[0]], df['Pwr'].iloc[np.where(df['Event'] == 'cluster')[0]])
print(f'single vs cluster P-waves -- T={round(p.statistic,3)}, p-value={round(p.pvalue,5)}')

#%%
###  Supp. FIGURE 4A - % time in each brain state before and during the laser   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_chr2_ol.txt')[1]

BS, t, df = AS.laser_brainstate(ppath, recordings, pre=400, post=520, flatten_tnrem=4, ma_state=3, ma_thr=20, edge=10, sf=0, ci='sem')

#%%
###   Supp. FIGURE 4B- averaged spectral band power before and during the laser   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_chr2_ol.txt')[1]

bands=[(0.5,4), (6,10), (11,15), (55,99)]; band_labels=['delta', 'theta', 'sigma', 'gamma']; band_colors=['firebrick', 'limegreen', 'cyan', 'purple']
AS.laser_triggered_eeg_avg(ppath, recordings, pre=400, post=520, fmax=100, laser_dur=120, pnorm=1, psmooth=3, harmcs=10, 
                           iplt_level=2, vm=[0.6,1.4], sf=7, bands=bands, band_labels=band_labels, band_colors=band_colors, ci=95)

#%%
###   Supp. FIGURE 4C - laser-triggered change in REM transition probability   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_chr2_ol.txt')[1]

AS.laser_transition_probability(ppath, recordings, pre=400, post=520, ma_state=3, ma_thr=20, sf=10)

#%%
###   Supp. FIGURE 4D, spectral power during NREM-->REM transitions   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_chr2_ol.txt')[1]
pre=40; post=40; si_threshold=[pre]*6; sj_threshold=[post]*6
bands=[(0.5,4), (6,12), (13,20), (50,100)]; band_labels=['delta', 'theta', 'sigma', 'gamma']; band_colors=['firebrick', 'limegreen', 'cyan', 'purple']

AS.avg_sp_transitions(ppath, recordings, transitions=[(3,1)], pre=pre, post=post, si_threshold=si_threshold, sj_threshold=sj_threshold, 
                      laser=1, bands=bands, band_labels=band_labels, band_colors=band_colors, flatten_tnrem=3, ma_thr=20, ma_state=3, 
                      fmax=100, pnorm=1, psmooth=[3,3], vm=[(0.1,2.5),(0.1,2.5)], mouse_avg='mouse', sf=0)

#%%
###   Supp. FIGURE 4E - power spectrum for each brain state, ChR2 & eYFP mice   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_chr2_ol.txt')[1]

_ = AS.sleep_spectrum_simple(ppath, recordings, istate=[1,2,3,4], pmode=1, pnorm=0, fmax=30, ma_thr=20,  # ChR2
                             ma_state=3, flatten_tnrem=4, harmcs=10)
recordings = sleepy.load_recordings(ppath, 'crh_yfp_chr2_ol.txt')[1]
_ = AS.sleep_spectrum_simple(ppath, recordings, istate=[1,2,3,4], pmode=1, pnorm=0, fmax=30, ma_thr=20,  # eYFP
                             ma_state=3, flatten_tnrem=4, harmcs=10)

#%%
###   Supp. FIGURE 4G,H - eYFP percent time spent in each brain state surrounding laser   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_yfp_chr2_ol.txt')[1]

BS, t, df = AS.laser_brainstate(ppath, recordings, pre=400, post=520, flatten_tnrem=4, ma_state=3, ma_thr=20, edge=10, sf=0, ci='sem', ylim=[0,80])

#%%
###   Supp. FIGURE 4I - eYFP averaged SPs and frequency band power surrounding laser   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_yfp_chr2_ol.txt')[1]
bands=[(0.5,4), (6,10), (11,15), (55,99)]; band_labels=['delta', 'theta', 'sigma', 'gamma']; band_colors=['firebrick', 'limegreen', 'cyan', 'purple']

AS.laser_triggered_eeg_avg(ppath, recordings, pre=400, post=520, fmax=100, laser_dur=120, pnorm=1, psmooth=3, harmcs=10, iplt_level=2,
                           vm=[0.6,1.4], sf=7, bands=bands, band_labels=band_labels, band_colors=band_colors, ci=95, ylim=[0.6,1.3])

#%%
###   Supp. FIGURE 4J - closed loop overall REM duration   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
ctr_rec = sleepy.load_recordings(ppath, 'crh_yfp_chr2_cl.txt')[1]; exp_rec = sleepy.load_recordings(ppath, 'crh_chr2_cl.txt')[1]
AS.compare_online_analysis(ppath, ctr_rec, exp_rec, istate=1, stat='dur', mouse_avg='mouse', group_colors=['gray','blue'], ylim=[0,120])  # eYFP vs ChR2

ctr_rec = sleepy.load_recordings(ppath, 'crh_yfp_ic_cl.txt')[1]; exp_rec = sleepy.load_recordings(ppath, 'crh_ic_cl.txt')[1]
AS.compare_online_analysis(ppath, ctr_rec, exp_rec, istate=1, stat='dur', mouse_avg='mouse', group_colors=['gray','red'], ylim=[0,120])  # eYFP vs iC++

#%%
###   Supp. FIGURE 4K - closed loop total % time in REM   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
ctr_rec = sleepy.load_recordings(ppath, 'crh_yfp_chr2_cl.txt')[1]; exp_rec = sleepy.load_recordings(ppath, 'crh_chr2_cl.txt')[1]
AS.compare_online_analysis(ppath, ctr_rec, exp_rec, istate=1, stat='perc', mouse_avg='mouse', group_colors=['gray','blue'], ylim=[0,12])  # eYFP vs ChR2

ctr_rec = sleepy.load_recordings(ppath, 'crh_yfp_ic_cl.txt')[1]; exp_rec = sleepy.load_recordings(ppath, 'crh_ic_cl.txt')[1]
AS.compare_online_analysis(ppath, ctr_rec, exp_rec, istate=1, stat='perc', mouse_avg='mouse', group_colors=['gray','red'], ylim=[0,12])  # eYFP vs iC++

#%%
###   Supp. FIGURE 5B - average amplitude and half-width of spontaneous & laser-triggered P-waves   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]
filename = 'lsr_stats' 
df = pwaves.get_lsr_stats(ppath, recordings, istate=[1,2,3,4], post_stim=0.1, flatten_tnrem=4, ma_thr=20, ma_state=3, psave=filename)

pwaves.lsr_pwave_size(df, stat='amp2', plotMode='03', istate=1, mouse_avg='mouse')
pwaves.lsr_pwave_size(df, stat='halfwidth', plotMode='03', istate=1, mouse_avg='mouse')

#%%
###   Supp. FIGURE 5C - average spectral power surrounding single & cluster P-waves   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]

# top - averaged spectrograms
filename = 'sp_win3_single_lsr'; win=[-3,3]; pnorm=2; p_iso=0.8; pcluster=0
pwaves.avg_SP(ppath, recordings, istate=[1], mode='pwaves', win=win, plaser=True, post_stim=0.1, mouse_avg='mouse',  # single lsr P-waves
              pnorm=pnorm, psmooth=[(3,3),(5,5)], vm=[(0.6,1.65),(0.8,1.5)], fmax=25, recalc_highres=False, 
              p_iso=p_iso, pcluster=pcluster, clus_event='waves', pload=filename, psave=filename)
filename = 'sp_win3_cluster_lsr'; win=[-3,3]; pnorm=2; p_iso=0; pcluster=0.5
pwaves.avg_SP(ppath, recordings, istate=[1], mode='pwaves', win=win, plaser=True, post_stim=0.1, mouse_avg='mouse',  # clustered lsr P-waves
              pnorm=pnorm, psmooth=[(7,7),(5,5)], vm=[(0.6,1.65),(0.8,1.5)], fmax=25, recalc_highres=False, 
              p_iso=p_iso, pcluster=pcluster, clus_event='waves', pload=filename, psave=filename)

# bottom - averaged high theta power
filename = 'sp_win3_single_lsr'; win=[-3,3]; pnorm=2; p_iso=0.8; pcluster=0
mice,lsr_iso,spon_iso,t = pwaves.avg_band_power(ppath, recordings, istate=[1], mode='pwaves', win=win, plaser=True, post_stim=0.1,  # single ls P-waves
                                        mouse_avg='mouse', bands=[(8,15)], band_colors=[('green')], pnorm=pnorm, psmooth=(4,4), 
                                        fmax=25, p_iso=p_iso, pcluster=pcluster, clus_event='waves', pload=filename, psave=filename, ylim=[0.5,2])
filename = 'sp_win3_cluster_lsr'; win=[-3,3]; pnorm=2; p_iso=0; pcluster=0.5
mice,lsr_clus,spon_clus,t = pwaves.avg_band_power(ppath, recordings, istate=[1], mode='pwaves', win=win, plaser=True, post_stim=0.1,  # clustered lsr P-waves
                                        mouse_avg='mouse', bands=[(8,15)], band_colors=[('green')], pnorm=pnorm, psmooth=(4,4), 
                                        fmax=25, p_iso=p_iso, pcluster=pcluster, clus_event='waves', pload=filename, psave=filename, ylim=[0.5,2])

# right - mean power in 1 s time window
x = np.intersect1d(np.where(t>=-0.5)[0], np.where(t<=0.5)[0])  # get columns between -0.5 s and +0.5 s
df = pd.DataFrame({'Mouse' : np.tile(mice,2),
                   'Event' : np.repeat(['single', 'cluster'], len(mice)),
                   'Pwr'   : np.concatenate((lsr_iso[(8,15)][:,x].mean(axis=1), lsr_clus[(8,15)][:,x].mean(axis=1))) })
fig = plt.figure(); sns.barplot(x='Event', y='Pwr', data=df, ci=68, palette={'single':'salmon', 'cluster':'mediumslateblue'})
sns.pointplot(x='Event', y='Pwr', hue='Mouse', data=df, ci=None, markers='', color='black'); plt.gca().get_legend().remove()
plt.title('Laser-triggered Single vs Clustered P-waves'); plt.show()

# stats
p = stats.ttest_rel(df['Pwr'].iloc[np.where(df['Event'] == 'single')[0]], df['Pwr'].iloc[np.where(df['Event'] == 'cluster')[0]])
print(f'single vs cluster laser P-waves -- T={round(p.statistic,3)}, p-value={round(p.pvalue,5)}')

#%%
###   Supp. FIGURE 5D,E,F - spectral power preceding successful & failed laser pulses   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]

# D - normalized power spectrum
filename = 'sp_win3pre_pnorm1'; win=[-3,0]; pnorm=1
pwaves.lsr_prev_theta_success(ppath, recordings, win=win, mode='spectrum', theta_band=[0,20], post_stim=0.1, pnorm=pnorm, psmooth=3, 
                              ci='sem', nbins=14, prange1=(), prange2=(), mouse_avg='trials', pload=filename, psave=filename)
# E - mean theta power
filename = 'sp_win3pre_pnorm1'; win=[-3,0]; pnorm=1
pwaves.lsr_prev_theta_success(ppath, recordings, win=win, mode='power', theta_band=[6,12], post_stim=0.1, pnorm=pnorm, psmooth=0, 
                              ci='sem', nbins=14, prange1=(), prange2=(0,4), mouse_avg='trials', pload=filename, psave=filename)
# F - mean theta frequency
filename = 'sp_win3pre_pnorm0'; win=[-3,0]; pnorm=0
pwaves.lsr_prev_theta_success(ppath, recordings, win=win, mode='mean freq', theta_band=[6,12], post_stim=0.1, pnorm=pnorm, psmooth=0, 
                              ci='sem', nbins=14, prange1=(), prange2=(6.5,9.5), mouse_avg='trials', pload=filename, psave=filename)

#%%
###   Supp. FIGURE 6A - hm3dq power spectrums, saline vs cno   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(c, e) = AS.load_recordings(ppath, 'crh_hm3dq_tnrem.txt', dose=True, pwave_channel=False); e=e['0.25']

AS.compare_power_spectrums(ppath, [c, e], ['hm3dq-saline', 'hm3dq-cno'], istate=[1,2,3,4], pmode=0, pnorm=0, 
                           fmax=30, flatten_tnrem=4, ma_thr=20, ma_state=3, colors=['gray', 'blue'])

#%%
###   Supp. FIGURE 6B - hm4di power spectrums, saline vs cno   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(c, e) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=False); e=e['5']

AS.compare_power_spectrums(ppath, [c, e], ['hm4di-saline', 'hm4di-cno'], istate=[1,2,3,4], pmode=0, pnorm=0, 
                           fmax=30, flatten_tnrem=4, ma_thr=20, ma_state=3, colors=['gray', 'red'])

#%%
###   Supp. FIGURE 6D - mCherry percent time spent in REM   ##
ppath = '/media/fearthekraken/Mandy_HardDrive1/dreadds_processed/'
(c, e) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False); hm3dq=e['0.25']; hm4di=e['5']

m1, T1 = pwaves.sleep_timecourse(ppath, c, istate=[1], tbin=18000, n=1, stats='perc', flatten_tnrem=4, pplot=False)  # saline
m2, T2 = pwaves.sleep_timecourse(ppath, hm3dq, istate=[1], tbin=18000, n=1, stats='perc', flatten_tnrem=4, pplot=False)  # 0.25 mg/kg CNO
m3, T3 = pwaves.sleep_timecourse(ppath, hm4di, istate=[1], tbin=18000, n=1, stats='perc', flatten_tnrem=4, pplot=False)  # 5.0 mg/kg CNO
pwaves.plot_sleep_timecourse([T1,T2,T3], [m1,m2,m3], tstart=0, tbin=18000, stats='perc', plotMode='03', 
                             group_colors=['gray', 'orangered', 'brown'], group_labels=['saline','0.25 mg/kg cno', '5.0 mg/kg cno'])
# stats
df = pwaves.df_from_timecourse_dict([T1,T2,T3], [m1,m2,m3], ['0','0.25', '5'])
res_anova = AnovaRM(data=df, depvar='t0', subject='Mouse', within=['dose']).fit()
print('\n\n   ### REM PERCENTAGE ###\n'); print(res_anova)

#%%
###   Supp. FIGURE 6E - mCherry mean REM duration   ##
ppath = '/media/fearthekraken/Mandy_HardDrive1/dreadds_processed/'
(c, e) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False); hm3dq=e['0.25']; hm4di=e['5']

m1, T1 = pwaves.sleep_timecourse(ppath, c, istate=[1], tbin=18000, n=1, stats='dur', flatten_tnrem=4, pplot=False)  # saline
m2, T2 = pwaves.sleep_timecourse(ppath, hm3dq, istate=[1], tbin=18000, n=1, stats='dur', flatten_tnrem=4, pplot=False)  # 0.25 mg/kg CNO
m3, T3 = pwaves.sleep_timecourse(ppath, hm4di, istate=[1], tbin=18000, n=1, stats='dur', flatten_tnrem=4, pplot=False)  # 5.0 mg/kg CNO
pwaves.plot_sleep_timecourse([T1,T2,T3], [m1,m2,m3], tstart=0, tbin=18000, stats='dur', plotMode='03', 
                             group_colors=['gray', 'orangered', 'brown'], group_labels=['saline','0.25 mg/kg cno', '5.0 mg/kg cno'])
# stats
df = pwaves.df_from_timecourse_dict([T1,T2,T3], [m1,m2,m3], ['0','0.25', '5'])
res_anova = AnovaRM(data=df, depvar='t0', subject='Mouse', within=['dose']).fit()
print('\n\n   ### REM DURATION ###\n'); print(res_anova)

#%%
###   Supp. FIGURE 6F - mCherry mean REM frequency   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/dreadds_processed/'
(c, e) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False); hm3dq=e['0.25']; hm4di=e['5']

m1, T1 = pwaves.sleep_timecourse(ppath, c, istate=[1], tbin=18000, n=1, stats='freq', flatten_tnrem=4, pplot=False)  # saline
m2, T2 = pwaves.sleep_timecourse(ppath, hm3dq, istate=[1], tbin=18000, n=1, stats='freq', flatten_tnrem=4, pplot=False)  # 0.25 mg/kg CNO
m3, T3 = pwaves.sleep_timecourse(ppath, hm4di, istate=[1], tbin=18000, n=1, stats='freq', flatten_tnrem=4, pplot=False)  # 5.0 mg/kg CNO
pwaves.plot_sleep_timecourse([T1,T2,T3], [m1,m2,m3], tstart=0, tbin=18000, stats='freq', plotMode='03', 
                             group_colors=['gray', 'orangered', 'brown'], group_labels=['saline','0.25 mg/kg cno', '5.0 mg/kg cno'])
# stats
df = pwaves.df_from_timecourse_dict([T1,T2,T3], [m1,m2,m3], ['0','0.25', '5'])
res_anova = AnovaRM(data=df, depvar='t0', subject='Mouse', within=['dose']).fit()
print('\n\n   ### REM FREQUENCY ###\n'); print(res_anova)

#%%
###   Supp. FIGURE 6G - mCherry percent time spent in Wake/NREM/IS   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/dreadds_processed/'
(c, e) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False); hm3dq=e['0.25']; hm4di=e['5']

m1, T1 = pwaves.sleep_timecourse(ppath, c, istate=[2,3,4], tbin=18000, n=1, stats='perc', flatten_tnrem=4, pplot=False)  # saline
m2, T2 = pwaves.sleep_timecourse(ppath, hm3dq, istate=[2,3,4], tbin=18000, n=1, stats='perc', flatten_tnrem=4, pplot=False)  # 0.25 mg/kg CNO
m3, T3 = pwaves.sleep_timecourse(ppath, hm4di, istate=[2,3,4], tbin=18000, n=1, stats='perc', flatten_tnrem=4, pplot=False)  # 5.0 mg/kg CNO
pwaves.plot_sleep_timecourse([T1,T2,T3], [m1,m2,m3], tstart=0, tbin=18000, stats='perc', plotMode='03', 
                             group_colors=['gray', 'orangered', 'brown'], group_labels=['saline','0.25 mg/kg cno', '5.0 mg/kg cno'])
# stats
df = pwaves.df_from_timecourse_dict([T1,T2,T3], [m1,m2,m3], ['0','0.25', '5'])
for s in [2,3,4]:
    sdf = df.iloc[np.where(df['state']==s)[0],:]
    res_anova = AnovaRM(data=sdf, depvar='t0', subject='Mouse', within=['dose']).fit()
    print(f'\n\n   ### STATE = {s} ###\n'); print(res_anova)
    
#%%
###   Supp. FIGURE 6H - mCherry REM sleep transition probability   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/dreadds_processed/'
(c, e) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False); hm3dq=e['0.25']; hm4di=e['5']

m1, T1 = pwaves.sleep_timecourse(ppath, c, istate=[1], tbin=18000, n=1, stats='transition probability', flatten_tnrem=False, pplot=False)  # saline
m2, T2 = pwaves.sleep_timecourse(ppath, hm3dq, istate=[1], tbin=18000, n=1, stats='transition probability', flatten_tnrem=False, pplot=False)  # 0.25 mg/kg CNO
m3, T3 = pwaves.sleep_timecourse(ppath, hm4di, istate=[1], tbin=18000, n=1, stats='transition probability', flatten_tnrem=False, pplot=False)  # 5.0 mg/kg CNO
pwaves.plot_sleep_timecourse([T1,T2,T3], [m1,m2,m3], tstart=0, tbin=18000, stats='transition probability', plotMode='03', 
                             group_colors=['gray', 'orangered', 'brown'], group_labels=['saline','0.25 mg/kg cno', '5.0 mg/kg cno'])
df = pwaves.df_from_timecourse_dict([T1,T2,T3], [m1,m2,m3], ['0','0.25', '5'])
res_anova = AnovaRM(data=df, depvar='t0', subject='Mouse', within=['dose']).fit()
print('\n\n   ### REM TRANSITION PROBABILITY ###\n'); print(res_anova)

#%%
###   Supp. FIGURE 6I - mCherry time-normalized P-wave frequency across brain state transitions   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/dreadds_processed/'
(c, e) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=True); hm3dq=e['0.25']; hm4di=e['5']
c = [i[0] for i in c if i[1] != 'X']; hm3dq = [i[0] for i in hm3dq if i[1] != 'X']; hm4di = [i[0] for i in hm4di if i[1] != 'X']
sequence=[3,4,1,2]; state_thres=[(0,10000)]*len(sequence); nstates=[20,20,20,20]  # NREM --> IS --> REM --> WAKE

m1,mx1,spe1 = pwaves.stateseq(ppath, c, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25, pnorm=1,  # saline
                                 psmooth=[2,2], mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
m2,mx2,spe2 = pwaves.stateseq(ppath, hm3dq, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25, pnorm=1,  # 0.25 mg/kg CNO
                                 psmooth=[2,2], mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
m3,mx3,spe3 = pwaves.stateseq(ppath, hm4di, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25, pnorm=1,  # 5.0 mg/kg CNO
                                 psmooth=[2,2], mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
mx_list = [mx1,mx2,mx2]

# plot timecourses
pwaves.plot_activity_transitions(mx_list, [m1,m2,m3], plot_id=['gray', 'orangered', 'brown'], xlim=nstates, 
                                 group_labels=['mCherry-saline', 'mCherry-0.25mg/kg cno', 'mCherry-5.0mg/kg cno'], 
                                 xlabel='Time (normalized', ylabel='P-waves/s', title='NREM-->tNREM-->REM-->Wake', sem=True)

#%%
###   Supp. FIGURE 6J - mCherry average P-wave frequency in each brain state   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/dreadds_processed/'
(c, e) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=True); hm3dq=e['0.25']; hm4di=e['5']
c = [i[0] for i in c if i[1] != 'X']; hm3dq = [i[0] for i in hm3dq if i[1] != 'X']; hm4di = [i[0] for i in hm4di if i[1] != 'X']

m1, x, f1, w1 = pwaves.state_freq(ppath, c, istate=[1,2,3,4], flatten_tnrem=4, pplot=False, print_stats=False)  # saline
m2, x, f2, w2 = pwaves.state_freq(ppath, hm3dq, istate=[1,2,3,4], flatten_tnrem=4, pplot=False, print_stats=False)  # 0.25 mg/kg CNO
m3, x, f3, w3 = pwaves.state_freq(ppath, hm4di, istate=[1,2,3,4], flatten_tnrem=4, pplot=False, print_stats=False)  # 5.0 mg/kg CNO
f_list = [f1,f2,f3]; w_list = [w1,w2,w3]
pwaves.plot_state_freq(x, [m1, m2, m3], f_list, w_list, group_colors=['gray', 'orangered', 'brown'], 
                       group_labels=['mCherry-saline', 'mCherry-0.25mg/kg cno', 'mCherry-5.0mg/kg cno'])
# stats
df = pd.DataFrame(columns=['Mouse', 'State', 'Dose', 'Freq'])
for i,s in enumerate([1,2,3,4]):
    df = df.append(pd.DataFrame({'Mouse' : m1 + m2 + m3,
                                 'State' : [s]*len(m1) + [s]*len(m2) + [s]*len(m3),
                                 'Dose' : ['saline']*len(m1) + ['0.25']*len(m2) + ['5']*len(m3),
                                 'Freq' : np.concatenate((f1[:,i], f2[:,i], f3[:,i])) }))
for s in [1,2,3,4]:
    sdf = df.iloc[np.where(df['State']==s)[0],:]
    res_anova = AnovaRM(data=sdf, depvar='Freq', subject='Mouse', within=['Dose']).fit()
    print(f'\n\n   ### STATE = {s} ###\n'); print(res_anova)
    
#%%
###   Supp. FIGURE 6K - mCherry power spectrums   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/dreadds_processed/'
(c, e) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False); hm3dq=e['0.25']; hm4di=e['5']

AS.compare_power_spectrums(ppath, [c, hm3dq, hm4di], ['mCherry-saline', 'mCherry-0.25mg/kg cno', 'mCherry-5.0mg/kg cno'], 
                           istate=[1,2,3,4], pmode=0, pnorm=0, fmax=30, flatten_tnrem=4, ma_thr=20, ma_state=3, colors=['gray', 'orangered', 'brown'])