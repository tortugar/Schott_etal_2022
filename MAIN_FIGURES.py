#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 18:30:46 2021

@author: fearthekraken
"""
import AS
import pwaves
import sleepy
import pandas as pd

#%%
###   FIGURE 1C - example EEGs for NREM, IS, and REM   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'

AS.plot_example(ppath, 'hans_091118n1', ['EEG'], tstart=721.5, tend=728.5, eeg_nbin=4, ylims=[(-0.6, 0.6)]) # NREM EEG
AS.plot_example(ppath, 'hans_091118n1', ['EEG'], tstart=780.0, tend=787.0, eeg_nbin=4, ylims=[(-0.6, 0.6)]) # IS EEG
AS.plot_example(ppath, 'hans_091118n1', ['EEG'], tstart=818.5, tend=825.5, eeg_nbin=4, ylims=[(-0.6, 0.6)]) # REM EEG

#%%
###   FIGURE 1E - example photometry recording   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
AS.plot_example(ppath, 'hans_091118n1', tstart=170, tend=2900, PLOT=['EEG', 'SP', 'EMG_AMP', 'HYPNO', 'DFF'], dff_nbin=1800, 
                eeg_nbin=130, fmax=25, vm=[50,1800], highres=False, pnorm=0, psmooth=[2,5], flatten_tnrem=4, ma_thr=0)

#%%
###   FIGURE 1F - average DF/F signal in each brain state   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'crh_photometry.txt')[1]

df = AS.dff_activity(ppath, recordings, istate=[1,2,3,4], ma_thr=20, flatten_tnrem=4, ma_state=3)

#%%
###   FIGURE 1G - example EEG theta burst & DF/F signal   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
AS.plot_example(ppath, 'hans_091118n1', tstart=2415, tend=2444, PLOT=['SP', 'DFF'], dff_nbin=450, fmax=20, 
                vm=[0,5], highres=True, recalc_highres=False, nsr_seg=2.5, perc_overlap=0.8, pnorm=1, psmooth=[4,4])

#%%
###   FIGURE 1H - average spectral field during REM   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'crh_photometry.txt')[1]

pwaves.spectralfield_highres_mice(ppath, recordings, pre=4, post=4, istate=[1], theta=[1,10,100,1000,10000], pnorm=1, 
                                  psmooth=[6,1], fmax=25, nsr_seg=2, perc_overlap=0.8, recalc_highres=True)

#%%
###   FIGURE 2B - recorded P-waveforms  ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions'

# left - example LFP trace with P-waves   
AS.plot_example(ppath, 'Fincher_040221n1', tstart=16112, tend=16119, PLOT=['LFP'], lfp_nbin=7, ylims=[(-0.4, 0.2)])

# right - average P-waveform
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]
pwaves.avg_waveform(ppath, recordings, istate=[],  win=[0.15,0.15], mode='pwaves', plaser=False, p_iso=0, pcluster=0, clus_event='waves')

#%%
###   FIGURE 2C - average P-wave frequency in each brain state   ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]
istate = [1,2,3,4]; p_iso=0; pcluster=0

_,_,_,_ = pwaves.state_freq(ppath, recordings, istate, plotMode='03', ma_thr=20, flatten_tnrem=4, ma_state=3,
                            p_iso=p_iso, pcluster=pcluster, ylim2=[-0.3, 0.1])

#%%
###   FIGURE 2D - time-normalized P-wave frequency across brain state transitions   ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]
sequence=[3,4,1,2]; state_thres=[(0,10000)]*len(sequence); nstates=[20,20,20,20]; vm=[0.2, 2.1]  # NREM --> IS --> REM --> WAKE

_, mx_pwave, _ = pwaves.stateseq(ppath, recordings, sequence=sequence, nstates=nstates, state_thres=state_thres, ma_thr=20, ma_state=3, 
                                       flatten_tnrem=4, fmax=25, pnorm=1, vm=vm, psmooth=[2,2], mode='pwaves', mouse_avg='mouse', print_stats=False)

#%%
###   FIGURE 2E - example theta burst & P-waves   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/dreadds_processed/'
AS.plot_example(ppath, 'Scrabble_072420n1', tstart=11318.6, tend=11323, PLOT=['SP','EEG','LFP'], eeg_nbin=1, lfp_nbin=6, fmax=20, 
                vm=[0,4.5], highres=True, recalc_highres=False, nsr_seg=1, perc_overlap=0.85, pnorm=1, psmooth=[4,5])

#%%
###   FIGURE 2F - averaged spectral power surrounding P-waves   ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]
filename = 'sp_win3'

# top - averaged spectrogram
pwaves.avg_SP(ppath, recordings, istate=[1], win=[-3,3], mouse_avg='mouse', plaser=False, pnorm=2, psmooth=[2,2], fmax=25, 
              vm=[0.8,1.5], pload=filename, psave=filename)

# bottom - averaged high theta power
_ = pwaves.avg_band_power(ppath, recordings, istate=[1], bands=[(8,15)], band_colors=['green'], win=[-3,3], mouse_avg='mouse', 
                          plaser=False, pnorm=2, psmooth=0, ylim=[0.6,1.8], pload=filename, psave=filename)

#%%
###   FIGURE 2H - example DF/F signal and P-waves   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
AS.plot_example(ppath, 'Fritz_032819n1', tstart=2991, tend=2996.75, PLOT=['DFF','LFP_THRES_ANNOT'], dff_nbin=50, lfp_nbin=10)

#%%
###   FIGURE 2I - DF/F signal surrounding P-waves   ###
ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'

# top - diagrams of P-waveforms
recordings = sleepy.load_recordings(ppath, 'pwaves_mice.txt')[0]
p_iso=0.8; pcluster=0; clus_event='waves'  # single P-waves
#p_iso=0; pcluster=0.1; clus_event='cluster start'  # clustered P-waves
pwaves.avg_waveform(ppath, recordings, istate=[],  win=[1,1], mode='pwaves', plaser=False, p_iso=p_iso, 
                    pcluster=pcluster, clus_event=clus_event, wform_std=False)

# middle/bottom - heatmaps & average DF/F plots
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'pwaves_photometry.txt')[1]
# single P-waves
pzscore=[2,2,2]; p_iso=0.8; pcluster=0; ylim=[-0.4,1.0]; vm=[-1,1.5]
iso_mx = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='ht', dff_win=[10,10], pzscore=pzscore, mouse_avg='mouse',
                               base_int=2.5, baseline_start=0, p_iso=p_iso, pcluster=pcluster, clus_event='waves', ylim=ylim, vm=vm, 
                               psmooth=(8,15), ds=1000, sf=1000)[0]
# clustered P-waves
pzscore=[2,2,2]; p_iso=0; pcluster=0.5; ylim=[-0.4,1.0]; vm=[-1,1.5]
clus_mx = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='ht', dff_win=[10,10], pzscore=pzscore, mouse_avg='mouse', 
                                base_int=2.5, baseline_start=0, p_iso=p_iso, pcluster=pcluster, clus_event='waves', ylim=ylim, vm=vm, 
                                psmooth=(4,15), ds=1000, sf=1000)[0]
# random points
pzscore=[2,2,2]; p_iso=0.8; pcluster=0; ylim=[-0.4,1.0]; vm=[-1,1.5]
jter_mx = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='ht', dff_win=[10,10], pzscore=pzscore, mouse_avg='mouse', 
                                base_int=2.5, baseline_start=0, p_iso=p_iso, pcluster=pcluster, clus_event='waves', ylim=ylim, vm=vm,
                                psmooth=(8,15), ds=1000, sf=1000, jitter=10)[0]

#%%
###   FIGURE 3B - example open loop opto recording   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'
AS.plot_example(ppath, 'Huey_082719n1', tstart=12300, tend=14000, PLOT=['LSR', 'SP', 'HYPNO'], fmax=25, vm=[50,1800], highres=False,
                pnorm=0, psmooth=[2,2], flatten_tnrem=4, ma_thr=10)

#%%
###   FIGURE 3C,D - percent time spent in each brain state surrounding laser   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_chr2_ol.txt')[1]

BS, t, df = AS.laser_brainstate(ppath, recordings, pre=400, post=520, flatten_tnrem=4, ma_state=3, ma_thr=20, edge=10, sf=0, ci='sem', ylim=[0,80])

#%%
###   FIGURE 3E - averaged SPs and frequency band power surrounding laser   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_chr2_ol.txt')[1]
bands=[(0.5,4), (6,10), (11,15), (55,99)]; band_labels=['delta', 'theta', 'sigma', 'gamma']; band_colors=['firebrick', 'limegreen', 'cyan', 'purple']

AS.laser_triggered_eeg_avg(ppath, recordings, pre=400, post=520, fmax=100, laser_dur=120, pnorm=1, psmooth=3, harmcs=10, iplt_level=2,
                           vm=[0.6,1.4], sf=7, bands=bands, band_labels=band_labels, band_colors=band_colors, ci=95, ylim=[0.6,1.3])

#%%
###   FIGURE 3G - example closed loop opto recording   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'

AS.plot_example(ppath, 'Cinderella_022420n1', tstart=7100, tend=10100, PLOT=['LSR', 'SP', 'HYPNO'], fmax=25, vm=[0,1500],
                highres=False, pnorm=0, psmooth=[2,3], flatten_tnrem=4, ma_thr=0)

#%%
###   FIGURE 3H - closed-loop ChR2 graph   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_chr2_cl.txt')[1]

_ = AS.state_online_analysis(ppath, recordings, istate=1, plotMode='03', ylim=[0,130])

#%%
###   FIGURE 3I - eYFP controls for ChR2   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_yfp_chr2_cl.txt')[1]

_ = AS.state_online_analysis(ppath, recordings, istate=1, plotMode='03', ylim=[0,130])

#%%
###   FIGURE 3J - closed-loop iC++ graph   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_ic_cl.txt')[1]

_ = AS.state_online_analysis(ppath, recordings, istate=1, plotMode='03', ylim=[0,130])

#%%
###   FIGURE 3K - eYFP controls for iC++   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'crh_yfp_ic_cl.txt')[1]

_ = AS.state_online_analysis(ppath, recordings, istate=1, plotMode='03', ylim=[0,130])

#%%
###   FIGURE 4B - example spontaneous & laser-triggered P-wave   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]

AS.plot_example(ppath, 'Huey_101719n1', tstart=5925, tend=5930, PLOT=['LSR', 'EEG', 'LFP'], eeg_nbin=5, lfp_nbin=10)

#%%
###   FIGURE 4C,D,E - waveforms & spectral power surrounding P-waves/laser   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]

# top - averaged waveforms surrounding P-waves & laser
filename = 'wf_win025'; wform_win = [0.25,0.25]; istate=[1]
pwaves.avg_waveform(ppath, recordings, istate, mode='pwaves', win=wform_win, mouse_avg='trials',  # spontaneous & laser-triggered P-waves
                    plaser=True, post_stim=0.1, pload=filename, psave=filename, ylim=[-0.3,0.1])
pwaves.avg_waveform(ppath, recordings, istate, mode='lsr', win=wform_win, mouse_avg='trials',     # successful & failed laser
                    plaser=True, post_stim=0.1, pload=filename, psave=filename, ylim=[-0.3,0.1])

# middle - averaged SPs surrounding P-waves & laser
filename = 'sp_win3'; win=[-3,3]; pnorm=2
pwaves.avg_SP(ppath, recordings, istate=[1], mode='pwaves', win=win, plaser=True, post_stim=0.1,  # spontaneous & laser-triggered P-waves
              mouse_avg='mouse', pnorm=pnorm, psmooth=[(8,8),(8,8)], vm=[(0.82,1.32),(0.8,1.45)], 
              fmax=25, recalc_highres=False, pload=filename, psave=filename)
pwaves.avg_SP(ppath, recordings, istate=[1], mode='lsr', win=win, plaser=True, post_stim=0.1,     # successful & failed laser
              mouse_avg='mouse', pnorm=pnorm, psmooth=[(8,8),(8,8)], vm=[(0.82,1.32),(0.6,1.8)], 
              fmax=25, recalc_highres=False, pload=filename, psave=filename)

# bottom - average high theta power surrounding P-waves & laser
_ = pwaves.avg_band_power(ppath, recordings, istate=[1], mode='pwaves', win=win, plaser=True,     # spontaneous & laser-triggered P-waves
                          post_stim=0.1, mouse_avg='mouse', bands=[(8,15)], band_colors=[('green')], 
                          pnorm=pnorm, psmooth=0, fmax=25, pload=filename, psave=filename, ylim=[0.5,1.5])
# successful and failed laser
_ = pwaves.avg_band_power(ppath, recordings, istate=[1], mode='lsr', win=win, plaser=True,        # successful & failed laser
                          post_stim=0.1, mouse_avg='mouse', bands=[(8,15)], band_colors=[('green')], 
                          pnorm=pnorm, psmooth=0, fmax=25, pload=filename, psave=filename, ylim=[0.5,1.5])

#%%
###   FIGURE 4F - spectral profiles: null vs spon vs success lsr vs fail lsr    ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]
filename = 'sp_win3'
spon_win=[-0.5, 0.5]; lsr_win=[0,1]; collect_win=[-3,3]; frange=[0, 20]; pnorm=2; null=True; null_win=0; null_match='lsr'

df = pwaves.sp_profiles(ppath, recordings, spon_win=spon_win, lsr_win=lsr_win, collect_win=collect_win, frange=frange, 
                        null=null, null_win=null_win, null_match=null_match, plaser=True, post_stim=0.1, pnorm=pnorm, 
                        psmooth=12, mouse_avg='mouse', ci='sem', pload=filename, psave=filename)

#%%
###   FIGURE 4G - probability of laser success per brainstate   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]
filename = 'lsr_stats'
df = pwaves.get_lsr_stats(ppath, recordings, istate=[1,2,3,4], lsr_jitter=5, post_stim=0.1, 
                          flatten_tnrem=4, ma_thr=20, ma_state=3, psave=filename)
_ = pwaves.lsr_state_success(df, istate=[1,2,3,4])  # true laser success
_ = pwaves.lsr_state_success(df, istate=[1], jstate=[1])  # true vs sham laser success

#%%
###   FIGURE 4H - latencies of elicited P-waves to laser   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]
df = pd.read_pickle('lsr_stats.pkl')

pwaves.lsr_pwave_latency(df, istate=1, jitter=True)

#%%
###   FIGURE 4I - phase preferences of spontaneous & laser-triggered P-waves   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed'  
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]
filename = 'lsr_phases'

pwaves.lsr_hilbert(ppath, recordings, istate=1, bp_filt=[6,12], min_state_dur=30, stat='perc', mode='pwaves', 
                   mouse_avg='trials', bins=9, pload=filename, psave=filename)

#%%
###   FIGURE 5B,C - example recordings of hm3dq + saline vs cno   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'

AS.plot_example(ppath, 'Dahl_030321n1', tstart=3960, tend=5210, PLOT=['EEG', 'SP', 'HYPNO', 'EMG_AMP'], eeg_nbin=100,  # saline
                fmax=25, vm=[15,2200], psmooth=(1,2), flatten_tnrem=4, ma_thr=0, ylims=[[-0.6,0.6],'','',[0,300]])
AS.plot_example(ppath, 'Dahl_031021n1', tstart=3620, tend=4870, PLOT=['EEG', 'SP', 'HYPNO', 'EMG_AMP'], eeg_nbin=100,  # CNO
                fmax=25, vm=[15,2200], psmooth=(1,2), flatten_tnrem=4, ma_thr=0, ylims=[[-0.6,0.6],'','',[0,300]])

#%%
###   FIGURE 5D - hm3dq percent time spent in REM   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(c, e) = AS.load_recordings(ppath, 'crh_hm3dq_tnrem.txt', dose=True, pwave_channel=False); e=e['0.25']

cmice, cT = pwaves.sleep_timecourse(ppath, c, istate=[1], tbin=18000, n=1, stats='perc', flatten_tnrem=4, pplot=False)  # saline
emice, eT = pwaves.sleep_timecourse(ppath, e, istate=[1], tbin=18000, n=1, stats='perc', flatten_tnrem=4, pplot=False)  # CNO
pwaves.plot_sleep_timecourse([cT,eT], [cmice, emice], tstart=0, tbin=18000, stats='perc', plotMode='03', 
                             group_colors=['gray', 'blue'], group_labels=['saline','cno'])
# stats
df = pwaves.df_from_timecourse_dict([cT,eT], [cmice,emice], ['0','0.25'])
pwaves.pairT_from_df(df.iloc[np.where(df['state']==1)[0],:], 'dose', '0', '0.25', ['t0'], print_notice='###   STATE = 1    ###')
    
#%%
###   FIGURE 5E - hm3dq mean REM duration   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(c, e) = AS.load_recordings(ppath, 'crh_hm3dq_tnrem.txt', dose=True, pwave_channel=False); e=e['0.25']

cmice, cT = pwaves.sleep_timecourse(ppath, c, istate=[1], tbin=18000, n=1, stats='dur', flatten_tnrem=4, pplot=False)  # saline
emice, eT = pwaves.sleep_timecourse(ppath, e, istate=[1], tbin=18000, n=1, stats='dur', flatten_tnrem=4, pplot=False)  # CNO
pwaves.plot_sleep_timecourse([cT,eT], [cmice, emice], tstart=0, tbin=18000, stats='dur', plotMode='03', 
                             group_colors=['gray', 'blue'], group_labels=['saline','cno'])
# stats
df = pwaves.df_from_timecourse_dict([cT,eT], [cmice,emice], ['0','0.25'])
pwaves.pairT_from_df(df.iloc[np.where(df['state']==1)[0],:], 'dose', '0', '0.25', ['t0'], print_notice='###   STATE = 1    ###')
    
#%%
###   FIGURE 5F - hm3dq mean REM frequency   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(c, e) = AS.load_recordings(ppath, 'crh_hm3dq_tnrem.txt', dose=True, pwave_channel=False); e=e['0.25']

cmice, cT = pwaves.sleep_timecourse(ppath, c, istate=[1], tbin=18000, n=1, stats='freq', flatten_tnrem=4, pplot=False)  # saline
emice, eT = pwaves.sleep_timecourse(ppath, e, istate=[1], tbin=18000, n=1, stats='freq', flatten_tnrem=4, pplot=False)  # CNO
pwaves.plot_sleep_timecourse([cT,eT], [cmice, emice], tstart=0, tbin=18000, stats='freq', plotMode='03', 
                             group_colors=['gray', 'blue'], group_labels=['saline','cno'])
# stats
df = pwaves.df_from_timecourse_dict([cT,eT], [cmice,emice], ['0','0.25'])
pwaves.pairT_from_df(df.iloc[np.where(df['state']==1)[0],:], 'dose', '0', '0.25', ['t0'], print_notice='###   STATE = 1    ###')

#%%
###   FIGURE 5G - hm3dq percent time spent in Wake/NREM/IS   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(c, e) = AS.load_recordings(ppath, 'crh_hm3dq_tnrem.txt', dose=True, pwave_channel=False); e=e['0.25']

cmice, cT = pwaves.sleep_timecourse(ppath, c, istate=[2,3,4], tbin=18000, n=1, stats='perc', flatten_tnrem=4, pplot=False)  # saline
emice, eT = pwaves.sleep_timecourse(ppath, e, istate=[2,3,4], tbin=18000, n=1, stats='perc', flatten_tnrem=4, pplot=False)  # CNO
pwaves.plot_sleep_timecourse([cT,eT], [cmice, emice], tstart=0, tbin=18000, stats='perc', plotMode='03', 
                             group_colors=['gray', 'blue'], group_labels=['saline','cno'])
# stats
df = pwaves.df_from_timecourse_dict([cT,eT], [cmice,emice], ['0','0.25'])
for s in [2,3,4]:
    pwaves.pairT_from_df(df.iloc[np.where(df['state']==s)[0],:], 'dose', '0', '0.25', ['t0'], print_notice='###   STATE = ' + str(s) + '   ###')
    
#%%
###   FIGURE 5H - hm3dq probability of IS-->REM transition   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(c, e) = AS.load_recordings(ppath, 'crh_hm3dq_tnrem.txt', dose=True, pwave_channel=False); e=e['0.25']

cmice, cT = pwaves.sleep_timecourse(ppath, c, istate=[1], tbin=18000, n=1, stats='transition probability', flatten_tnrem=False, pplot=False)  # saline
emice, eT = pwaves.sleep_timecourse(ppath, e, istate=[1], tbin=18000, n=1, stats='transition probability', flatten_tnrem=False, pplot=False)  # CNO
pwaves.plot_sleep_timecourse([cT,eT], [cmice, emice], tstart=0, tbin=18000, stats='transition probability', plotMode='03', 
                             group_colors=['gray', 'blue'], group_labels=['saline','cno'])
# stats
df = pwaves.df_from_timecourse_dict([cT,eT], [cmice,emice], ['0','0.25'])
pwaves.pairT_from_df(df, 'dose', '0', '0.25', ['t0'], print_notice='###   STATE = 1   ###')

#%%
###   FIGURE 5I - example P-waves during NREM-->IS-->REM transitions   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
AS.plot_example(ppath, 'King_071020n1', ['HYPNO', 'EEG', 'LFP'], tstart=16097, tend=16172, ylims=['',(-0.6, 0.6), (-0.3, 0.15)])  # saline
AS.plot_example(ppath, 'King_071520n1', ['HYPNO', 'EEG', 'LFP'], tstart=5600, tend=5675, ylims=['',(-0.6, 0.6), (-0.3, 0.15)])  # CNO

#%%
###   FIGURE 5J - hm3dq time-normalized P-wave frequency across brain state transitions   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(c, e) = AS.load_recordings(ppath, 'crh_hm3dq_tnrem.txt', dose=True, pwave_channel=True); e=e['0.25']
c = [i[0] for i in c if i[1] != 'X']; e = [i[0] for i in e if i[1] != 'X']
sequence=[3,4,1,2]; state_thres=[(0,10000)]*len(sequence); nstates=[20,20,20,20]; cvm=[0.3,2.5]; evm= [0.28,2.2]  # NREM --> IS --> REM --> WAKE

mice,cmx,cspe = pwaves.stateseq(ppath, c, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25, pnorm=1,  # saline
                                 vm=cvm, psmooth=[2,2], mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
mice,emx,espe = pwaves.stateseq(ppath, e, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25, pnorm=1,  # CNO
                                 vm=evm, psmooth=[2,2], mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)

# plot timecourses
pwaves.plot_activity_transitions([cmx, emx], [mice, mice], plot_id=['gray', 'blue'], group_labels=['saline', 'cno'], 
                                 xlim=nstates, xlabel='Time (normalized)', ylabel='P-waves/s', title='NREM-->tNREM-->REM-->Wake')

#%%
###   FIGURE 5K - hm3dq average P-wave frequency in each brain state   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(c, e) = AS.load_recordings(ppath, 'crh_hm3dq_tnrem.txt', dose=True, pwave_channel=True); e=e['0.25']
c = [i[0] for i in c if i[1] != 'X']; e = [i[0] for i in e if i[1] != 'X']

# top - mean P-wave frequency
mice, x, cf, cw = pwaves.state_freq(ppath, c, istate=[1,2,3,4], flatten_tnrem=4, pplot=False, print_stats=False)  # saline
mice, x, ef, ew = pwaves.state_freq(ppath, e, istate=[1,2,3,4], flatten_tnrem=4, pplot=False, print_stats=False)  # CNO
pwaves.plot_state_freq(x, [mice, mice], [cf, ef], [cw, ew], group_colors=['gray', 'blue'], group_labels=['saline','cno'])

# bottom - change in P-wave frequency from saline to CNO
fdif = (ef-cf)
df = pd.DataFrame(columns=['Mouse','State','Change'])
for i,state in enumerate(x):
    df = df.append(pd.DataFrame({'Mouse':mice, 'State':[state]*len(mice), 'Change':fdif[:,i]}))
plt.figure(); sns.barplot(x='State', y='Change', data=df, order=['NREM', 'tNREM', 'REM', 'Wake'], color='lightblue', ci=68)
sns.swarmplot(x='State', y='Change', data=df, order=['NREM', 'tNREM', 'REM', 'Wake'], color='black', size=9); plt.show()

# stats
for i,s in enumerate([1,2,3,4]):
    p = stats.ttest_rel(cf[:,i], ef[:,i], nan_policy='omit')
    print(f'saline vs cno, state={s} -- T={round(p.statistic,3)}, p-value={round(p.pvalue,5)}')

#%%
###   FIGURE 5L - hm4di percent time spent in REM   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(c, e) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=False); e=e['5']

cmice, cT = pwaves.sleep_timecourse(ppath, c, istate=[1], tbin=18000, n=1, stats='perc', flatten_tnrem=4, pplot=False)  # saline
emice, eT = pwaves.sleep_timecourse(ppath, e, istate=[1], tbin=18000, n=1, stats='perc', flatten_tnrem=4, pplot=False)  # CNO
pwaves.plot_sleep_timecourse([cT,eT], [cmice, emice], tstart=0, tbin=18000, stats='perc', plotMode='03', 
                             group_colors=['gray', 'red'], group_labels=['saline','cno'])
# stats
df = pwaves.df_from_timecourse_dict([cT,eT], [cmice,emice], ['0','5'])
pwaves.pairT_from_df(df.iloc[np.where(df['state']==1)[0],:], 'dose', '0', '5', ['t0'], print_notice='###   STATE = 1    ###')
    
#%%
###   FIGURE 5M - hm4di mean REM duration   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(c, e) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=False); e=e['5']

cmice, cT = pwaves.sleep_timecourse(ppath, c, istate=[1], tbin=18000, n=1, stats='dur', flatten_tnrem=4, pplot=False)  # saline
emice, eT = pwaves.sleep_timecourse(ppath, e, istate=[1], tbin=18000, n=1, stats='dur', flatten_tnrem=4, pplot=False)  # CNO
pwaves.plot_sleep_timecourse([cT,eT], [cmice, emice], tstart=0, tbin=18000, stats='dur', plotMode='03', 
                             group_colors=['gray', 'red'], group_labels=['saline','cno'])
# stats
df = pwaves.df_from_timecourse_dict([cT,eT], [cmice,emice], ['0','5'])
pwaves.pairT_from_df(df.iloc[np.where(df['state']==1)[0],:], 'dose', '0', '5', ['t0'], print_notice='###   STATE = 1    ###')
    
#%%
###   FIGURE 5N - hm4di mean REM frequency   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(c, e) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=False); e=e['5']

cmice, cT = pwaves.sleep_timecourse(ppath, c, istate=[1], tbin=18000, n=1, stats='freq', flatten_tnrem=4, pplot=False)  # saline
emice, eT = pwaves.sleep_timecourse(ppath, e, istate=[1], tbin=18000, n=1, stats='freq', flatten_tnrem=4, pplot=False)  # CNO
pwaves.plot_sleep_timecourse([cT,eT], [cmice, emice], tstart=0, tbin=18000, stats='freq', plotMode='03', 
                             group_colors=['gray', 'red'], group_labels=['saline','cno'])
# stats
df = pwaves.df_from_timecourse_dict([cT,eT], [cmice,emice], ['0','5'])
pwaves.pairT_from_df(df.iloc[np.where(df['state']==1)[0],:], 'dose', '0', '5', ['t0'], print_notice='###   STATE = 1    ###')

#%%
###   FIGURE 5O - hm4di percent time spent in Wake/NREM/IS   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(c, e) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=False); e=e['5']

cmice, cT = pwaves.sleep_timecourse(ppath, c, istate=[2,3,4], tbin=18000, n=1, stats='perc', flatten_tnrem=4, pplot=False)  # saline
emice, eT = pwaves.sleep_timecourse(ppath, e, istate=[2,3,4], tbin=18000, n=1, stats='perc', flatten_tnrem=4, pplot=False)  # CNO
pwaves.plot_sleep_timecourse([cT,eT], [cmice, emice], tstart=0, tbin=18000, stats='perc', plotMode='03', 
                             group_colors=['gray', 'red'], group_labels=['saline','cno'])
# stats
df = pwaves.df_from_timecourse_dict([cT,eT], [cmice,emice], ['0','5'])
for s in [2,3,4]:
    pwaves.pairT_from_df(df.iloc[np.where(df['state']==s)[0],:], 'dose', '0', '5', ['t0'], print_notice='###   STATE = ' + str(s) + '   ###')
    
#%%
###   FIGURE 5P - hm4di probability of IS-->REM transition   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(c, e) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=False); e=e['5']

cmice, cT = pwaves.sleep_timecourse(ppath, c, istate=[1], tbin=18000, n=1, stats='transition probability', flatten_tnrem=False, pplot=False)  # saline
emice, eT = pwaves.sleep_timecourse(ppath, e, istate=[1], tbin=18000, n=1, stats='transition probability', flatten_tnrem=False, pplot=False)  # CNO
pwaves.plot_sleep_timecourse([cT,eT], [cmice, emice], tstart=0, tbin=18000, stats='transition probability', plotMode='03', 
                             group_colors=['gray', 'red'], group_labels=['saline','cno'])
# stats
df = pwaves.df_from_timecourse_dict([cT,eT], [cmice,emice], ['0','5'])
pwaves.pairT_from_df(df, 'dose', '0', '5', ['t0'], print_notice='###   STATE = 1   ###')

#%%
###   FIGURE 5Q - hm4di time-normalized P-wave frequency across brain state transitions   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(c, e) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=True); e=e['5']
c = [i[0] for i in c if i[1] != 'X']; e = [i[0] for i in e if i[1] != 'X']
sequence=[3,4,1,2]; state_thres=[(0,10000)]*len(sequence); nstates=[20,20,20,20]; cvm=[0.3,2.5]; evm= [0.28,2.2]  # NREM --> IS --> REM --> WAKE

mice,cmx,cspe = pwaves.stateseq(ppath, c, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25, pnorm=1,  # saline
                                 vm=cvm, psmooth=[2,2], mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
mice,emx,espe = pwaves.stateseq(ppath, e, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25, pnorm=1,  # CNO
                                 vm=evm, psmooth=[2,2], mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)

# plot timecourses
pwaves.plot_activity_transitions([cmx, emx], [mice, mice], plot_id=['gray', 'red'], group_labels=['saline', 'cno'], 
                                 xlim=nstates, xlabel='Time (normalized)', ylabel='P-waves/s', title='NREM-->tNREM-->REM-->Wake')

#%%
###   FIGURE 5R - hm4di average P-wave frequency in each brain state   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(c, e) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=True); e=e['5']
c = [i[0] for i in c if i[1] != 'X']; e = [i[0] for i in e if i[1] != 'X']

# top - mean P-wave frequency
mice, x, cf, cw = pwaves.state_freq(ppath, c, istate=[1,2,3,4], flatten_tnrem=4, pplot=False, print_stats=False)  # saline
mice, x, ef, ew = pwaves.state_freq(ppath, e, istate=[1,2,3,4], flatten_tnrem=4, pplot=False, print_stats=False)  # CNO
pwaves.plot_state_freq(x, [mice, mice], [cf, ef], [cw, ew], group_colors=['gray', 'red'], group_labels=['saline','cno'])

# bottom - change in P-wave frequency from saline to CNO
fdif = (ef-cf)
df = pd.DataFrame(columns=['Mouse','State','Change'])
for i,state in enumerate(x):
    df = df.append(pd.DataFrame({'Mouse':mice, 'State':[state]*len(mice), 'Change':fdif[:,i]}))
plt.figure(); sns.barplot(x='State', y='Change', data=df, order=['NREM', 'tNREM', 'REM', 'Wake'], color='salmon', ci=68)
sns.swarmplot(x='State', y='Change', data=df, order=['NREM', 'tNREM', 'REM', 'Wake'], color='black', size=9); plt.show()

# stats
for i,s in enumerate([1,2,3,4]):
    p = stats.ttest_rel(cf[:,i], ef[:,i], nan_policy='omit')
    print(f'saline vs cno, state={s} -- T={round(p.statistic,3)}, p-value={round(p.pvalue,5)}')