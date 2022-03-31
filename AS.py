#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:21:14 2020

@author: fearthekraken
"""
import sys
import re
import os.path
import numpy as np
import pandas as pd
import copy
from itertools import chain
from functools import reduce
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.io as so
import scipy.stats as stats
import math
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.anova import AnovaRM
import pdb
import pyphi
import sleepy
import pwaves


#################            SUPPORT FUNCTIONS            #################

def upsample_mx(x, nbin, axis=1):
    """
    Upsample input data by duplicating each element (if $x is a vector)
    or each row/column (if $x is a 2D array) $nbin times
    @Params
    x - input data
    nbin - factor by which to duplicate
    axis - specifies dimension to upsample for 2D input data
           if 0 or 'y' - duplicate rows
           if 1 or 'x' - duplicate columns
    @Returns
    y - upsampled data
    """
    if nbin == 1:
        return x
    # get no. elements in vector to be duplicated
    if axis == 0 or axis == 'y' or x.ndim == 1:
        nelem = x.shape[0]
    elif axis == 1 or axis == 'x':
        nelem = x.shape[1]
    # upsample 1D input data
    if x.ndim == 1:
        y = np.zeros((nelem * nbin,))
        for k in range(nbin):
            y[k::nbin] = x
    # upsample 2D input data
    else:
        if axis == 0 or axis == 'y':
            y = np.zeros((nelem * nbin, x.shape[1]))
            for k in range(nbin):
                y[k::nbin, :] = x
        elif axis == 1 or axis == 'x':
            y = np.zeros((x.shape[0], nelem * nbin))
            for k in range(nbin):
                y[:, k::nbin] = x
    return y

def downsample_vec(x, nbin):
    """
    Downsample input vector by replacing $nbin consecutive bins by their mean
    @Params
    x - input vector
    bin - factor by which to downsample
    @Returns
    y - downsampled vector
    """
    n_down = int(np.floor(len(x) / nbin))
    x = x[0:n_down*nbin]
    x_down = np.zeros((n_down,))
 
    for i in range(nbin) :
        idx = list(range(i, int(n_down*nbin), int(nbin)))
        x_down += x[idx]
    y = x_down / nbin
    return y

def downsample_mx(x, nbin, axis=1):
    """
    Downsample input matrix by replacing $nbin consecutive rows or columns by their mean
    @Params
    x - input matrix
    nbin - factor by which to downsample
    axis - specifies dimension to downsample
           if 0 or 'y' - downsample rows
           if 1 or 'x' - downsample columns
    @Returns
    y - downsampled matrix
    """
    # downsample rows
    if axis == 0 or axis == 'y':
        n_down = int(np.floor(x.shape[0] / nbin))
        x = x[0:n_down * nbin, :]
        x_down = np.zeros((n_down, x.shape[1]))
        for i in range(nbin):
            idx = list(range(i, int(n_down * nbin), int(nbin)))
            x_down += x[idx, :]
    # downsample columns
    elif axis == 1 or axis == 'x':
        n_down = int(np.floor(x.shape[1] / nbin))
        x = x[:, 0:n_down * nbin]
        x_down = np.zeros((x.shape[0], n_down))
        for i in range(nbin):
            idx = list(range(i, int(n_down * nbin), int(nbin)))
            x_down += x[:, idx]
    y = x_down / nbin
    return y

def time_morph(X, nstates):
    """
    Set size of input data to $nstates bins
    X - input data; if 2D matrix, resample columns
    nstates - no. elements/columns in returned vector/matrix
    @Returns
    Y - resampled data
    """
    # upsample elements/columns by $nstates
    if X.ndim == 1:
        m = X.shape[0]
    else:
        m = X.shape[1]
    A = upsample_mx(X, nstates)
    
    # downsample A to desired size
    if X.ndim == 1:
        Y = downsample_vec(A, int((m * nstates) / nstates))
    else:
        Y = downsample_mx(A, int((m * nstates) / nstates))
    return Y

def smooth_data(x, sig):
    """
    Smooth data vector using Gaussian kernel
    @Params
    x - input data
    sig - standard deviation for smoothing
    @Returns
    sm_data - smoothed data
    """
    sig = float(sig)
    if sig == 0.0:
        return x
        
    # create gaussian
    gauss = lambda x, sig : (1/(sig*np.sqrt(2.*np.pi)))*np.exp(-(x*x)/(2.*sig*sig))
    bound = 1.0/10000
    L = 10.
    p = gauss(L, sig)
    while (p > bound):
        L = L+10
        p = gauss(L, sig)

    # create smoothing filter
    F = [gauss(x, sig) for x in np.arange(-L, L+1.)]
    F = F / np.sum(F)
    # convolve data vector with filter
    sm_data = scipy.signal.convolve2d(np.array((x,)), np.array((F,)), 'same', 'symm')
    
    return sm_data

def smooth_data2(x, nstep):
    """
    Smooth data by replacing each of $nstep consecutive bins with their mean
    @Params
    x - input data
    nstep - no. consecutive samples to average
    @Returns
    x2 - smoothed data
    """
    x2 = [[np.mean(x[i:i+nstep])]*nstep for i in np.arange(0, len(x), nstep)]
    x2 = list(chain.from_iterable(x2))
    x2 = np.array((x2))
    x2 = x2[0:len(x)]
    
    return x2

def convolve_data(x, psmooth, axis=2):
    """
    Smooth data by convolving with filter defined by $psmooth
    @Params
    x - input data
    psmooth - integer or 2-element tuple describing filter for convolution
              * for 2-element $psmooth param, idx1 smooths across rows and idx2 smooths 
                 across columns
    axis - specifies filter if $psmooth is an integer
	    if 0 or 'y' - convolve across rows
            if 1 or 'x' - convolve across columns
            if 2 or 'xy' - convolve using box filter
    @Returns
    smooth - smoothed data
    """
    if not psmooth:
        return x
    if np.isnan(x).any():
        raise KeyError('ERROR: NaN(s) found in data')
    # smooth across 1D data vector
    if x.ndim == 1:
        if type(psmooth) in [int, float]:
            filt = np.ones(psmooth) / np.sum(np.ones(psmooth))
        elif type(psmooth) in [list, tuple] and len(psmooth)==1:
            filt = np.ones(psmooth[0]) / np.sum(np.ones(psmooth[0]))
        else:
            raise KeyError('ERROR: incorrect number of values in $psmooth parameter for 1-dimensional data')
        xsmooth = scipy.signal.convolve(x, filt, mode='same')
    # smooth 2D data matrix
    elif x.ndim == 2:
        if type(psmooth) in [int, float]:
            if axis == 0 or axis == 'y':
                filt = np.ones((psmooth,1))
            elif axis == 1 or axis == 'x':
                filt = np.ones((1, psmooth))
            elif axis == 2 or axis == 'xy':
                filt = np.ones((psmooth, psmooth))
        elif type(psmooth) in [list, tuple] and len(psmooth)==2:
            filt = np.ones((psmooth[0], psmooth[1]))
        else:
            raise KeyError('ERROR: incorrect number of values in $psmooth parameter for 2-dimensional data')
        filt = filt / np.sum(filt)
        xsmooth = scipy.signal.convolve2d(x, filt, boundary='symm', mode='same')
    else:
        raise KeyError('ERROR: inputted data must be a 1 or 2-dimensional array')
    
    return xsmooth

def load_recordings(ppath, trace_file, dose, pwave_channel=False):
    """
    Load recording names, drug doses, and P-wave channels from .txt file
    @Params
    ppath - path to $trace_file
    trace_file - .txt file with recording information
    dose - if True, load drug dose info (e.g. '0' or '0.25' for DREADD experiments)
    pwave_channel - if True, load P-wave detection channel ('X' for mice without clear P-waves)
    @Returns
    ctr_list - list of control recordings
    * if $dose=True:  exp_dict - dictionary of experimental recordings (keys=doses, values=list of recording names)
    * if $dose=False: exp_list - list of experimental recordings
    """
    
    # read in $trace_file
    rfile = os.path.join(ppath, trace_file)
    f = open(rfile, newline=None)
    lines = f.readlines()
    f.close()
    
    # list of control recordings
    ctr_list = []
    # if file includes drug dose info, store experimental recordings in dictionary
    if not dose:
        exp_list = []
    # if no dose info, store exp recordings in list
    else:
        exp_dict = {}
    
    for l in lines :
        # if line starts with $ or #, skip it
        if re.search('^\s+$', l) :
            continue
        if re.search('^\s*#', l) :
            continue
        # a is any line that doesn't start with $ or #
        a = re.split('\s+', l)
        
        # for control recordings
        if re.search('C', a[0]) :
            # if file includes P-wave channel info, collect recording name and P-wave channel
            if pwave_channel:
                ctr_list.append([a[1], a[-2]])
            # if no P-wave channel info, collect recording name
            else:
                ctr_list.append(a[1])
        
        # for experimental recordings
        if re.search('E', a[0]) :
            #  if no dose info, collect exp recordings in list
            if not dose:
                if pwave_channel:
                    exp_list.append([a[1], a[-2]])
                else:
                    exp_list.append(a[1])
            # if file has dose info, collect exp recordings in dictionary
            # (keys=doses, values=lists of recording names)
            else:
                if a[2] in exp_dict:
                    if pwave_channel:
                        exp_dict[a[2]].append([a[1], a[-2]])
                    else:
                        exp_dict[a[2]].append(a[1])
                else:
                    if pwave_channel:
                        exp_dict[a[2]] = [[a[1], a[-2]]]
                    else:
                        exp_dict[a[2]] = [a[1]]
    # returs 1 list and 1 dict if file has drug dose info, or 2 lists if not
    if dose:
        return ctr_list, exp_dict
    else:
        return ctr_list, exp_list
    
def load_surround_files(ppath, pload, istate, plaser, null, signal_type=''):
    """
    Load raw data dictionaries from saved .mat files
    @Params
    ppath - folder with .mat files
    pload - base filename
    istate - brain state(s) to load data files
    plaser - if True, load files for laser-triggered P-waves, spontaneous P-waves,
                      successful laser pulses, and failed laser pulses
             if False, load file for all P-waves
    null - if True, load file for randomized control points
    signal_type - string indicating type of data loaded (e.g. SP, LFP), completes 
                  default filename
    @Returns
    *if plaser --> lsr_pwaves  - dictionaries with brain states as keys, and sub-dictionaries as values
                                 Sub-dictionaries have mouse recordings as keys, with lists of 2D or 3D signals as values
                                 Signals represent the time window surrounding each laser-triggered P-wave
                   spon_pwaves - signals surrounding each spontaneous P-wave
                   success_lsr - signals surrounding each successful laser pulse
                   fail_lsr    - signals surrounding each failed laser pulse
                   null_pts    - signals surrounding each random control point
                   data_shape  - tuple with shape of the data from one trial 
    
    *if not plaser --> p_signal    - signals surrounding each P-wave
                       null_signal - signals surrounding each random control point
                       data_shape  - tuple with shape of the data from one trial 
    """
    if plaser:
        filename = pload if isinstance(pload, str) else f'lsrSurround_{signal_type}'
        lsr_pwaves = {}
        spon_pwaves = {}
        success_lsr = {}
        fail_lsr = {}
        null_pts = {}
        try:
            for s in istate:
                # load .mat files with stored data dictionaries
                lsr_pwaves[s] = so.loadmat(os.path.join(ppath, f'{filename}_lsr_pwaves_{s}.mat'))
                spon_pwaves[s] = so.loadmat(os.path.join(ppath, f'{filename}_spon_pwaves_{s}.mat'))
                success_lsr[s] = so.loadmat(os.path.join(ppath, f'{filename}_success_lsr_{s}.mat'))
                fail_lsr[s] = so.loadmat(os.path.join(ppath, f'{filename}_fail_lsr_{s}.mat'))
                if null:
                    null_pts[s] = so.loadmat(os.path.join(ppath, f'{filename}_null_pts_{s}.mat'))
                # remove MATLAB keys so later functions can get recording list
                for mat_key in ['__header__', '__version__', '__globals__']:
                    _ = lsr_pwaves[s].pop(mat_key)
                    _ = spon_pwaves[s].pop(mat_key)
                    _ = success_lsr[s].pop(mat_key)
                    _ = fail_lsr[s].pop(mat_key)
                    if null:
                        _ = null_pts[s].pop(mat_key)
            data_shape = so.loadmat(os.path.join(ppath, f'{filename}_data_shape.mat'))['data_shape'][0]
            print('\nLoading data dictionaries ...\n')
            return lsr_pwaves, spon_pwaves, success_lsr, fail_lsr, null_pts, data_shape
        except:
            print('\nUnable to load .mat files - collecting new data ...\n')
            return []
        
    elif not plaser:
        filename = pload if isinstance(pload, str) else f'Surround_{signal_type}'
        p_signal = {}
        null_signal = {}
        try:
            for s in istate:
                # load .mat files with stored data
                p_signal[s] = so.loadmat(os.path.join(ppath, f'{filename}_pwaves_{s}.mat'))
                if null:
                    null_signal[s] = so.loadmat(os.path.join(ppath, f'{filename}_null_{s}.mat'))
                # remove MATLAB keys so later functions can get recording list
                for mat_key in ['__header__', '__version__', '__globals__']:
                    _ = p_signal[s].pop(mat_key)
                    if null:
                        _ = null_signal[s].pop(mat_key)
            data_shape = so.loadmat(os.path.join(ppath, f'{filename}_data_shape.mat'))['data_shape'][0]
            print('\nLoading data dictionaries ...\n')
            return p_signal, null_signal, data_shape
        except:
            print('\nUnable to load .mat files - calculating new spectrograms ...\n')
            return []

def get_emg_amp(mSP, mfreq, r_mu = [10,500]):
    """
    Calculate EMG amplitude from input EMG spectrogram
    @Params
    mSP - EMG spectrogram
    mfreq - list of frequencies, corresponding to $mSP rows
    r_mu - [min,max] frequencies summed to get EMG amplitude
    @Returns
    p_mu - EMG amplitude vector
    """
    i_mu = np.where((mfreq >= r_mu[0]) & (mfreq <= r_mu[1]))[0]
    p_mu = np.sqrt(mSP[i_mu, :].sum(axis=0) * (mfreq[1] - mfreq[0]))
    return p_mu

def highres_spectrogram(ppath, rec, nsr_seg=2, perc_overlap=0.95, recalc_highres=False, 
                        mode='EEG', get_M=False):
    """
    Load or calculate high-resolution spectrogram for a recording
    @Params
    ppath - base folder
    rec - name of recording    
    nsr_seg, perc_overlap - set FFT bin size (s) and overlap (%) for spectrogram calculation
    recalc_highres - if True, recalculate high-resolution spectrogram from EEG, 
                              using $nsr_seg and $perc_overlap params
                     if False, load existing spectrogram from saved file
    mode - specifies EEG channel for calculating spectrogram
           'EEG' - return hippocampal spectrogram
           'EEG2' - return prefrontal spectrogram
           'EMG' - return EMG spectrogram
    get_M - if True, load high-res brain state annotation from saved file
    @Returns
    SP - loaded or calculated high-res spectrogram
    freq - list of spectrogram frequencies, corresponding to SP rows
    t - list of spectrogram time bins, corresponding to SP columns
    dt - no. seconds per SP time bin
    M_dt - brain state annotation upsampled to match SP time resolution
    """
    import time
    if mode == 'EEG':
        sp_file = 'SP'
    elif mode == 'EEG2':
        sp_file = 'SP2'
    elif mode == 'EMG':
        sp_file = 'mSP'
    M_dt = np.nan
    
    # load high-resolution spectrogram if it exists
    if not recalc_highres:
        if os.path.exists(os.path.join(ppath, rec, '%s_highres_%s.mat' % (sp_file.lower(), rec))):
            try:
                SPEC = so.loadmat(os.path.join(ppath, rec, '%s_highres_%s.mat' % (sp_file.lower(), rec)))
                SP = SPEC[sp_file]
                freq = SPEC['freq'][0]
                t = SPEC['t'][0]
                dt = SPEC['dt'][0][0]  # number of seconds in each SP bin
                nbin = SPEC['nbin'][0][0]  # number of EEG/EMG samples in each SP bin
            except:
                recalc_highres = True
            
            if get_M:
                # load high-resolution brain state annotation if it exists, calculate if not
                if os.path.exists(os.path.join(ppath, rec, 'remidx_highres_%s.mat' % rec)):
                    M_dt = np.squeeze(so.loadmat(os.path.join(ppath, rec, 'remidx_highres_%s.mat' % rec))['M'])
                else:
                    M, _ = sleepy.load_stateidx(ppath, rec)
                    if len(M) == SP.shape[1]:
                        M_dt = M
                    elif len(M) < SP.shape[1]:
                        M_dt = time_morph(M, SP.shape[1])
                        so.savemat(os.path.join(ppath, rec, 'remidx_highres_%s.mat' % rec), {'M' : M_dt, 'S' : {}})
        else:
            recalc_highres = True
        
    if recalc_highres:
        print('Calculating high-resolution %s spectrogram for %s ...' % (mode, rec))
        
        # load sampling rate, s per time bin, and raw EEG/EMG
        sr = sleepy.get_snr(ppath, rec)
        dt = nsr_seg*(1-perc_overlap)
        if mode == 'EEG':
            data = so.loadmat(os.path.join(ppath, rec, 'EEG.mat'), squeeze_me=True)['EEG']
        elif mode == 'EEG2':
            data = so.loadmat(os.path.join(ppath, rec, 'EEG2.mat'), squeeze_me=True)['EEG2']
        elif mode == 'EMG':
            data = so.loadmat(os.path.join(ppath, rec, 'EMG.mat'), squeeze_me=True)['EMG']
        
        # calculate and save high-res spectrogram
        freq, t, SP = scipy.signal.spectrogram(data, fs=sr, window='hanning', nperseg=int(nsr_seg * sr), 
                                                     noverlap=int(nsr_seg * sr * perc_overlap))
        nbin = len(data) / SP.shape[1]
        so.savemat(os.path.join(ppath, rec, '%s_highres_%s.mat' % (sp_file.lower(), rec)), {sp_file:SP, 'freq':freq, 't':t, 'dt':dt, 'nbin':nbin})
        time.sleep(1)
        
        if get_M:
            # upsample brainstate annotation in SP-resolution bins
            M, _ = sleepy.load_stateidx(ppath, rec)
            if len(M) == SP.shape[1]:
                M_dt = M
            elif len(M) < SP.shape[1]:
                M_dt = time_morph(M, SP.shape[1])
            elif len(M) > SP.shape[1]:
                M_dt = np.zeros((SP.shape[1], )) # no valid way to downsample M yet
            so.savemat(os.path.join(ppath, rec, 'remidx_highres_%s.mat' % rec), {'M' : M_dt, 'S' : {}})
    return SP, freq, t, dt, nbin, M_dt

def adjust_brainstate(M, dt, ma_thr=20, ma_state=3, flatten_tnrem=False, keep_MA=[1,4,5], noise_state=2):
    """
    Handle microarousals and transition states in brainstate annotation
    @Params
    M - brain state annotation
    dt - s per time bin in M
    ma_thr - microarousal threshold
    ma_state - brain state to assign microarousals (2=wake, 3=NREM, 6=separate MA state)
    flatten_tnrem - specifies handling of transition states, manually annotated
                     as 4 for successful transitions and 5 for failed transitions
                     if False - no change in annotation
                     if integer - assign all transitions to specified brain state (3=NREM, 4=general "transition state")
    keep_MA - microarousals directly following any brain state in $keep_MA are exempt from
              assignment to $ma_state; do not change manual annotation
    noise_state - brain state to assign manually annotated EEG/LFP noise
    @Returns
    M - adjusted brain state annotation
    """
    # assign EEG noise (X) to noise state
    M[np.where(M==0)[0]] = noise_state
    # handle microarousals
    ma_seq = sleepy.get_sequences(np.where(M == 2.0)[0])
    for s in ma_seq:
        if 0 < len(s) < ma_thr/dt:
            if M[s[0]-1] not in keep_MA:
                M[s] = ma_state
    # handle transition states
    if flatten_tnrem:
        M[np.where((M==4) | (M==5))[0]] = flatten_tnrem
    return M

def adjust_spectrogram(SP, pnorm, psmooth, freq=[], fmax=False):
    """
    Normalize and smooth spectrogram
    @Params
    SP - input spectrogram
    pnorm - if True, normalize each frequency in SP its mean power
    psmooth - 2-element tuple describing filter to convolve with SP
               * idx1 smooths across rows/frequencies, idx2 smooths across columns/time
    freq - optional list of SP frequencies, corresponding to rows in SP
    fmax - optional cutoff indicating the maximum frequency in adjusted SP
    @Returns
    SP - adjusted spectrogram
    """
    if psmooth:
        if psmooth == True:  # default box filter
            filt = np.ones((3,3))
        elif isinstance(psmooth, int):  # integer input creates box filter with area $psmooth^2
            filt = np.ones((psmooth, psmooth))
        elif type(psmooth) in [tuple, list] and len(psmooth) == 2:
            filt = np.ones((psmooth[0], psmooth[1]))
        # smooth SP
        filt = filt / np.sum(filt)
        SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')
        
    # normalize SP
    if pnorm:
        SP_mean = SP.mean(axis=1)
        SP = np.divide(SP, np.repeat([SP_mean], SP.shape[1], axis=0).T)
    
    # cut off SP rows/frequencies above $fmax
    if len(freq) > 0:
        if fmax:
            ifreq = np.where(freq <= fmax)[0]
            if len(ifreq) < SP.shape[0]:
                SP = SP[ifreq, :]
    return SP


###############            DATA ANALYSIS FUNCTIONS            ###############

def dff_activity(ppath, recordings, istate, tstart=10, tend=-1, pzscore=False, 
                 ma_thr=20, ma_state=3, flatten_tnrem=False):
    """
    Plot average DF/F signal in each brain state
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state(s) to analyze
    tstart, tend - time (s) into recording to start and stop collecting data
    pzscore - if True, z-score DF/F signal by its mean across the recording
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_tnrem - brain state for transition sleep
    @Returns
    df - dataframe with avg DF/F activity in each brain state for each mouse
    """
    states = {1:'REM', 2:'Wake', 3:'NREM', 4:'tNREM', 5:'failed-tNREM', 6:'Microarousals'}
    
    # clean data inputs
    if type(recordings) != list:
        recordings = [recordings]
    if type(istate) != list:
        istate = [istate]
    state_labels = [states[s] for s in istate]
    
    mice = dict()
    # get all unique mice
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())
    nmice = len(mice)
    
    # create data dictionaries 
    nstates = len(istate)
    mean_act = {m:[] for m in mice}
    mean_var = {m:[] for m in mice}
    mean_yerr = {m:[] for m in mice}
    
    for rec in recordings:
        idf = re.split('_', rec)[0]
        print('Getting data for ' + rec + ' ... ')
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M = sleepy.load_stateidx(ppath, rec)[0]
        M = adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, flatten_tnrem=flatten_tnrem)
        
        # define start and end points of analysis
        istart = int(np.round(tstart/dt))
        if tend == -1:
            iend = M.shape[0]
        else:
            iend = int(np.round(tend/dt))
        M = M[istart:iend]
        
        # calculate DF/F signal using high cutoff frequency for 465 signal
        # and very low cutoff frequency for 405 signal
        pyphi.calculate_dff(ppath, rec, wcut=10, wcut405=2, shift_only=False)
        
        # load DF/F calcium signal
        dff = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['dffd'][istart:iend]
        if pzscore:
            dff = (dff-dff.mean()) / dff.std()
        else:
            dff *= 100.0
        
        dff_mean = np.zeros((nstates,))
        dff_var = np.zeros((nstates,))
        dff_yerr = np.zeros((nstates,))
        
        # get mean, variance, & SEM of DF/F signal in each brain state
        for i, s in enumerate(istate):
            sidx = np.where(M==s)[0]
            dff_mean[i] = np.mean(dff[sidx])
            dff_var[i]  = np.var(dff[sidx])
            dff_yerr[i] = np.std(dff[sidx]) / np.sqrt(len(dff[sidx]))  # sem
        mean_act[idf].append(dff_mean)
        mean_var[idf].append(dff_var)
        mean_yerr[idf].append(dff_yerr)
    # create matrices of mouse-averaged DF/F stats (mice x brain states)
    mean_mx = np.zeros((nmice, nstates))
    var_mx  = np.zeros((nmice, nstates))
    yerr_mx = np.zeros((nmice, nstates))
    for i,idf in enumerate(mice):
        mean_mx[i,:] = np.array(mean_act[idf]).mean(axis=0)
        var_mx[i, :] = np.array(mean_var[idf]).mean(axis=0)
        yerr_mx[i, :] = np.array(mean_yerr[idf]).mean(axis=0)
    
    # create dataframe with DF/F activity data
    df = pd.DataFrame({'Mouse' : np.tile(mice, len(state_labels)),
                      'State' : np.repeat(state_labels, len(mice)),
                      'DFF' : np.reshape(mean_mx,-1,order='F')})
    
    # plot signal in each state
    plt.figure()
    sns.barplot(x='State', y='DFF', data=df, ci=68, palette={'REM':'cyan','Wake':'darkviolet',
                                                             'NREM':'darkgray', 'tNREM':'darkblue'})
    sns.pointplot(x='State', y='DFF', hue='Mouse', data=df, ci=None, markers='', color='black')
    plt.gca().get_legend().remove()
    plt.show()
    
    # stats
    res_anova = AnovaRM(data=df, depvar='DFF', subject='Mouse', within=['State']).fit()
    mc = MultiComparison(df['DFF'], df['State']).allpairtest(stats.ttest_rel, method='bonf')
    print(res_anova)
    print('p = ' + str(float(res_anova.anova_table['Pr > F'])))
    print(''); print(mc[0])
    
    return df

def laser_brainstate(ppath, recordings, pre, post, tstart=0, tend=-1, ma_thr=20, ma_state=3, 
                     flatten_tnrem=4, single_mode=False, sf=0, cond=0, edge=0, ci='sem',
                     offset=0, pplot=True, ylim=[]):
    """
    Calculate laser-triggered probability of REM, Wake, NREM, and IS
    @Params
    ppath - base folder
    recordings - list of recordings
    pre, post - time window (s) before and after laser onset
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_tnrem - brain state for transition sleep
    single_mode - if True, plot individual mice
    sf - smoothing factor for vectors of brain state percentages
    cond - if 0, plot all laser trials
           if integer, only plot laser trials where mouse was in brain state $cond 
                       during onset of the laser
    edge - buffer time (s) added to edges of [-pre,post] window, prevents filtering artifacts
    ci - plot data variation ('sd'=standard deviation, 'sem'=standard error, 
                              integer between 0 and 100=confidence interval)
    offset - shift (s) of laser time points, as control
    pplot - if True, show plots
    ylim - set y axis limits of brain state percentage plot
    @Returns
    BS - 3D data matrix of brain state percentages (mice x time bins x brain state)
    t - array of time points, corresponding to columns in $BS
    df - dataframe of brain state percentages in time intervals before, during, and
         after laser stimulation
    """
    # clean data inputs
    if type(recordings) != list:
        recordings = [recordings]
    pre += edge
    post += edge

    BrainstateDict = {}
    mouse_order = []
    for rec in recordings:
        # get unique mice
        idf = re.split('_', rec)[0]
        BrainstateDict[idf] = []
        if not idf in mouse_order:
            mouse_order.append(idf)
    nmice = len(BrainstateDict)

    for rec in recordings:
        idf = re.split('_', rec)[0]
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M = sleepy.load_stateidx(ppath, rec)[0]
        M = adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, 
                              flatten_tnrem=flatten_tnrem)
        
        # define start and end points of analysis
        istart = int(np.round(tstart / dt))
        if tend == -1:
            iend = len(M)
        else:
            iend = int(np.round(tend / dt))
        # define start and end points for collecting laser trials
        ipre  = int(np.round(pre/dt))
        ipost = int(np.round(post/dt))
        
        # load and downsample laser vector
        lsr = sleepy.load_laser(ppath, rec)
        (idxs, idxe) = sleepy.laser_start_end(lsr, offset=offset)
        idxs = [int(i/nbin) for i in idxs]
        idxe = [int(i/nbin) for i in idxe]
        laser_dur = np.mean((np.array(idxe) - np.array(idxs))) * dt
        
        # collect brain states surrounding each laser trial
        for (i,j) in zip(idxs, idxe):
            if i>=ipre and i+ipost<=len(M)-1 and i>istart and i < iend:
                bs = M[i-ipre:i+ipost+1]                
                BrainstateDict[idf].append(bs) 

    t = np.linspace(-ipre*dt, ipost*dt, ipre+ipost+1)
    izero = np.where(t>0)[0][0]  # 1st time bin overlapping with laser
    izero -= 1

    # create 3D data matrix of mice x time bins x brain states
    BS = np.zeros((nmice, len(t), 4))
    Trials = []
    imouse = 0
    for mouse in mouse_order:
        if cond==0:
            # collect all laser trials
            M = np.array(BrainstateDict[mouse])
            Trials.append(M)
            for state in range(1,5):
                C = np.zeros(M.shape)
                C[np.where(M==state)] = 1
                BS[imouse,:,state-1] = C.mean(axis=0)
        if cond>0:
            # collect laser trials during brain state $cond
            M = BrainstateDict[mouse]
            Msel = []
            for trial in M:
                if trial[izero] == cond:
                    Msel.append(trial)
            M = np.array(Msel)
            Trials.append(M)
            for state in range(1,5):
                C = np.zeros(M.shape)
                C[np.where(M==state)] = 1
                BS[imouse,:,state-1] = C.mean(axis=0)
        imouse += 1

    # flatten Trials
    Trials = reduce(lambda x,y: np.concatenate((x,y), axis=0),  Trials)
    # smooth mouse averages
    if sf > 0:
        for state in range(4):
            for i in range(nmice):
                BS[i, :, state] = smooth_data(BS[i, :, state], sf)
    nmice = imouse
    
    ###   GRAPHS   ###
    if pplot:
        state_label = {0:'REM', 1:'Wake', 2:'NREM', 3:'IS'}
        it = np.where((t >= -pre + edge) & (t <= post - edge))[0]
        plt.ion()
        
        if not single_mode:
            # plot average % time in each brain state surrounding laser
            plt.figure()
            ax = plt.axes([0.15, 0.15, 0.6, 0.7])
            colors = ['cyan', 'purple', 'gray', 'darkblue']
            for state in [3,2,1,0]:
                if type(ci) in [int, float]:
                    # plot confidence interval
                    BS2 = BS[:,:,state].reshape(-1,order='F') * 100
                    t2 = np.repeat(t, BS.shape[0])
                    sns.lineplot(x=t2, y=BS2, color=colors[state], ci=ci, err_kws={'alpha':0.4, 'zorder':3}, 
                                 linewidth=3, ax=ax)
                else:
                    # plot SD or SEM
                    tmp = BS[:, :, state].mean(axis=0) *100
                    plt.plot(t[it], tmp[it], color=colors[state], lw=3, label=state_label[state])
                    if nmice > 1:
                        if ci == 'sem':
                            smp = BS[:,:,state].std(axis=0) / np.sqrt(nmice) * 100
                        elif ci == 'sd':
                            smp = BS[:,:,state].std(axis=0) * 100
                        plt.fill_between(t[it], tmp[it]-smp[it], tmp[it]+smp[it], 
                                         color=colors[state], alpha=0.4, zorder=3)
            # set axis limits and labels
            plt.xlim([-pre+edge, post-edge])
            if len(ylim) == 2:
                plt.ylim(ylim)
            ax.add_patch(patches.Rectangle((0,0), laser_dur, 100, facecolor=[0.6, 0.6, 1], 
                                           edgecolor=[0.6, 0.6, 1]))
            sleepy.box_off(ax)
            plt.xlabel('Time (s)')
            plt.ylabel('Brain state (%)')
            plt.legend()
            plt.draw()
        else:
            # plot % brain states surrounding laser for each mouse
            plt.figure(figsize=(7,7))
            clrs = sns.color_palette("husl", nmice)
            for state in [3,2,1,0]:
                ax = plt.subplot('51' + str(5-state))
                for i in range(nmice):
                    plt.plot(t[it], BS[i,it,state]*100, color=clrs[i], label=mouse_order[i])
                # plot laser interval
                ax.add_patch(patches.Rectangle((0, 0), laser_dur, 100, facecolor=[0.6, 0.6, 1], 
                                               edgecolor=[0.6, 0.6, 1], alpha=0.8))
                # set axis limits and labels
                plt.xlim((t[it][0], t[it][-1]))
                if len(ylim) == 2:
                    plt.ylim(ylim)
                plt.ylabel('% ' + state_label[state])
                if state==0:
                    plt.xlabel('Time (s)')
                else:
                    ax.set_xticklabels([])
                if state==3:
                    ax.legend(mouse_order, bbox_to_anchor=(0., 1.0, 1., .102), loc=3, 
                              mode='expand', ncol=len(mouse_order), frameon=False)
            sleepy.box_off(ax)

        # plot brain state surrounding each laser trial
        plt.figure(figsize=(4,6))
        sleepy.set_fontarial()
        ax = plt.axes([0.15, 0.1, 0.8, 0.8])
        cmap = plt.cm.jet
        my_map = cmap.from_list('ha', [[0,1,1],[0.5,0,1],[0.6, 0.6, 0.6],[0.1,0.1,0.5]], 4)
        x = list(range(Trials.shape[0]))
        im = ax.pcolorfast(t,np.array(x), np.flipud(Trials), cmap=my_map)
        im.set_clim([1,4])
        # plot laser interval
        ax.plot([0,0], [0, len(x)-1], color='white')
        ax.plot([laser_dur,laser_dur], [0, len(x)-1], color='white')
        # set axis limits and labels
        ax.axis('tight')
        plt.draw()
        plt.xlabel('Time (s)')
        plt.ylabel('Trial No.')
        sleepy.box_off(ax)
        plt.show()

    # create dataframe with baseline and laser values for each trial
    ilsr   = np.where((t>=0) & (t<=laser_dur))[0]
    ibase  = np.where((t>=-laser_dur) & (t<0))[0]
    iafter = np.where((t>=laser_dur) & (t<laser_dur*2))[0]
    S = ['REM', 'Wake', 'NREM', 'IS']
    mice = mouse_order + mouse_order + mouse_order
    lsr  = np.concatenate((np.ones((nmice,), dtype='int'), np.zeros((nmice,), dtype='int'), 
                           np.ones((nmice,), dtype='int')*2))
    lsr_char = pd.Series(['LSR']*nmice + ['PRE']*nmice + ['POST']*nmice, 
                         dtype='category')
    df = pd.DataFrame(columns = ['Mouse'] + S + ['Lsr'])
    df['Mouse'] = mice
    df['Lsr'] = lsr
    # slightly different dataframe organization
    df2 = pd.DataFrame(columns = ['Mouse', 'State', 'Perc', 'Lsr'])
    for i, state in enumerate(S):
        state_perc = np.concatenate((BS[:,ilsr,i].mean(axis=1), BS[:,ibase,i].mean(axis=1), 
                                     BS[:,iafter,i].mean(axis=1)))*100
        state_label = [state]*len(state_perc)
        df[state]  = state_perc  
        df2 = df2.append(pd.DataFrame({'Mouse':mice, 'State':state_label, 
                                       'Perc':state_perc, 'Lsr':lsr_char}))
    if pplot:
        # plot bar grah of % time in each brain state during pre-laser vs. laser interval
        plt.figure()
        fig, axs = plt.subplots(2,2, constrained_layout=True)
        axs = axs.reshape(-1)
        if ci == 'sem':
            ci = 68
        for i in range(len(S)):
            sdf = df2.iloc[np.where(df2['State'] == S[i])[0], :]
            sns.barplot(x='Lsr', y='Perc', order=['PRE', 'LSR'], data=sdf, ci=ci, 
                        palette={'PRE':'gray', 'LSR':'blue'}, ax=axs[i])
            sns.pointplot(x='Lsr', y='Perc', hue='Mouse', order=['PRE', 'LSR'], data=sdf, 
                          color='black', markers='', ci=None, ax=axs[i])
            axs[i].get_legend().remove()
            axs[i].set_title(S[i]); axs[i].set_ylabel('Amount (%)')
        plt.show()

    # stats
    clabs = ['% time spent in ' + state for state in S]
    pwaves.pairT_from_df(df, cond_col='Lsr', cond1=1, cond2=0, test_cols=S, 
                         c1_label='during-laser', c2_label='pre-laser', test_col_labels=clabs)
    return BS, t, df

def laser_triggered_eeg(ppath, name, pre, post, fmax, pnorm=1, psmooth=0, vm=[], tstart=0,
                        tend=-1, cond=0, harmcs=0, iplt_level=1, peeg2=False, prune_trials=False, 
                        mu=[10,100], offset=0, pplot=True):
    """
    Calculate average laser-triggered spectrogram for a recording
    @Params
    ppath - base folder
    name - recording folder
    pre, post - time window (s) before and after laser onset
    fmax - maximum frequency in spectrogram
    pnorm - method for spectrogram normalization (0=no normalization
                                                  1=normalize SP by recording
                                                  2=normalize SP by pre-lsr baseline interval)
    psmooth - method for spectrogram smoothing (1 element specifies convolution along X axis, 
                                                2 elements define a box filter for smoothing)
    vm - controls spectrogram saturation
    tstart, tend - time (s) into recording to start and stop collecting data
    cond - if 0, plot all laser trials
           if integer, only plot laser trials where mouse was in brain state $cond 
                       during onset of the laser
                       
    harmcs - if > 0, interpolate harmonics of base frequency $harmcs
    iplt_level - if 1, interpolate one SP row above/below harmonic frequencies
                 if 2, interpolate two SP rows
    peeg2 - if True, load prefrontal spectrogram 
    prune_trials - if True, automatically remove trials with EEG/EMG artifacts
    mu - [min,max] frequencies summed to get EMG amplitude
    offset - shift (s) of laser time points, as control
    pplot - if True, show plot
    @Returns
    EEGLsr - average EEG spectrogram surrounding laser (freq x time bins)
    EMGLsr - average EMG spectrogram
    freq[ifreq] - array of frequencies, corresponding to rows in $EEGLsr
    t - array of time points, corresponding to columns in $EEGLsr
    """
    def _interpolate_harmonics(SP, freq, fmax, harmcs, iplt_level):
        """
        Interpolate harmonics of base frequency $harmcs by averaging across 3-5 
        surrounding frequencies
        """
        df = freq[2]-freq[1]
        for h in np.arange(harmcs, fmax, harmcs):
            i = np.argmin(np.abs(freq - h))
            if np.abs(freq[i] - h) < df and h != 60: 
                if iplt_level == 2:
                    SP[i,:] = (SP[i-2:i,:] + SP[i+1:i+3,:]).mean(axis=0) * 0.5
                else:
                    SP[i,:] = (SP[i-1,:] + SP[i+1,:]) * 0.5
        return SP
    
    # load sampling rate
    sr = sleepy.get_snr(ppath, name)
    nbin = int(np.round(sr) * 2.5)
    
    # load laser, get start and end idx of each stimulation train
    lsr = sleepy.load_laser(ppath, name)
    idxs, idxe = sleepy.laser_start_end(lsr, sr, offset=offset)
    laser_dur = np.mean((idxe-idxs)/sr)
    print('Average laser duration: %f; Number of trials %d' % (laser_dur, len(idxs)))
    # downsample laser to SP time    
    idxs = [int(i/nbin) for i in idxs]
    idxe = [int(i/nbin) for i in idxe]
    
    # load EEG and EMG signals
    if peeg2:
        P = so.loadmat(os.path.join(ppath, name,  'sp2_' + name + '.mat'))
    else:
        P = so.loadmat(os.path.join(ppath, name,  'sp_' + name + '.mat'))
    Q = so.loadmat(os.path.join(ppath, name, 'msp_' + name + '.mat'))
    
    # load spectrogram
    if not peeg2:
        SPEEG = np.squeeze(P['SP'])
    else:
        SPEEG = np.squeeze(P['SP2'])
    SPEMG = np.squeeze(Q['mSP'])
    freq  = np.squeeze(P['freq'])
    t     = np.squeeze(P['t'])
    dt    = float(np.squeeze(P['dt']))
    ifreq = np.where(freq<=fmax)[0]
    # get indices of time window surrounding laser
    ipre  = int(np.round(pre/dt))
    ipost = int(np.round(post/dt))
    speeg_mean = SPEEG.mean(axis=1)
    spemg_mean = SPEMG.mean(axis=1)
    
    # interpolate harmonic frequencies
    if harmcs > 0:
        SPEEG = _interpolate_harmonics(SPEEG, freq, fmax, harmcs, iplt_level)
        SPEMG = _interpolate_harmonics(SPEMG, freq, fmax, harmcs, iplt_level)
    # normalize spectrograms by recording
    if pnorm == 1:
        SPEEG = np.divide(SPEEG, np.repeat(speeg_mean, len(t)).reshape(len(speeg_mean), len(t)))
        SPEMG = np.divide(SPEMG, np.repeat(spemg_mean, len(t)).reshape(len(spemg_mean), len(t)))
    
    # define start and and points of analysis
    if tend > -1:
        i = np.where((np.array(idxs)*dt >= tstart) & (np.array(idxs)*dt <= tend))[0]
    else:
        i = np.where(np.array(idxs)*dt >= tstart)[0]
    idxs = [idxs[j] for j in i]
    idxe = [idxe[j] for j in i]

    # eliminate laser trials with detected EEG/EMG artifacts
    skips = []
    skipe = []
    if prune_trials:
        for (i,j) in zip(idxs, idxe):
            A = SPEEG[0,i-ipre:i+ipost+1] / speeg_mean[0]
            B = SPEMG[0,i-ipre:i+ipost+1] / spemg_mean[0]
            k = np.where(A >= np.median(A)*50)[0]
            l = np.where(B >= np.median(B)*500)[0]
            if len(k) > 0 or len(l) > 0:
                skips.append(i)
                skipe.append(j)
    print("kicking out %d trials" % len(skips))
    prn_lsr = [[i,j] for i,j in zip(idxs, idxe) if i not in skips]
    idxs, idxe = zip(*prn_lsr)
    # collect laser trials starting in brain state $cond
    if cond > 0:
        M = sleepy.load_stateidx(ppath, name)[0]
        cnd_lsr = [[i,j] for i,j in zip(idxs, idxe) if i < len(M) and M[i]==cond]
        idxs, idxe = zip(*cnd_lsr)

    # collect and average spectrograms surrounding each qualifying laser trial
    eeg_sps = []
    emg_sps = []
    for (i,j) in zip(idxs, idxe):
        if i>=ipre and j+ipost < len(t): 
            eeg_sps.append(SPEEG[ifreq,i-ipre:i+ipost+1])
            emg_sps.append(SPEMG[ifreq,i-ipre:i+ipost+1])
    EEGLsr = np.array(eeg_sps).mean(axis=0)
    EMGLsr = np.array(emg_sps).mean(axis=0)
    
    # normalize spectrograms by pre-laser baseline interval
    if pnorm == 2:    
        for i in range(EEGLsr.shape[0]):
            EEGLsr[i,:] = np.divide(EEGLsr[i,:], np.sum(np.abs(EEGLsr[i,0:ipre]))/(1.0*ipre))
            EMGLsr[i,:] = np.divide(EMGLsr[i,:], np.sum(np.abs(EMGLsr[i,0:ipre]))/(1.0*ipre))
    # smooth spectrograms
    EEGLsr = adjust_spectrogram(EEGLsr, pnorm=0, psmooth=psmooth)
    EMGLsr = adjust_spectrogram(EMGLsr, pnorm=0, psmooth=psmooth)
        
    dt = (1.0 / sr) * nbin
    t = np.linspace(-ipre*dt, ipost*dt, ipre+ipost+1)
    f = freq[ifreq]

    if pplot:
        # plot laser-triggered EEG spectrogram
        plt.ion()
        plt.figure(figsize=(10,8))
        ax = plt.axes([0.1, 0.55, 0.4, 0.35])
        im = ax.pcolorfast(t, f, EEGLsr, cmap='jet')
        if len(vm) == 2:
            im.set_clim(vm)
        # plot laser interval
        plt.plot([0,0], [0,f[-1]], color=(1,1,1))
        plt.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        # set axis limits and labels
        plt.axis('tight')    
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        sleepy.box_off(ax)
        plt.title('EEG', fontsize=12)
        cbar = plt.colorbar()
        if pnorm > 0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')
        # plot EEG power spectrum during laser vs pre-laser interval
        ax = plt.axes([0.62, 0.55, 0.35, 0.35])
        ilsr = np.where((t>=0) & (t<=120))[0]        
        plt.plot(f,EEGLsr[:,0:ipre].mean(axis=1), color='gray', label='baseline', lw=2)
        plt.plot(f,EEGLsr[:,ilsr].mean(axis=1), color='blue', label='laser', lw=2)
        sleepy.box_off(ax)
        plt.xlabel('Freq. (Hz)')
        plt.ylabel('Power (uV^2)')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0.)
        
        # plot laser-triggered EMG spectrogram
        ax = plt.axes([0.1, 0.1, 0.4, 0.35])
        im = ax.pcolorfast(t, f, EMGLsr, cmap='jet')
        # plot laser interval
        plt.plot([0,0], [0,f[-1]], color=(1,1,1))
        plt.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        # set axis limits and labels
        plt.axis('tight')    
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        sleepy.box_off(ax)    
        plt.title('EMG', fontsize=12)
        cbar = plt.colorbar()
        if pnorm > 0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')
        # plot EMG amplitude surrounding laser trials
        ax = plt.axes([0.62, 0.1, 0.35, 0.35])
        mf = np.where((f>=mu[0]) & (f<= mu[1]))[0]
        df = f[1]-f[0]
        # amplitude is square root of (integral over each frequency)
        avg_emg = np.sqrt(EMGLsr[mf,:].sum(axis=0)*df)    
        m = np.max(avg_emg)*1.5
        plt.plot([0,0], [0,np.max(avg_emg)*1.5], color=(0,0,0))
        plt.plot([laser_dur,laser_dur], [0,np.max(avg_emg)*1.5], color=(0,0,0))
        plt.xlim((t[0], t[-1]))
        plt.ylim((0,m))
        plt.plot(t,avg_emg, color='black', lw=2)
        sleepy.box_off(ax)     
        plt.xlabel('Time (s)')
        plt.ylabel('EMG ampl. (uV)')        
        plt.show()
    return EEGLsr, EMGLsr, freq[ifreq], t

def laser_triggered_eeg_avg(ppath, recordings, pre, post, fmax, laser_dur, pnorm=1, psmooth=0, vm=[],
                            bands=[(0.5,4),(6,10),(11,15),(55,99)], band_labels=[], band_colors=[],
                            tstart=0, tend=-1, cond=0, harmcs=0, iplt_level=1, peeg2=False, sf=0,
                            prune_trials=False, ci='sem', mu=[10,100], offset=0, pplot=True, ylim=[]):
    """
    Calculate average laser-triggered spectrogram and frequency band power for list of recordings
    @Params
    ppath - base folder
    recordings - list of recordings
    pre, post - time window (s) before and after laser onset
    fmax - maximum frequency in spectrogram
    laser_dur - duration (s) of laser stimulation trials
    pnorm - method for spectrogram normalization (0=no normalization
                                                  1=normalize SP by recording
                                                  2=normalize SP by pre-lsr baseline interval)
    psmooth - method for spectrogram smoothing (1 element specifies convolution along X axis, 
                                                2 elements define a box filter for smoothing)
    vm - controls spectrogram saturation
    bands - list of tuples with min and max frequencies in each power band
            e.g. [ [0.5,4], [6,10], [11,15], [55,100] ]
    band_labels - optional list of descriptive names for each freq band
            e.g. ['delta', 'theta', 'sigma', 'gamma']
    band_colors - optional list of colors to plot each freq band
            e.g. ['firebrick', 'limegreen', 'cyan', 'purple']
    tstart, tend - time (s) into recording to start and stop collecting data
    cond - if 0, plot all laser trials
           if integer, only plot laser trials where mouse was in brain state $cond 
                       during onset of the laser
    harmcs - if > 0, interpolate harmonics of base frequency $harmcs
    iplt_level - if 1, interpolate one SP row above/below harmonic frequencies
                 if 2, interpolate two SP rows
    peeg2 - if True, load prefrontal spectrogram
    sf - smoothing factor for vectors of frequency band power
    prune_trials - if True, automatically remove trials with EEG/EMG artifacts
    ci - plot data variation ('sd'=standard deviation, 'sem'=standard error, 
                          integer between 0 and 100=confidence interval)
    mu - [min,max] frequencies summed to get EMG amplitude
    offset - shift (s) of laser time points, as control
    pplot - if True, show plot
    ylim - set y axis limits for frequency band plot
    @Returns
    None
    """
    # clean data inputs
    if len(band_labels) != len(bands):
        band_labels = [str(b) for b in bands]
    if len(band_colors) != len(bands):
        band_colors = colorcode_mice([], return_colorlist=True)[0:len(bands)]
    if ci == 'sem':
        ci = 68
    
    # collect EEG and EMG spectrograms for each mouse
    EEGSpec = {}
    EMGSpec = {}
    mice = []
    for rec in recordings:
        # get unique mice
        idf = re.split('_', rec)[0]
        if not(idf in mice):
            mice.append(idf)
        EEGSpec[idf] = []
        EMGSpec[idf] = []
    
    for rec in recordings:
        idf = re.split('_', rec)[0]
        
        # get laser-triggered EEG and EMG spectrograms for each recording
        EEG, EMG, f, t = laser_triggered_eeg(ppath, rec, pre, post, fmax, pnorm=pnorm, psmooth=psmooth, 
                                             tstart=tstart, tend=tend, cond=cond, prune_trials=prune_trials, peeg2=peeg2,
                                             harmcs=harmcs, iplt_level=iplt_level, mu=mu, offset=offset, pplot=False)
        EEGSpec[idf].append(EEG)
        EMGSpec[idf].append(EMG)
    
    # create dictionary to store freq band power (key=freq band, value=matrix of mice x time bins)
    PwrBands = {b:np.zeros((len(mice), len(t))) for b in bands}
    
    for row, idf in enumerate(mice):
        # get average SP for each mouse
        ms_sp = np.array(EEGSpec[idf]).mean(axis=0)
        ms_emg = np.array(EMGSpec[idf]).mean(axis=0)
        # calculate power of each freq band from averaged SP
        for b in bands:
            ifreq = np.intersect1d(np.where(f >= b[0])[0], np.where(f <= b[1])[0])
            ms_band = np.mean(ms_sp[ifreq, :], axis=0)
            if sf > 0:
                ms_band = smooth_data(ms_band, sf)
            PwrBands[b][row, :] = ms_band
        EEGSpec[idf] = ms_sp
        EMGSpec[idf] = ms_emg
    # get average EEG/EMG spectrogram across all subjects
    EEGLsr = np.array([EEGSpec[k] for k in mice]).mean(axis=0)
    EMGLsr = np.array([EMGSpec[k] for k in mice]).mean(axis=0)
    
    # get indices of harmonic frequencies
    mf = np.where((f >= mu[0]) & (f <= mu[1]))[0]
    if harmcs > 0:
        harm_freq = np.arange(0, f.max(), harmcs)
        for h in harm_freq:
            mf = np.setdiff1d(mf, mf[np.where(f[mf]==h)[0]])
    # remove harmonics and calculate EMG amplitude
    df = f[1] - f[0]
    EMGAmpl = np.zeros((len(mice), EEGLsr.shape[1]))
    i=0
    for idf in mice:
        # amplitude is square root of (integral over each frequency)
        if harmcs == 0:
            EMGAmpl[i,:] = np.sqrt(EMGSpec[idf][mf,:].sum(axis=0)*df)
        else:
            tmp = 0
            for qf in mf:
                tmp += EMGSpec[idf][qf,:] * (f[qf] - f[qf-1])
            EMGAmpl[i,:] = np.sqrt(tmp)
        i += 1
    avg_emg = EMGAmpl.mean(axis=0)
    sem_emg = EMGAmpl.std(axis=0) / np.sqrt(len(mice))

    if pplot:
        plt.ion()
        plt.figure(figsize=(12,10))
        # plot average laser-triggered EEG spectrogram
        ax = plt.axes([0.1, 0.55, 0.4, 0.4])
        im = ax.pcolorfast(t, f, EEGLsr, cmap='jet')
        if len(vm) == 2:
            im.set_clim(vm)
        ax.plot([0,0], [0,f[-1]], color=(1,1,1))
        # plot laser interval
        ax.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        plt.axis('tight')    
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        sleepy.box_off(ax)
        plt.title('EEG')
        cbar = plt.colorbar(im, ax=ax, pad=0.0)
        if pnorm > 0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')
        # plot average power spectrum during laser and pre-laser interval
        ax = plt.axes([0.6, 0.55, 0.3, 0.4])
        ipre = np.where(t<0)[0]
        ilsr = np.where((t>=0) & (t<=round(laser_dur)))[0]        
        plt.plot(f,EEGLsr[:,ipre].mean(axis=1), color='gray', label='baseline', lw=2)
        plt.plot(f,EEGLsr[:,ilsr].mean(axis=1), color='blue', label='laser', lw=2)
        sleepy.box_off(ax)
        plt.xlabel('Freq. (Hz)')
        plt.ylabel('Power (uV^2)')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0.)
        
        # plot average laser-triggered EMG spectrogram
        ax = plt.axes([0.1, 0.05, 0.4, 0.4])
        im = ax.pcolorfast(t, f, EMGLsr, cmap='jet')
        ax.plot([0,0], [0,f[-1]], color=(1,1,1))
        # plot laser interval
        ax.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        plt.axis('tight')    
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        sleepy.box_off(ax)    
        plt.title('EMG')
        cbar = plt.colorbar(im, ax=ax, pad=0.0)
        if pnorm > 0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')
            
        # plot average laser-triggered power of frequency bands
        ax = plt.axes([0.6, 0.05, 0.3, 0.4])
        for b,l,c in zip(bands, band_labels, band_colors):
            data = PwrBands[b].mean(axis=0)
            yerr = PwrBands[b].std(axis=0) / np.sqrt(len(mice))
            ax.plot(t, data, color=c, label=l)
            ax.fill_between(t, data-yerr, data+yerr, color=c, alpha=0.3)
        ax.set_xlim((t[0], t[-1]))
        sleepy.box_off(ax)
        ax.set_xlabel('Time (s)')
        if pnorm > 0:
            ax.set_ylabel('Avg rel. power')
        else:
            ax.set_ylabel('Avg band power (uV^2)')
        if len(ylim) == 2:
            ax.set_ylim(ylim)
        ax.legend()
        plt.show()
    
    # get indices of pre-laser, laser, and post-laser intervals
    ilsr   = np.where((t>=0) & (t<=laser_dur))[0]
    ibase  = np.where((t>=-laser_dur) & (t<0))[0]
    iafter = np.where((t>=laser_dur) & (t<laser_dur*2))[0]
    m = mice + mice + mice
    lsr  = np.concatenate((np.ones((len(mice),), dtype='int'), 
                           np.zeros((len(mice),), dtype='int'), 
                           np.ones((len(mice),), dtype='int')*2))
    lsr_char = pd.Series(['LSR']*len(mice) + ['PRE']*len(mice) + ['POST']*len(mice), dtype='category')
    # create dataframes with power values for each frequency band
    df = pd.DataFrame(columns = ['Mouse'] + band_labels + ['Lsr'])
    df['Mouse'] = m
    df['Lsr'] = lsr
    df2 = pd.DataFrame(columns = ['Mouse', 'Band', 'Pwr', 'Lsr'])
    for b,l in zip(bands, band_labels):
        base_data = PwrBands[b][:,ibase].mean(axis=1)
        lsr_data = PwrBands[b][:,ilsr].mean(axis=1)
        post_data = PwrBands[b][:,iafter].mean(axis=1)
        # get mean power of each spectral band before/during/after laser
        b_pwr = np.concatenate((lsr_data, base_data, post_data))
        b_label = [l]*len(b_pwr)
        df[l] = b_pwr
        df2 = df2.append(pd.DataFrame({'Mouse':m, 'Band':b_label, 'Pwr':b_pwr, 'Lsr':lsr_char}))

    # plot average power of each frequency band during pre-laser vs. laser interval
    if pplot:
        plt.figure()
        fig, axs = plt.subplots(2,2, constrained_layout=True)
        axs = axs.reshape(-1)
        for i in range(len(band_labels)):
            bdf = df2.iloc[np.where(df2['Band'] == band_labels[i])[0], :]
            sns.pointplot(x='Lsr', y='Pwr', order=['PRE', 'LSR'], data=bdf, markers='o', ci=ci, 
                          palette={'PRE':'gray', 'LSR':'blue'}, ax=axs[i])
            sns.pointplot(x='Lsr', y='Pwr', hue='Mouse', order=['PRE', 'LSR'], data=bdf, 
                          color='black', markers='', ci=None, ax=axs[i])
            axs[i].get_legend().remove()
            axs[i].set_title(band_labels[i])
            if pnorm == 0:
                axs[i].set_ylabel('Power uV^2s')
            else:
                axs[i].set_ylabel('Rel. Power')
        plt.show()
    
    # stats - mean freq band power during pre-laser vs laser intervals
    clabs = [l + ' (' + str(b[0]) + '-' + str(b[1]) + ' Hz)' for b,l in zip(bands, band_labels)]
    pwaves.pairT_from_df(df, cond_col='Lsr', cond1=1, cond2=0, test_cols=band_labels, 
                         c1_label='during laser', c2_label='pre-laser', test_col_labels=clabs)


def laser_transition_probability(ppath, recordings, pre, post, tstart=0, tend=-1,
                                  ma_thr=20, ma_state=3, sf=0, offset=0):
    """
    Calculate laser-triggered likelihood of transition from IS --> REM sleep
    @Params
    ppath - base folder
    recordings - list of recordings
    pre, post - time window (s) before and after laser onset
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    sf - smoothing factor for transition state timecourses
    offset - shift (s) of laser time points, as control
    @Returns
    None
    """
    # clean data inputs
    if type(recordings) != list:
        recordings = [recordings]
    # get unique mice, create data dictionary
    mice = list({rec.split('_')[0]:[] for rec in recordings})
    BrainstateDict = {rec:[] for rec in recordings}
    # avg probability of transition during/before/after laser
    trans_prob = {m : [] for m in mice}
    
    for rec in recordings:
        print('Getting data for ' + rec + ' ...')
        idf = rec.split('_')[0]
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brainstate annotation
        M = sleepy.load_stateidx(ppath, rec)[0]
        M = adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, flatten_tnrem=False)

        # define start and end points of analysis
        istart = int(np.round(tstart / dt))
        if tend == -1:
            iend = len(M)
        else:
            iend = int(np.round(tend / dt))
        # get indices of time window surrounding laser
        ipre  = int(np.round(pre/dt))
        ipost = int(np.round(post/dt))
        
        # load laser, get start and end idx of each stimulation train
        lsr = sleepy.load_laser(ppath, rec)
        (idxs, idxe) = sleepy.laser_start_end(lsr, sr, offset=offset)
        idxs = [int(i/nbin) for i in idxs]
        idxe = [int(i/nbin) for i in idxe]
        laser_dur = np.mean((np.array(idxe) - np.array(idxs))) * dt
        laser_dn = np.zeros((len(M),))
        for (i,j) in zip(idxs, idxe):
            if i>=ipre and i+ipost<=len(M)-1 and i>istart and i < iend:
                # collect vector of pre-REM (1) and pre-wake (2) IS bouts
                tp = np.zeros((ipre+ipost+1,))
                M_cut = M[i-ipre:i+ipost+1]                
                tp[np.where(M_cut==4)[0]] = 1
                tp[np.where(M_cut==5)[0]] = 2
                BrainstateDict[rec].append(tp)
                # label downsampled indices of laser (1), pre-laser (2), and post-laser (3)
                laser_dn[i:j+1] = 1
                laser_dn[i-int(round(laser_dur/dt)) : i] = 2
                laser_dn[j+1 : j+1+int(round(laser_dur/dt))] = 3
        [laser_idx, pre_laser_idx, post_laser_idx] = [np.where(laser_dn==i)[0] for i in [1,2,3]]
        trans_idx = np.concatenate((np.where(M==4)[0], np.where(M==5)[0]), axis=0)
        trans_seq = sleepy.get_sequences(trans_idx)
        
        # collect total no. of transitions and % transitions ending in REM sleep
        l = {'num_trans':0, 'success_trans':0}
        pre_l = {'num_trans':0, 'success_trans':0}
        post_l = {'num_trans':0, 'success_trans':0}
        for tseq in trans_seq:
            # during laser period ($laser_dur s)
            if tseq[0] in laser_idx:
                l['num_trans'] += 1
                if all(M[tseq] == 4):
                    l['success_trans'] += 1
            # during pre-laser period ($laser_dur s)
            elif tseq[0] in pre_laser_idx:
                pre_l['num_trans'] += 1
                if all(M[tseq] == 4):
                    pre_l['success_trans'] += 1
            # during post-laser period ($laser_dur s)
            elif tseq[0] in post_laser_idx:
                post_l['num_trans'] += 1
                if all(M[tseq] == 4):
                    post_l['success_trans'] += 1
        trans_prob[idf].append(np.array(([pre_l['success_trans']/pre_l['num_trans']*100,
                           l['success_trans']/l['num_trans']*100,
                           post_l['success_trans']/post_l['num_trans']*100])))
    # create mouse-averaged matrix of transition probabilities (mice x pre/lsr/post)
    trans_prob_mx = np.zeros((len(mice), 3))
    for row,m in enumerate(mice):
        trans_prob_mx[row,:] = np.array((trans_prob[m])).mean(axis=0)
    
    conditions = ['pre-laser', 'laser', 'post-laser']
    # create dataframe with transition probability data
    df = pd.DataFrame({'Mouse' : np.tile(mice, len(conditions)),
                      'Cond' : np.repeat(conditions, len(mice)),
                      'Perc' : np.reshape(trans_prob_mx,-1,order='F')})
    
    ###   GRAPHS   ###
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(figsize=(7,10), nrows=2, ncols=1, gridspec_kw={'height_ratios':[2,3]})
    
    # create 3D timecourse data matrix (mice x time bins x pre/lsr/post)
    transitions_dict = pwaves.mx2d_dict(BrainstateDict, mouse_avg='mouse', d1_size=len(tp))
    transitions_mx = np.zeros((len(mice), len(tp), 3))
    for row, m in enumerate(mice):
        tt = np.sum(transitions_dict[m]>0, axis=0)  # no. transition state trials
        st = np.sum(transitions_dict[m]==1, axis=0)  # no. successful transition trials
        ft = np.sum(transitions_dict[m]==2, axis=0)  # no. failed transition trials
        
        st_perc = (st/transitions_dict[m].shape[0])*100  # % successful transition trials
        ft_perc = (ft/transitions_dict[m].shape[0])*100  # % failed transition trials
        bin_prob = [s/t*100 if t>0 else np.nan for s,t in zip(st,tt)]  # prob. of given transition being successful
        if sf > 0:
                st_perc = smooth_data(st_perc, sf)
                ft_perc = smooth_data(ft_perc, sf)
        transitions_mx[row, :, 0] = st_perc
        transitions_mx[row, :, 1] = ft_perc
        transitions_mx[row, :, 2] = bin_prob
    
    # plot timecourses of successful and failed transitions
    t = np.linspace(-ipre*dt, ipost*dt+1, ipre+ipost+1) 
    # % time in successful transitions
    sdata = np.nanmean(transitions_mx[:,:,0], axis=0)
    syerr = np.nanstd(transitions_mx[:,:,0], axis=0) / np.sqrt(len(mice))
    ax1.plot(t, sdata, color='darkblue', lw=3, label='successful tNREM')
    ax1.fill_between(t, sdata-syerr, sdata+syerr, color='darkblue', alpha=0.3)
    # % time in failed transitions
    fdata = np.nanmean(transitions_mx[:,:,1], axis=0)
    fyerr = np.nanstd(transitions_mx[:,:,1], axis=0) / np.sqrt(len(mice))
    ax1.plot(t, fdata, color='red', lw=3, label='failed tNREM')
    ax1.fill_between(t, fdata-fyerr, fdata+fyerr, color='red', alpha=0.3)
    ax1.add_patch(patches.Rectangle((0,0), laser_dur, ax1.get_ylim()[1], facecolor=[0.6, 0.6, 1], 
                                    edgecolor=[0.6, 0.6, 1], zorder=0))
    sleepy.box_off(ax1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('% time spent')
    ax1.legend()
    plt.draw()
    
    # plot bar graph of avg transition probability
    sns.barplot(x='Cond', y='Perc', data=df, ci=68, ax=ax2, palette={'pre-laser':'gray',
                                                                     'laser':'lightblue',
                                                                     'post-laser':'gray'})
    sns.pointplot(x='Cond', y='Perc', hue='Mouse', data=df, ci=None, markers='', color='black', ax=ax2)
    ax2.set_ylabel('Transition probability (%)');
    ax2.set_title('Percent IS bouts transitioning to REM')
    ax2.get_legend().remove()
    plt.show()
                
    # stats - transition probability during pre-laser vs laser vs post-laser intervals
    res_anova = AnovaRM(data=df, depvar='Perc', subject='Mouse', within=['Cond']).fit()
    mc = MultiComparison(df['Perc'], df['Cond']).allpairtest(stats.ttest_rel, method='bonf')
    print(res_anova)
    print('p = ' + str(float(res_anova.anova_table['Pr > F'])))
    print(''); print(mc[0])
    
def state_online_analysis(ppath, recordings, istate=1, plotMode='0', single_mode=False, 
                        overlap=0, ma_thr=20, ma_state=3, flatten_tnrem=False, ylim=[]):
    """
    Compare duration of laser-on vs laser-off brain states from closed-loop experiments
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state to analyze
    plotMode - parameters for bar plot
               '0' - error bar +/-SEM
               '1' - black dots for mice;  '2' - color-coded dots for mice
               '3' - black lines for mice; '4' - color-coded lines for mice
    single_mode - if True, plot individual brain state durations
                  if False, plot mean brain state duration for each mouse
    overlap - float between 0 and 100, specifying minimum percentage of overlap
              between detected (online) and annotated (offline) brain states
              required to include episode in analysis
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_tnrem - brain state for transition sleep
    ylim - set y axis limits of bar graph
    @Returns
    df - dataframe with durations of laser-on and laser-off brain states
    """
    states = {1:'REM', 2:'Wake', 3:'NREM', 4:'tNREM', 5:'failed-tNREM', 6:'Microarousals'}
    
    # clean data inputs
    if type(recordings) != list:
        recordings = [recordings]
    if type(istate) in [list, tuple]:
        istate = istate[0]
    if single_mode:
        plotMode = '05' if '0' in plotMode else '5'
    overlap = overlap / 100.0
    
    mice = dict()
    # get unique mice
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())
    if len(mice) == 1:
        single_mode=True
    
    # collect durations of control & experimental brain states
    dur_exp = {m:[] for m in mice}
    dur_ctr = {m:[] for m in mice}
    
    for rec in recordings:
        print('Getting data for ' + rec + ' ...')
        idf = re.split('_', rec)[0]
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M,_ = sleepy.load_stateidx(ppath, rec)
        M = adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, 
                              flatten_tnrem=flatten_tnrem)
        
        # load laser and online brain state detection
        laser = sleepy.load_laser(ppath, rec)
        rem_trig = so.loadmat(os.path.join(ppath, rec, 'rem_trig_%s.mat'%rec), 
                              squeeze_me=True)['rem_trig']
        # downsample to SP time
        laser = downsample_vec(laser, nbin)
        laser[np.where(laser>0)] = 1
        rem_trig = downsample_vec(rem_trig, nbin)
        rem_trig[np.where(rem_trig>0)] = 1
        laser_idx = np.where(laser==1)[0]
        rem_idx = np.where(rem_trig==1)[0]
    
        # get brain state sequences from offline analysis
        seq = sleepy.get_sequences(np.where(M==istate)[0])
        for s in seq:
            # check overlap between online & offline brain state sequences
            isect = np.intersect1d(s, rem_idx)
            if len(np.intersect1d(s, rem_idx)) > 0 and float(len(isect)) / len(s) >= overlap:
                drn = (s[-1]-s[0]+1)*dt
                # collect duration of laser-on or laser-off brain state
                if len(np.intersect1d(isect, laser_idx))>0:
                    dur_exp[idf].append(drn)
                else:
                    dur_ctr[idf].append(drn)
    
    data = {'mice':[], 'exp':[], 'ctr':[]}
    # get all brain state trials
    if len(mice) == 1 or single_mode==True:
        for m in mice:
            data['exp'] += dur_exp[m]
            data['ctr'] += dur_ctr[m]
        data['mice'] = ['']*max([len(data['ctr']), len(data['exp'])])
    # get mouse-averaged brain state duration
    else:
        for m in mice:
            dur_ctr[m] = np.array(dur_ctr[m]).mean()
            dur_exp[m] = np.array(dur_exp[m]).mean()
        data['exp'] = np.array(list(dur_exp.values()))
        data['ctr'] = np.array(list(dur_ctr.values()))
        data['mice'] = mice
    
    # create dataframe
    df = pd.DataFrame({'mice':data['mice'], 'ctr':pd.Series(data['ctr']), 
                       'exp':pd.Series(data['exp'])})
    if len(plotMode) > 0:
        if '5' not in plotMode:
            mcs = {}
            for m in mice:
                mcs.update(colorcode_mice(m))
        # plot bar graph of laser-off vs laser-on brain state duration
        plt.ion()
        plt.figure()
        ax = plt.gca()
        data_plot = [df['ctr'].mean(axis=0), df['exp'].mean(axis=0)]
        data_yerr = [df['ctr'].std(axis=0) / np.sqrt(len(df['ctr'])), 
                     df['exp'].std(axis=0) / np.sqrt(len(df['exp']))]  # SEM
    
        if '0' in plotMode:
            ax.bar([0,1], data_plot, yerr=data_yerr, align='center', 
                   color=['gray', 'blue'], edgecolor='black')
        else:
            ax.bar([0,1], data_plot, align='center', color=['gray', 'blue'], edgecolor='black')
        # plot individual brain state durations
        if '5' in plotMode:
            a = df['ctr']
            b = df['exp']
            ax.plot(np.zeros((len(a),)), a, '.', color='black')
            ax.plot(np.ones((len(b),)), b, '.', color='black')
        # plot averaged duration for each mouse
        else:
            for mrow, mname in enumerate(mice):
                points = [df['ctr'][mrow], df['exp'][mrow]]
                if '1' in plotMode:
                    markercolor = 'black'
                elif '2' in plotMode:
                    markercolor = mcs[mname]
                if '3' in plotMode:
                    linecolor = 'black'
                elif '4' in plotMode:
                    linecolor = mcs[mname]
                if '1' in plotMode or '2' in plotMode:
                    ax.plot([0,1], points, color=markercolor, marker='o', ms=8, markeredgewidth=2,
                            linewidth=0, markeredgecolor='black', label=mname, clip_on=False)
                if '3' in plotMode or '4' in plotMode:
                    ax.plot([0,1], points, color=linecolor, linewidth=2, label=mname)
        # set axis limits and labels
        ax.set_xticks([0,1])
        ax.set_xticklabels(['Lsr OFF', 'Lsr ON'])
        ax.set_ylabel('REM duration (s)')
        if len(ylim) == 2:
            ax.set_ylim(ylim)
        sleepy.box_off(ax)
        plt.show()
    
    # stats
    p = stats.ttest_rel(df['ctr'], df['exp'], nan_policy='omit')
    sig='yes' if p.pvalue < 0.05 else 'no'
    print('')
    print(f'REM duration lsr off vs on  -- T={round(p.statistic,3)}, p-value={round(p.pvalue,5)}, sig={sig}')
    print('')
    return df

def compare_online_analysis(ppath, ctr_rec, exp_rec, istate, stat, overlap=0, ma_thr=20, ma_state=3,
                            flatten_tnrem=False, mouse_avg='mouse', group_colors=[], ylim=[]):
    """
    Compare overall brain state between control and experimental mouse groups from closed-loop experiments
    @Params
    ppath - base folder
    ctr_rec - list of control recordings
    exp_rec - list of experimental recordings
    istate - brain state to analyze
    stat - statistic to compare
           'perc' - total percent time spent in brain state
           'dur' - mean overall brain state duration (across laser-on + laser-off bouts)
    overlap - float between 0 and 100, specifying minimum percentage of overlap
              between detected (online) and annotated (offline) brain states
              required to include episode in analysis
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_tnrem - brain state for transition sleep
    mouse_avg - method for data averaging; by 'mouse', 'rec'[ording], or 'trial'
    group_colors - optional 2-element list of colors for control and experimental groups
    ylim - set y axis limit for bar plot
    @Returns
    None
    """
    # clean data inputs
    if type(ctr_rec) != list:
        ctr_rec = [ctr_rec]
    if type(exp_rec) != list:
        exp_rec = [exp_rec]
    if len(group_colors) != 2:
        group_colors = ['gray', 'blue']
        
    # get control and experimental mouse names
    cmice = dict()
    for crec in ctr_rec:
        idf = re.split('_', crec)[0]
        if not idf in cmice:
            cmice[idf] = 1
    cmice = list(cmice.keys())
    emice = dict()
    for erec in exp_rec:
        idf = re.split('_', erec)[0]
        if not idf in emice:
            emice[idf] = 1
    emice = list(emice.keys())
    
    # collect list of $stat values for each control and experimental recording
    cdict = {crec:[] for crec in ctr_rec}
    edict = {erec:[] for erec in exp_rec}
    
    for rec in ctr_rec + exp_rec:
        if rec == ctr_rec[0]:
            print('\n### GETTING DATA FROM CONTROL MICE ###\n')
        elif rec == exp_rec[0]:
            print('\n### GETTING DATA FROM EXPERIMENTAL MICE ###\n')
        print('Analyzing ' + rec + ' ... ')
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M = sleepy.load_stateidx(ppath, rec)[0]
        M = adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, flatten_tnrem=flatten_tnrem)
        
        # get total % time spent in brain state
        if stat == 'perc':
            data = [(len(np.where(M==istate)[0]) / len(M)) * 100]
        # get overall brain state duration
        elif stat == 'dur':
            data = []
            # load laser and REM detection, downsample to SP time
            laser = sleepy.load_laser(ppath, rec)
            rem_trig = so.loadmat(os.path.join(ppath, rec, 'rem_trig_%s.mat'%rec), 
                                  squeeze_me=True)['rem_trig']
            laser = downsample_vec(laser, nbin)
            laser[np.where(laser>0)] = 1
            rem_trig = downsample_vec(rem_trig, nbin)
            rem_trig[np.where(rem_trig>0)] = 1
            laser_idx = np.where(laser==1)[0]
            rem_idx = np.where(rem_trig==1)[0]
            # get brain state sequences from offline analysis
            seq = sleepy.get_sequences(np.where(M==istate)[0])
            for s in seq:
                isect = np.intersect1d(s, rem_idx)
                # check overlap between online & offline brain state sequences, collect state duration
                if len(np.intersect1d(s, rem_idx)) > 0 and float(len(isect)) / len(s) >= overlap:
                    data.append((s[-1]-s[0]+1)*dt)
        if rec in ctr_rec:
            cdict[rec] = data
        elif rec in exp_rec:
            edict[rec] = data

    # create dataframes with $stat value for each mouse, recording, or trial
    cdf = pwaves.df_from_rec_dict(cdict, stat); cdf['group'] = 'control'
    edf = pwaves.df_from_rec_dict(edict, stat); edf['group'] = 'exp'
    if mouse_avg in ['mouse', 'rec']:
        cdf = cdf.groupby(mouse_avg, as_index=False)[stat].mean(); cdf['group'] = 'control'
        edf = edf.groupby(mouse_avg, as_index=False)[stat].mean(); edf['group'] = 'exp'
    df = pd.concat([cdf,edf], axis=0)
    
    # plot bar graph comparing control and experimental mice
    plt.figure()
    sns.barplot(x='group', y=stat, data=df, ci=68, palette={'control':group_colors[0], 
                                                            'exp':group_colors[1]})
    if mouse_avg in ['trial', 'trials']:
        sns.swarmplot(x='group', y=stat, data=df, palette={'control':group_colors[0], 
                                                            'exp':group_colors[1]})
    elif mouse_avg in ['mouse', 'rec']:
        sns.swarmplot(x='group', y=stat, data=df, color='black', size=8)
    if len(ylim) == 2:
        plt.ylim(ylim)
    if stat == 'perc':
        plt.ylabel('Percent time spent (%)')
    elif stat == 'dur':
        plt.ylabel('Duration (s)')
    plt.title(f'Control vs exp mice - statistic={stat}, state={istate}')
    
    # get single vectors of data
    cdata, clabels = pwaves.mx1d(cdict, mouse_avg)
    edata, elabels = pwaves.mx1d(edict, mouse_avg)
    
    # stats - unpaired t-test comparing control & experimental mice
    p = stats.ttest_ind(np.array((cdata)), np.array((edata)), nan_policy='omit')
    sig='yes' if p.pvalue < 0.05 else 'no'
    dof = len(cmice) + len(emice)
    print('')
    print(f'ctr vs exp, stat={stat}  -- T={round(p.statistic,3)}, DOF={dof}, p-value={round(p.pvalue,3)}, sig={sig}')
    print('')
    
def avg_sp_transitions(ppath, recordings, transitions, pre, post, si_threshold, sj_threshold, 
                       laser=0, bands=[(0.5,4), (6,10), (11,15), (55,99)], band_labels=[],
                       band_colors=[], tstart=0, tend=-1, fmax=30, pnorm=1, psmooth=0, vm=[], ma_thr=20, 
                       ma_state=3, flatten_tnrem=False, mouse_avg='mouse', sf=0, offset=0):
    """
    Plot average spectrogram and frequency band power at brain state transitions (absolute time)
    @Params
    ppath - base folder
    recordings - list of recordings
    transitions - list of tuples specifying brain state transitions to analyze
                  e.g. [(4,1), (1,2)] --> (IS to REM) and (REM to wake) transitions
    pre, post - time before and after state transition (s)
    si_threshold, sj_threshold - lists containing minimum duration of each of the following brain states: 
                                 ['REM', 'Wake', 'NREM', 'transition', 'failed transition', 'microarousal']
                                  * si_threshold indicates min. duration for pre-transition states, and
                                    sj_threshold indicates min. duration for post-transition states
    laser - if True, separate transitions into spontaneous vs laser-triggered
            if False - plot all state transitions
    bands - list of tuples with min and max frequencies in each power band
            e.g. [ [0.5,4], [6,10], [11,15], [55,100] ]
    band_labels - optional list of descriptive names for each freq band
            e.g. ['delta', 'theta', 'sigma', 'gamma']
    band_colors - optional list of colors to plot each freq band
            e.g. ['firebrick', 'limegreen', 'cyan', 'purple']
    tstart, tend - time (s) into recording to start and stop collecting data
    fmax - maximum frequency in spectrogram
    pnorm - if > 0, normalize each SP freq by its mean power across the recording
    psmooth - method for spectrogram smoothing (1 element specifies convolution along X axis, 
                                                2 elements define a box filter for smoothing)
    vm - 2-element list controlling saturation for [SP1, SP2]
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_tnrem - brain state for transition sleep
    mouse_avg - method for data averaging; by 'mouse' or by 'trial'
    sf - smoothing factor for vectors of frequency band power
    offset - shift (s) of laser time points, as control
    @Returns
    None
    """
    # clean data inputs
    if type(recordings) != list:
        recordings = [recordings]
    if len(vm) == 2:
        if type(vm[0]) in [int, float]:
            vm = [[vm], [vm]]
    else:
        vm = [[],[]]

    states = {1:'R', 2:'W', 3:'N', 4:'tN', 5:'ftN', 6:'MA'}
    
    mice = dict()
    # get all unique mice
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())
    
    # create data dictionaries to collect spontaneous & laser-triggered transitions
    spon_sp = {states[si] + states[sj] : [] for (si,sj) in transitions}
    if laser:
        lsr_sp = {states[si] + states[sj] : [] for (si,sj) in transitions}
    
    for (si,sj) in transitions:
        print('')
        print(f'NOW COLLECTING INFORMATION FOR {states[si]}{states[sj]} TRANSITIONS ...' )
        print('')
        
        # collect data for each transition
        sid = states[si] + states[sj]
        spon_sp_rec_dict = {rec:[] for rec in recordings}
        if laser:
            lsr_sp_rec_dict = {rec:[] for rec in recordings}
        
        for rec in recordings:
            print("Getting spectrogram for", rec, "...")
            
            # load sampling rate
            sr = sleepy.get_snr(ppath, rec)
            nbin = int(np.round(sr)*2.5)
            dt = (1.0 / sr)*nbin
            
            # load and adjust brain state annotation
            M, _ = sleepy.load_stateidx(ppath, rec)
            M = adjust_brainstate(M, dt, ma_thr, ma_state, flatten_tnrem)
            
            # load laser
            if laser:
                lsr_raw = sleepy.load_laser(ppath, rec)
                lsr_s, lsr_e = sleepy.laser_start_end(lsr_raw, sr, offset=offset)
                lsr = np.zeros((len(lsr_raw),))
                # remove pulse info
                for i, j in zip(lsr_s, lsr_e):
                    lsr[i:j] = 1
            
            # load and normalize spectrogram
            P = so.loadmat(os.path.join(ppath, rec,   'sp_' + rec + '.mat'), squeeze_me=True)
            SP = P['SP']
            f = P['freq']
            ifreq = np.where(f <= fmax)[0]
            if pnorm:
                sp_mean = SP.mean(axis=1)
                SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
            
            # define start and end points of analysis
            istart = int(np.round((1.0*tstart) / dt))
            if tend == -1: iend = len(M)
            else: iend = int(np.round((1.0*tend) / dt))
            ipre  = int(np.round(pre/dt))
            ipost = int(np.round(post/dt)) + 1
        
            # get sequences of pre-transition brain state
            seq = sleepy.get_sequences(np.where(M==si)[0])
            # if no instances of pre-transition state, continue to next recording
            if len(seq) <= 1:
                continue
            
            for s in seq:
                # last idx in pre-transition state si
                ti = s[-1]
                # check if next state is post-transition state sj; only then continue
                if ti < len(M)-1 and M[ti+1] == sj:
                    # go into future
                    p = ti+1
                    while p<len(M)-1 and M[p] == sj:
                        p += 1
                    p -= 1
                    sj_idx = list(range(ti+1, p+1))
                    # indices of state si = seq
                    # indices of state sj = sj_idx
                    
                    # if si and sj meet duration criteria, collect SP
                    if ipre <= ti < len(M)-ipost and len(s)*dt >= si_threshold[si-1]:
                        if len(sj_idx)*dt >= sj_threshold[sj-1] and istart <= ti < iend:
                            sp_si = SP[:, ti-ipre+1 : ti+1]
                            sp_sj = SP[:, ti+1 : ti+ipost+1]
                            sp_trans = np.concatenate((sp_si, sp_sj), axis=1)
                            # if $laser=1, save as either laser-triggered or spontaneous transition
                            if laser:
                                if lsr[int((ti+1)*sr*2.5)] == 1:
                                    lsr_sp_rec_dict[rec].append(sp_trans)
                                else:
                                    spon_sp_rec_dict[rec].append(sp_trans)
                            # if $laser=0, save as spontaneous transition
                            else:
                                spon_sp_rec_dict[rec].append(sp_trans)
        spon_sp[sid] = spon_sp_rec_dict
        if laser:
            lsr_sp[sid] = lsr_sp_rec_dict
    
    # get frequency band power
    for (si,sj) in transitions:
        sid = states[si]+states[sj]
        # create 3D data matrix for SPs (freq x time bins x subject)
        spon_sp_mx, labels = pwaves.mx3d(spon_sp[sid], mouse_avg)
        # create dictionary for freq band power (key=freq band, value=matrix of subject x time bins)
        spon_PwrBands = {b : np.zeros((spon_sp_mx.shape[2], spon_sp_mx.shape[1])) for b in bands}
        for layer in range(spon_sp_mx.shape[2]):
            trial_sp = spon_sp_mx[:,:,layer]
            # get mean power of each freq band from SP
            for b in bands:
                bfreq = np.intersect1d(np.where(f >= b[0])[0], np.where(f <= b[1])[0])
                band_mean = np.nanmean(trial_sp[bfreq, :], axis=0)
                if sf > 0:
                    band_mean = convolve_data(band_mean, sf)
                spon_PwrBands[b][layer, :] = band_mean
        # average/adjust spectrogram
        spon_sp_plot = adjust_spectrogram(np.nanmean(spon_sp_mx, axis=2), pnorm=0, 
                                          psmooth=psmooth, freq=f, fmax=fmax)
        # collect laser-triggered transitions
        if laser:
            lsr_sp_mx, labels = pwaves.mx3d(lsr_sp[sid], mouse_avg)
            lsr_PwrBands = {b : np.zeros((lsr_sp_mx.shape[2], lsr_sp_mx.shape[1])) for b in bands}
            for layer in range(lsr_sp_mx.shape[2]):
                trial_sp = lsr_sp_mx[:,:,layer]
                # get mean power of each freq band from SP
                for b in bands:
                    bfreq = np.intersect1d(np.where(f >= b[0])[0], np.where(f <= b[1])[0])
                    band_mean = np.nanmean(trial_sp[bfreq, :], axis=0)
                    if sf > 0:
                        band_mean = convolve_data(band_mean, sf)
                    lsr_PwrBands[b][layer, :] = band_mean
            # average/adjust spectrogram
            lsr_sp_plot = adjust_spectrogram(np.nanmean(lsr_sp_mx, axis=2), False, psmooth, f, fmax)

        t = np.linspace(-pre, post, spon_sp_plot.shape[1])
        freq = f[ifreq]
        
        ###   GRAPHS   ###
        plt.ion()
        if laser:
            fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
            [ax1,ax3,ax2,ax4] = axs.reshape(-1)
            ax1_title = 'Spontaneous Transitions'
        else:
            fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
            ax1_title = ''
        fig.suptitle(sid + ' TRANSITIONS')
        
        # plot spectrogram for spontaneous transitions
        im = ax1.pcolorfast(t, freq, spon_sp_plot, cmap='jet')
        if len(vm[0]) == 2:
            im.set_clim(vm[0])
        cbar = plt.colorbar(im, ax=ax1, pad=0.0)
        if pnorm >0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')
        # set axis limits
        ax1.set_xlim((t[0], t[-1]))
        ax1.set_xticklabels([])
        ax1.set_ylabel('Freq. (Hz)')
        ax1.set_title(ax1_title)
        # plot mean frequency band power
        for b,l,c in zip(bands, band_labels, band_colors):
            data = spon_PwrBands[b].mean(axis=0)
            yerr = spon_PwrBands[b].std(axis=0) / np.sqrt(spon_PwrBands[b].shape[0])
            ax2.plot(t, data, color=c, label=l)
            ax2.fill_between(t, data-yerr, data+yerr, color=c, alpha=0.3)
        ax2.set_xlim((t[0], t[-1]))
        ax2.set_xlabel('Time (s)')
        if pnorm > 0:
            ax2.set_ylabel('Rel. Power')
        else:
            ax2.set_ylabel('Avg. band power (uV^2)')
        ax2.legend()
        
        # plot spectrogram for laser-triggered transitions
        if laser:
            im = ax3.pcolorfast(t, freq, lsr_sp_plot, cmap='jet')
            if len(vm[1]) == 2:
                im.set_clim(vm[1])
            cbar = plt.colorbar(im, ax=ax3, pad=0.0)
            if pnorm >0:
                cbar.set_label('Rel. Power')
            else:
                cbar.set_label('Power uV^2s')
            # set axis limits
            ax3.set_xlim((t[0], t[-1]))
            ax3.set_xticklabels([])
            ax3.set_ylabel('Freq. (Hz)')
            ax3.set_title('Laser-Triggered Transitions')
            # plot mean frequency band power
            for b,l,c in zip(bands, band_labels, band_colors):
                data = lsr_PwrBands[b].mean(axis=0)
                yerr = lsr_PwrBands[b].std(axis=0) / np.sqrt(lsr_PwrBands[b].shape[0])
                ax4.plot(t, data, color=c, label=l)
                ax4.fill_between(t, data-yerr, data+yerr, color=c, alpha=0.3)
            ax4.set_xlim((t[0], t[-1]))
            ax4.set_xlabel('Time (s)')
            if pnorm > 0: ax4.set_ylabel('Rel. Power')
            else: ax4.set_ylabel('Avg. band power (uV^2)')
            ax4.legend()
            # set equal y axis limits
            y = (min([ax2.get_ylim()[0], ax4.get_ylim()[0]]), max([ax2.get_ylim()[1], ax4.get_ylim()[1]]))
            ax2.set_ylim(y); ax4.set_ylim(y)
    plt.show()

def sleep_spectrum_simple(ppath, recordings, istate=1, pnorm=0, pmode=1, fmax=30, tstart=0, tend=-1,  
                          ma_thr=20, ma_state=3, flatten_tnrem=False, noise_state=0, mu=[10,100], ci='sd', 
                          harmcs=0, pemg2=False, exclusive_mode=False, pplot=True, ylims=[]):
    """
    Get EEG power spectrum using pre-calculated spectogram saved in sp_"name".mat file
    @Params
    ppath - base folder
    recordings - list of recordings
    istate - brain state(s) to analyze
    pnorm - if > 0, normalize each SP freq by its mean power across the recording
    pmode - method for analyzing laser
            0 - plot all state episodes regardless of laser
            1 - compare states during laser vs. baseline outside laser interval
    fmax - maximum frequency in power spectrum
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_tnrem - brain state for transition sleep
    noise_state - brain state to assign manually annotated regions of EEG noise
                  (if 0, do not analyze)
    mu - [min,max] frequencies summed to get EMG amplitude
    ci - plot data variation ('sd'=standard deviation, 'sem'=standard error, 
                          integer between 0 and 100=confidence interval)
    harmcs - if > 0, interpolate harmonics of base frequency $harmcs
    pemg2 - if True, use EMG2 for EMG amplitude calcuation
    exclusive_mode - if True, isolate portions of brain state episodes with laser as "laser ON"
                     if False, consider brain state episodes with any laser overlap as "laser ON"
    pplot - if True, show plots
    ylims - optional list of y axis limits for each brain state plot
    @Returns
    ps_mx - data dictionary (key=laser state, value=power value matrix of mice x frequencies)
    freq - list of frequencies, corresponding to columns in $ps_mx arrays
    df - dataframe with EEG power spectrums
    df_amp - dataframe with EMG amplitudes
    """
    states = {1:'REM', 2:'Wake', 3:'NREM', 4:'tNREM', 5:'failed-tNREM', 6:'Microarousals'}
    
    # clean data inputs
    if type(istate) != list:
        istate=[istate]
    if len(ylims) != len(istate):
        ylims = [[]]*len(istate)
    
    mice = []
    # get unique mice
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice.append(idf)
    
    # create data dictionaries to store mouse-averaged power spectrums
    ps_mice = {s: {0:{m:[] for m in mice}, 1:{m:[] for m in mice} } for s in istate}
    amp_mice = {s: {0:{m:0 for m in mice}, 1:{m:0 for m in mice} } for s in istate}
    count_mice = {s: {0:{m:0 for m in mice}, 1:{m:0 for m in mice} } for s in istate}
    
    data = []
    for rec in recordings:
        idf = re.split('_', rec)[0]
        print('Getting data for ' + rec + ' ...')
        
        # load sampling rate
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin
        
        # load and adjust brain state annotation
        M = sleepy.load_stateidx(ppath, rec)[0]
        M = adjust_brainstate(M, dt, ma_thr=ma_thr, ma_state=ma_state, flatten_tnrem=flatten_tnrem, 
                              noise_state=noise_state)
        
        # define start and end points of analysis
        istart = int(np.round(tstart / dt))
        if tend > -1:
            iend = int(np.round(tend / dt))
        else:
            iend = len(M)
        istart_eeg = istart*nbin
        iend_eeg   = iend*nbin
        M = M[istart:iend]
        
        # load/normalize EEG spectrogram
        tmp = so.loadmat(os.path.join(ppath, rec, 'sp_%s.mat' % rec), squeeze_me=True)
        SP = tmp['SP'][:,istart:iend]
        if pnorm:
            sp_mean = np.mean(SP, axis=1)
            SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
        freq = tmp['freq']
        df = freq[1]-freq[0]
        if fmax > -1:
            ifreq = np.where(freq <= fmax)[0]
            freq = freq[ifreq]
            SP = SP[ifreq,:]
        
        # load EMG spectrogram
        tmp = so.loadmat(os.path.join(ppath, rec, 'msp_%s.mat' % rec), squeeze_me=True)
        if not pemg2:
            MSP = tmp['mSP'][:,istart:iend]
            freq_emg = tmp['freq']
        else:
            MSP = tmp['mSP2'][:,istart:iend]
        imu = np.where((freq_emg>=mu[0]) & (freq_emg<=mu[-1]))[0]
        # remove harmonic frequencies
        if harmcs > 0:
            harm_freq = np.arange(0, freq_emg.max(), harmcs)
            for h in harm_freq:
                imu = np.setdiff1d(imu, imu[np.where(np.round(freq_emg[imu], decimals=1)==h)[0]])
            tmp = 0
            for i in imu:
                tmp += MSP[i,:] * (freq_emg[i]-freq_emg[i-1])
            emg_ampl = np.sqrt(tmp)            
        else:
            emg_ampl = np.sqrt(MSP[imu,:].sum(axis=0)*df)

        # load laser, downsample to SP time
        if pmode == 1:
            lsr = sleepy.load_laser(ppath, rec)
            idxs, idxe = sleepy.laser_start_end(lsr[istart_eeg:iend_eeg])
            # downsample laser
            idxs = [int(i/nbin) for i in idxs]
            idxe = [int(i/nbin) for i in idxe]
            lsr_vec = np.zeros((len(M),))
            for (i,j) in zip(idxs, idxe):
                lsr_vec[i:j+1] = 1
            lsr_vec = lsr_vec[istart:iend]
            laser_idx = np.where(lsr_vec==1)[0]
        
        for state in istate:
            idx = np.where(M==state)[0]
            # get indices of laser ON and laser OFF brain state episodes
            if pmode == 1:
                idx_lsr   = np.intersect1d(idx, laser_idx)
                idx_nolsr = np.setdiff1d(idx, laser_idx)
                # eliminate bins without laser in each episode
                if exclusive_mode == True:
                    rm_idx = []
                    state_seq = sleepy.get_sequences(np.where(M==state)[0])
                    for s in state_seq:
                        d = np.intersect1d(s, idx_lsr)
                        if len(d) > 0:
                            drm = np.setdiff1d(s, d)
                            rm_idx.append(drm)
                            idx_nolsr = np.setdiff1d(idx_nolsr, drm)
                            idx_lsr = np.union1d(idx_lsr, drm)
                # get no. of laser ON and laser OFF episodes
                count_mice[state][0][idf] += len(idx_nolsr)
                count_mice[state][1][idf] += len(idx_lsr)
                # collect summed SPs & EMG amplitudes
                ps_lsr   = SP[:,idx_lsr].sum(axis=1)
                ps_nolsr = SP[:,idx_nolsr].sum(axis=1)
                ps_mice[state][1][idf].append(ps_lsr)
                ps_mice[state][0][idf].append(ps_nolsr)
                amp_mice[state][1][idf] += emg_ampl[idx_lsr].sum()
                amp_mice[state][0][idf] += emg_ampl[idx_nolsr].sum()
            # collect all brain state episodes, regardless of laser
            else:
                count_mice[state][0][idf] += len(idx)
                ps_nolsr = SP[:,idx].sum(axis=1)
                ps_mice[state][0][idf].append(ps_nolsr)
                amp_mice[state][0][idf] += emg_ampl[idx].sum()
    lsr_cond = []
    if pmode == 0:
        lsr_cond = [0]
    else:
        lsr_cond = [0,1]
    
    # create dataframes for EEG power spectrums and EMG amplitudes
    df = pd.DataFrame(columns=['Idf', 'Freq', 'Pow', 'Lsr', 'State'])
    df_amp = pd.DataFrame(columns=['Idf', 'Amp', 'Lsr', 'State'])
    for state, y in zip(istate, ylims):
        ps_mx  = {0:[], 1:[]}
        amp_mx = {0:[], 1:[]}
        for l in lsr_cond:
            mx  = np.zeros((len(mice), len(freq)))
            amp = np.zeros((len(mice),))
            # get mouse-averaged data
            for (i,idf) in zip(range(len(mice)), mice):
                mx[i,:] = np.array(ps_mice[state][l][idf]).sum(axis=0) / count_mice[state][l][idf]
                amp[i]  = amp_mice[state][l][idf] / count_mice[state][l][idf]
            ps_mx[l]  = mx
            amp_mx[l] = amp
        # transform data arrays to store in dataframes
        data_nolsr = list(np.reshape(ps_mx[0], (len(mice)*len(freq),)))
        amp_freq = list(freq)*len(mice)
        amp_idf = reduce(lambda x,y: x+y, [[b]*len(freq) for b in mice])
        if pmode == 1:
            data_lsr = list(np.reshape(ps_mx[1], (len(mice)*len(freq),)))
            list_lsr = ['yes']*len(freq)*len(mice) + ['no']*len(freq)*len(mice)
            data = [[a,b,c,d] for (a,b,c,d) in zip(amp_idf*2, amp_freq*2, data_lsr+data_nolsr, list_lsr)]
        else:
            list_lsr = ['no']*len(freq)*len(mice)
            data = [[a,b,c,d] for (a,b,c,d) in zip(amp_idf, amp_freq, data_nolsr, list_lsr)]
        sdf = pd.DataFrame(columns=['Idf', 'Freq', 'Pow', 'Lsr'], data=data)
        # store EMG amplitudes
        sdf_amp = pd.DataFrame(columns=['Idf', 'Amp', 'Lsr'])
        if pmode == 1:
            sdf_amp['Idf'] = mice*2
            sdf_amp['Amp'] = list(amp_mx[0]) + list(amp_mx[1])
            sdf_amp['Lsr'] = ['no'] * len(mice) + ['yes'] * len(mice)
        else:
            sdf_amp['Idf'] = mice
            sdf_amp['Amp'] = list(amp_mx[0]) 
            sdf_amp['Lsr'] = ['no'] * len(mice) 
        sdf['State'] = state
        sdf_amp['State'] = state
        df = df.append(sdf)
        df_amp = df_amp.append(sdf_amp)
            
        # plot power spectrum(s)
        if pplot:
            plt.ion()
            plt.figure()
            sns.set_style('ticks')
            sns.lineplot(data=sdf, x='Freq', y='Pow', hue='Lsr', ci=ci, 
                         palette={'yes':'blue', 'no':'gray'})
            sns.despine()
            # set axis limits and labels
            plt.xlim([freq[0], freq[-1]])
            plt.xlabel('Freq. (Hz)')
            if not pnorm:    
                plt.ylabel('Power ($\mathrm{\mu V^2}$)')
            else:
                plt.ylabel('Norm. Pow.')
            if len(y) == 2:
                plt.ylim(y)
            plt.title(f'Power spectral density during {state}')
            plt.show()
    return ps_mx, freq, df, df_amp

def compare_power_spectrums(ppath, rec_list, cond_list, istate, pnorm=0, pmode=0, fmax=30, 
                            tstart=0, tend=-1, ma_thr=20, ma_state=3, flatten_tnrem=4, 
                            noise_state=0, exclusive_mode=False, colors=[], ylims=[]):
    """
    Plot average power spectrums for any brain state; compare between multiple groups of mice
    @Params
    ppath - base folder
    rec_list - list of lists; each sub-list contains recording folders for one mouse group
    cond_list - list of labels for each group
    istate - brain state(s) to analyze
    pnorm - if > 0, normalize each SP freq by its mean power across the recording
    pmode - method for analyzing laser
            0 - plot all state episodes regardless of laser
            1 - compare states during laser vs. baseline outside laser interval
    fmax - maximum frequency in power spectrum
    tstart, tend - time (s) into recording to start and stop collecting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_tnrem - brain state for transition sleep
    noise_state - brain state to assign manually annotated regions of EEG noise
                  (if 0, do not analyze)
    exclusive_mode - if True, isolate portions of brain state episodes with laser as "laser ON"
                     if False, consider brain state episodes with any laser overlap as "laser ON"
    colors - optional list of colors for each group
    ylims - optional list of y axis limits for each brain state plot
    @Returns
    None
    """
    states = {1:'REM', 2:'Wake', 3:'NREM', 4:'tNREM', 5:'failed-tNREM', 6:'Microarousals'}
    
    # clean data inputs
    if len(cond_list) != len(rec_list):
        cond_list = ['group ' + str(i) for i in np.arange(1,len(rec_list)+1)]
    if len(colors) != len(cond_list):
        colors = colorcode_mice([], return_colorlist=True)[0:len(rec_list)]
    pal = {cond:col for cond,col in zip(cond_list, colors)}
    if len(ylims) != len(istate):
        ylims = [[]]*len(istate)
    # create dataframe of frequency power values, with mouse/group/brain state/laser info
    dfs = []
    for recordings, condition in zip(rec_list, cond_list):
        # calculate brain state power spectrums for each mouse group
        grp_df = sleep_spectrum_simple(ppath, recordings, istate, pmode=pmode, pnorm=pnorm, fmax=fmax,
                                       tstart=tstart, tend=tend, ma_thr=ma_thr, ma_state=ma_state, 
                                       flatten_tnrem=flatten_tnrem, noise_state=noise_state, pplot=False)[2]
        grp_df['Cond'] = condition
        dfs.append(grp_df)
    df = pd.concat(dfs, axis=0)
    
    # compare group power spectrums for each brain state
    for s,y in zip(istate, ylims):
        sdf = df.iloc[np.where(df['State']==s)[0], :]
        plt.ion()
        plt.figure()
        sns.set_style('ticks')
        sns.lineplot(data=sdf, x='Freq', y='Pow', hue='Cond', ci='sd', palette=pal)
        sns.despine()
        plt.xlabel('Freq. (Hz)')
        if not pnorm:    
            plt.ylabel('Power ($\mathrm{\mu V^2}$)')
        else:
            plt.ylabel('Norm. Pow.')
        if len(y) == 2:
            plt.ylim(y)
        plt.title(states[s])
        plt.show()
    
    
#################            PLOTTING FUNCTIONS            #################

def hypno_colormap():
    """
    Create colormap for Weber lab sleep annotations
    @Params
    None
    @Returns
    my_map - colormap for brain state annotations
    vmin - minimum brain state value
    vmax - maximum brain state value
    """
    # assign each brain state to a color
    state_names = ['Noise', 'REM', 'Wake', 'NREM', 'tNREM', 'failed-tNREM', 'Microarousals']
    state_colors = ['black', 'cyan', 'darkviolet', 'darkgray', 'darkblue', 'red', 'magenta']
    rgb_colors = [matplotlib.colors.to_rgba(sc) for sc in state_colors]
    
    # create colormap
    cmap = plt.cm.jet
    my_map = cmap.from_list('brs', rgb_colors, len(rgb_colors))
    vmin = 0
    vmax = 6

    return my_map, vmin, vmax

def colorcode_mice(names, return_colorlist=False):   
    """
    Load .txt file with mouse/brain state names and associated colors
    @Params
    names - list of mouse or brain state names
    return_colorlist - if True, return list of 20 pre-chosen colors
    @Returns
    colors - dictionary (key=mouse/state name, value=color) OR list of 20 colors
    """
    # 20 colors, maximally distinguishable from each other
    colorlist = ['red', 'green', 'blue', 'black', 'orange', 'fuchsia', 'yellow', 'brown', 
                 'pink', 'dodgerblue', 'chocolate', 'turquoise', 'darkviolet', 'lime', 
                 'skyblue', 'lightgray', 'darkgreen', 'yellowgreen', 'maroon', 'gray']
    if return_colorlist:
        colors = colorlist
        return colors
    
    # load txt file of mouse/brain state names and assigned colors
    colorpath = '/home/fearthekraken/Documents/Data/sleepRec_processed/mouse_colors.txt'
    f = open(colorpath, newline=None)
    lines = f.readlines()
    f.close()
    # create dictionary with mouse/brain state names as keys
    if type(names) != list:
        names = [names]
    names = [n.split('_')[0] for n in names]
    colors = {n:'' for n in names}
    
    # match mouse/brain state name to paired color in txt file
    for l in lines:
        mouse_name = re.split('\s+', l)[0]
        assigned_mouse = [n for n in names if n.lower() == mouse_name.lower()]
        if len(assigned_mouse) > 0:
            colors[assigned_mouse[0]] = re.split('\s+', l)[1]
    # assign colors to mice/states not in txt file
    unassigned_mice = [name for name in names if colors[name]=='']
    unassigned_colors =  [color for color in colorlist if color not in list(colors.values())]
    for i, um in enumerate(unassigned_mice):
        colors[um] = unassigned_colors[i]
    
    return colors


def plot_example(ppath, rec, PLOT, tstart, tend, ma_thr=20, ma_state=3, flatten_tnrem=False,
                 eeg_nbin=1, emg_nbin=1, lfp_nbin=17, dff_nbin=250, highres=False,
                 recalc_highres=True, nsr_seg=2.5, perc_overlap=0.8, pnorm=0, psmooth=0,
                 fmax=30, vm=[], cmap='jet', ylims=[], add_boxes=[]):
    """
    Plot any combination of available signals on the same time scale
    @Params
    ppath - base folder
    rec - recording folder
    PLOT - list of signals to be plotted
           'HYPNO'               - brain state annotation
           'SP', 'SP2'           - hippocampal or prefrontal EEG spectrogram
           'EEG', 'EEG2'         - raw hippocampal or prefrontal EEG signal
           'EMG', 'EMG2'         - raw EMG signals
           'EMG_AMP'             - amplitude of EMG signal
           'LFP'                 - filtered LFP signal
                                   * to plot P-wave detection threshold, add '_THRES'
                                   * to label detected P-waves, add '_ANNOT'
           'DFF'                 - DF/F signal
           'LSR'                 - laser stimulation train
           'AUDIO'               - audio stimulation train
	   
           e.g. PLOT = ['EEG', 'EEG2', 'LSR', 'LFP_THRES_ANNOT'] will plot both EEG 
           channels, the laser train, and the LFP signal with P-wave detection threshold 
           and labeled P-waves, in order from top to bottom.

    tstart, tend - time (s) into recording to start and stop plotting data
    ma_thr, ma_state - max duration and brain state for microarousals
    flatten_tnrem - brain state for transition sleep
    eeg_nbin, emg_nbin, lfp_nbin, dff_nbin - factors by which to downsample raw EEG, EMG, LFP, and DF/F signals
    highres - if True, plot high-resolution spectrogram; if False, plot standard SP (2.5 s time resolution)
    recalc_highres - if True, recalculate high-res spectrogram from EEG using $nsr_seg and $perc_overlap params
    nsr_seg, perc_overlap - set FFT bin size (s) and overlap (%) for spectrogram calculation
    pnorm - if > 0, normalize each spectrogram frequency by its mean power across the recording
    psmooth - method for spectrogram smoothing (1 element specifies convolution along X axis, 
                                                2 elements define a box filter for smoothing)
    fmax - maximum frequency in spectrogram
    vm - controls spectrogram saturation
    cmap - colormap to use for spectrogram plot
    ylims = optional list of y axis limits for each plot
    add_boxes - optional list of tuples specifying (start, end) time points (s) to highlight with red box
    @Returns
    None 
    """
    
    pplot = dict.fromkeys(PLOT, None)
    if len(ylims) != len(PLOT):
        ylims = ['']*len(PLOT)

    # load sampling rate
    sr = sleepy.get_snr(ppath, rec)
    
    # load EEG, get no. of Intan samples in recording
    EEG = so.loadmat(os.path.join(ppath, rec, 'EEG.mat'), squeeze_me=True)['EEG']
    nsamples = len(EEG)
    
    # get start and end indices for Intan data (EEG/EMG/LFP/DFF/LSR/AUDIO)
    intan_start = int(round(tstart*sr))
    if tend == -1:
        intan_end = nsamples
        tend = int(round(nsamples/sr))
    else:
        intan_end = int(round(tend*sr))
    
    # adjust Intan idx to properly translate to Fourier idx (SP/EMG_AMP)
    f_adjust = np.linspace(-(nsr_seg*sr/2), (nsr_seg*sr/2), len(EEG))
    # set initial dt and Fourier time bins
    dt = 2.5
    fourier_start = int(round(intan_start/sr/dt))
    fourier_end = int(round(intan_end/sr/dt))
    
    if 'SP' in PLOT:
        # load hippocampal spectrogram
        if not highres:
            SPEEG = so.loadmat(os.path.join(ppath, rec, 'sp_%s.mat' % rec))
            SP = SPEEG['SP']
            freq = SPEEG['freq'][0]
            t = SPEEG['t'][0]
            sp_dt = SPEEG['dt'][0][0]
            sp_nbin = sp_dt*sr
        # load/calculate hippocampal high-res spectrogram
        else:
            SP, freq, t, sp_dt, sp_nbin, M_dt = highres_spectrogram(ppath, rec, nsr_seg=nsr_seg, 
                                                                    perc_overlap=perc_overlap, 
                                                                    recalc_highres=recalc_highres, mode='EEG')
        # normalize/smooth and collect spectrogram
        SP = adjust_spectrogram(SP, pnorm=pnorm, psmooth=psmooth, freq=freq, fmax=fmax)
        fourier_start = int(round((intan_start+f_adjust[intan_start])/sp_nbin))
        fourier_end = int(round((intan_end+f_adjust[intan_end])/sp_nbin))
        SP_cut = SP[:, fourier_start:fourier_end]
        pplot['SP'] = SP_cut
    
    if 'SP2' in PLOT:
        # load prefrontal spectrogram
        if not highres:
            SPEEG2 = so.loadmat(os.path.join(ppath, rec, 'sp2_%s.mat' % rec))
            SP2 = SPEEG2['SP2']
            freq = SPEEG2['freq'][0]
            t = SPEEG2['t'][0]
            sp_dt = SPEEG2['dt'][0][0]
            sp_nbin = sp_dt*sr
        # load/calculate prefrontal high-res spectrogram
        else:
            SP2, freq, t, sp_dt, 
            sp_nbin, M_dt = highres_spectrogram(ppath, rec, nsr_seg=nsr_seg, perc_overlap=perc_overlap, 
                                                recalc_highres=recalc_highres, mode='EEG2')
        # normalize/smooth and collect spectrogram
        SP2 = adjust_spectrogram(SP2, pnorm=pnorm, psmooth=psmooth, freq=freq, fmax=fmax)
        fourier_start = int(round((intan_start+f_adjust[intan_start])/sp_nbin))
        fourier_end = int(round((intan_end+f_adjust[intan_end])/sp_nbin))
        SP2_cut = SP2[:, fourier_start:fourier_end]
        pplot['SP2'] = SP2_cut
    
    if 'HYPNO' in PLOT:
        # get colormap for brain state annotation
        hypno_cmap, vmin, vmax = hypno_colormap()
        M_dt = sleepy.load_stateidx(ppath, rec)[0]
        M_dt_cut = M_dt[int(round(tstart/2.5)) : int(round(tend/2.5))]
        # adjust and collect brain state annotation
        M_dt_cut = adjust_brainstate(M_dt_cut, dt, ma_thr=ma_thr, ma_state=ma_state, flatten_tnrem=flatten_tnrem)
        pplot['HYPNO'] = M_dt_cut
    
    if 'EMG_AMP' in PLOT:
        # load EMG spectrogram & calculate amplitude
        if round(sp_dt, 1) == 2.5:  # if standard SP was used, load standard mSP
            SPEMG = so.loadmat(os.path.join(ppath, rec, 'msp_%s.mat' % rec))
            mSP = SPEMG['mSP']
            mfreq = SPEMG['freq'][0]
            EMG_amp = get_emg_amp(mSP, mfreq)
        else:  # if high-res SP was used, load/calculate high-res mSP
            mSP, mfreq, mt, 
            mdt, mnbin, _ = highres_spectrogram(ppath, rec, nsr_seg=nsr_seg, perc_overlap=perc_overlap, 
                                                recalc_highres=recalc_highres, mode='EMG')
            if mdt == sp_dt:
                EMG_amp = get_emg_amp(mSP, mfreq)
            # return zeroes if mSP resolution doesn't match SP resolution
            else:
                EMG_amp = np.zeros((len(EEG)))
        # collect EMG amplitude
        EMG_amp_cut = EMG_amp[fourier_start : fourier_end]
        pplot['EMG_AMP'] = EMG_amp_cut
            
    if 'EEG' in PLOT:
        # EEG1 data already loaded
        # divide by 1000 to convert Intan data from uV to mV
        EEG_cut = EEG[intan_start:intan_end] / 1000
        EEG_cut_dn = downsample_vec(EEG_cut, eeg_nbin)
        pplot['EEG'] = EEG_cut_dn
    
    if 'EEG2' in PLOT:
        # load & collect EEG2 data
        EEG2 = so.loadmat(os.path.join(ppath, rec, 'EEG2.mat'), squeeze_me=True)['EEG2']
        EEG2_cut = EEG2[intan_start:intan_end] / 1000
        EEG2_cut_dn = downsample_vec(EEG2_cut, eeg_nbin)
        pplot['EEG'] = EEG_cut_dn
        
    if 'EMG' in PLOT:
        # load & collect EMG data        
        EMG = so.loadmat(os.path.join(ppath, rec, 'EMG.mat'), squeeze_me=True)['EMG']
        EMG_cut = EMG[intan_start:intan_end] / 1000
        EMG_cut_dn = downsample_vec(EMG_cut, emg_nbin)
        pplot['EMG'] = EMG_cut_dn
        
    if 'EMG2' in PLOT:
        # load & collect EMG2 data        
        EMG2 = so.loadmat(os.path.join(ppath, rec, 'EMG2.mat'), squeeze_me=True)['EMG2']
        EMG2_cut = EMG2[intan_start:intan_end] / 1000
        EMG2_cut_dn = downsample_vec(EMG2_cut, emg_nbin)
        pplot['EMG2'] = EMG2_cut_dn
    
    lfps = [i for i in PLOT if 'LFP' in i]
    for l in lfps:
        # load & collect LFP data
        LFP = so.loadmat(os.path.join(ppath, rec, 'LFP_processed.mat'), squeeze_me=True)['LFP_processed']
        LFP_cut = LFP[intan_start:intan_end] / 1000
        LFP_cut_dn = downsample_vec(LFP_cut, lfp_nbin)
        # collect LFP signal, P-wave detection threshold, and P-wave indices
        ldata = [LFP_cut_dn, [], []]
        if l != 'LFP':
            pwave_info = so.loadmat(os.path.join(ppath, rec, 'p_idx.mat'), squeeze_me=True)
            # P-wave detection threshold
            if 'THRES' in l:
                pthres = np.empty((len(LFP,)))
                if 'thres' in pwave_info.keys():
                    pthres[:] = -pwave_info['thres']
                    ldata[1] = pthres[intan_start:intan_end] / 1000
            # P-wave indices
            if 'ANNOT' in l:
                pidx = np.zeros((len(LFP,)))
                pi = pwave_info['p_idx']
                for i in pi:
                    pidx[i] = LFP[i]
                ldata[2] = pidx[intan_start:intan_end] / 1000
        pplot[l] = ldata
        
    if 'DFF' in PLOT:
        #load & collect DF/F data
        DFF = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['dff']*100
        DFF_cut = DFF[intan_start:intan_end]
        DFF_cut_dn = downsample_vec(DFF_cut, dff_nbin)
        pplot['DFF'] = DFF_cut_dn        
    
    if 'LSR' in PLOT:
        # load & collect laser stimulation vector
        LSR = sleepy.load_laser(ppath, rec)
        LSR_cut = LSR[intan_start:intan_end]
        pplot['LSR'] = LSR_cut
    
    if 'AUDIO' in PLOT:
        # load & collect audio stimulation vector
        AUDIO = load_audio(ppath, rec)
        AUDIO_cut = AUDIO[intan_start:intan_end]
        pplot['AUDIO'] = AUDIO_cut


    ###   GRAPHS   ###
    plt.ion()
    fig, axs = plt.subplots(nrows=len(PLOT), ncols=1, constrained_layout=True, sharex=True)
    if len(PLOT) == 1:
        axs = [axs]
    
    # create subplot for each item in $PLOT
    for i, data_type in enumerate(PLOT):
        data = pplot[data_type]
        ax = axs[i]
        y = ylims[i]
        # plot spectrogram
        if data_type == 'SP' or data_type == 'SP2':
            x = np.linspace(tstart, tend, data.shape[1])
            im = ax.pcolorfast(x, freq[np.where(freq <= fmax)[0]], data, cmap=cmap)
            if len(vm) == 2:
                im.set_clim(vm)
            ax.set_ylabel('Freq. (Hz)')
            plt.colorbar(im, ax=ax, pad=0.0)
        elif 'LFP' in data_type:
            # plot LFP signal
            ax.plot(np.linspace(tstart, tend, len(data[0])), data[0], color='black')
            # plot P-wave detection threshold
            if len(data[1]) > 0:
                ax.plot(np.linspace(tstart, tend, len(data[1])), data[1], color='green')
            # label detected P-waves
            if len(data[2]) > 0:
                x = np.linspace(tstart, tend, len(data[2]))
                ax.plot(x[np.where(data[2] != 0)[0]], data[2][np.where(data[2]!=0)[0]], 
                        color='red', marker='o', linewidth=0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if len(y) == 2:
                ax.set_ylim(y)
            ax.set_ylabel('LFP (mV)')
        else:
            x = np.linspace(tstart, tend, len(data))
            # plot brain state annotation
            if data_type == 'HYPNO':
                ax.pcolorfast(x, [0, 1], np.array([data]), vmin=vmin, vmax=vmax, cmap=hypno_cmap)
                ax.axes.get_yaxis().set_visible(False)
                
            # plot other data (EEG/EMG/EMG_AMP/DFF/LSR/AUDIO)
            else:
                ax.plot(x, data, color='black')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                if len(y) == 2:
                    ax.set_ylim(y)
                if data_type == 'DFF':
                    ax.set_ylabel('DF/F (%)')
                elif data_type == 'LSR':
                    ax.set_ylabel('Laser')
                elif data_type == 'AUDIO':
                    ax.set_ylabel('Audio')
                else:
                    ax.set_ylabel('mV')
    # draw boxes
    if len(add_boxes) > 0:
        draw_boxes(axs, add_boxes)
    # set plot title
    axs[0].set_title(f'{rec}: {tstart}s - {tend}s')
    axs[-1].set_xlabel('Time (s)')
    
    plt.show()

def draw_boxes(axs, coors):
    """
    Draw box(es) across vertically stacked subplots with shared x axes
    @Params
    axs - list of subplots
    coors - list of x coordinate pairs specifying (left edge, right edge) of each box
    @Returns
    None
    """
    line_kw = dict(color='red', linewidth=2, clip_on=False)
    box_t, box_b = axs[0].get_ylim()[1], axs[-1].get_ylim()[0]
    for coor in coors:
        # left/right lines
        box_l = coor[0]
        box_r = coor[1]
        # top/bottom lines
        axs[0].hlines(box_t, box_l, box_r, **line_kw)
        axs[-1].hlines(box_b, box_l, box_r, **line_kw)
        # connect lines
        line_l = matplotlib.patches.ConnectionPatch(xyA=[box_l,box_b], xyB=[box_l,box_t], 
                                                    coordsA='data', coordsB='data', 
                                                    axesA=axs[-1], axesB=axs[0], **line_kw)
        line_r = matplotlib.patches.ConnectionPatch(xyA=[box_r,box_b], xyB=[box_r,box_t], 
                                                    coordsA='data', coordsB='data', 
                                                    axesA=axs[-1], axesB=axs[0], **line_kw)
        axs[-1].add_artist(line_l)
        axs[-1].add_artist(line_r)

def get_unique_labels(ax):
    """
    Add legend of all uniquely labeled plot elements
    @Params
    ax - plot axis
    @Returns
    None
    """
    h,l = ax.get_legend_handles_labels()
    l_idx = list( dict.fromkeys([l.index(x) for x in l]) )
    legend = ax.legend(handles=[h[i] for i in l_idx], labels=[l[i] for i in l_idx], framealpha=0.3)
    ax.add_artist(legend)

def legend_mice(ax, mouse_names, symbol=''):
    """
    Add legend of all markers labeled with unique mouse names
    @Params
    ax - plot axis
    mouse_names - list of mouse names to include in legend
    symbol - if multiple marker types (e.g. '*' and 'o') are labeled with the same
             mouse name, the marker specified by $symbol is included in legend
    @Returns
    None
    """
    # find plot handles labeled by mouse names
    h,l = ax.get_legend_handles_labels()
    mouse_idx = [idx for idx, mouse in enumerate(l) if mouse in mouse_names]
    
    unique_mice = []
    # find preferred marker symbol in axes
    if symbol!='':
        symbol_idx = [idx for idx, handle in enumerate(h) if idx in mouse_idx and handle.get_marker() == symbol]
    else:
        symbol_idx = np.arange(0,len(h))
    for mname in list( dict.fromkeys(mouse_names) ):
        ms_symbol_idx = [si for si in symbol_idx if l[si] == mname]
        # use preferred symbol if it appears in the graph
        if len(ms_symbol_idx) > 0:
            unique_mice.append(ms_symbol_idx[0])
        else:
            unique_mice.append([mi for mi in mouse_idx if l[mi] == mname][0])
    # add legend of mouse names & markers 
    legend = ax.legend(handles=[h[i] for i in unique_mice], 
                       labels=[l[i] for i in unique_mice], framealpha=0.3)
    ax.add_artist(legend)

def legend_lines(ax, skip=[], loc=0):
    """
    Add legend of all uniquely labeled lines in plot
    @Params
    ax - plot axis
    skip - optional list of labels to exclude from legend
    loc - location of legend (0='best')
    @Returns
    None
    """
    # find handles & labels of lines in plot
    h,l = ax.get_legend_handles_labels()
    line_idx = [idx for idx,line in enumerate(h) if line in ax.lines]
    skip_idx = [idx for idx,lab in enumerate(l) if lab in skip]
    line_idx = [li for li in line_idx if li not in skip_idx]
    legend = ax.legend(handles=[h[i] for i in line_idx], labels=[l[i] for i in line_idx], framealpha=0.3, loc=loc)
    ax.add_artist(legend)
    
def legend_bars(ax, loc=0):
    """
    Add legend of all uniquely labeled bars in plot
    @Params
    ax - plot axis
    loc - location of legend (0='best')
    @Returns
    None
    """
    # find handles & labels of bars in plot
    h,l = ax.get_legend_handles_labels()
    bar_idx = [idx for idx,bar in enumerate(h) if bar in ax.containers]
    if len(bar_idx) > 0:
        legend = ax.legend(handles=[h[i] for i in bar_idx], labels=[l[i] for i in bar_idx], framealpha=0.3, loc=loc)
        ax.add_artist(legend)
    else:
        print('***No labeled bar containers found in these axes.')

def label_bars(ax, text=[], y_pos=[], dec=0, box=False):
    """
    Add text labels to bars in plot
    @Params
    ax - plot axis
    text - optional list of text labels for each bar
           * if empty list, label bars with y axis values
    y_pos - position of text on y axis
    dec - no. of decimals in bar value labels
    box - if True, draw box around text labels
    @Returns
    None
    """
    for i, bar in enumerate(ax.patches):
        # label bar with y axis value
        height = bar.get_height()
        if len(text) == 0:
            if dec==0:
                txt = int(height)
            else:
                txt = round(height, dec)
        # label bar with input text
        else:
            txt = text[i]
        # set position of text label on y axis
        x = bar.get_x() + bar.get_width()/2
        if len(y_pos)==0:
            y = bar.get_y() + height + (height/4)
        elif len(y_pos) == 1:
            y = bar.get_y() + y_pos[0]
        elif len(y_pos) == len(ax.patches):
            y = bar.get_y() + y_pos[i]
        # draw text
        if box:
            ax.text(x, y, txt, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
        else:
            ax.text(x, y, txt, ha='center', va='bottom')

def create_auto_title(brainstates=[], group_labels=[], add_lines=[]):
    """
    Generate descriptive plot title
    @Params
    brainstates - list of brain states in plot
    group_labels - list of groups in plot
    add_lines - list of extra text lines to add to title
    """
    states = {1:'REM', 2:'Wake', 3:'NREM', 4:'tNREM', 5:'failed-tNREM', 6:'Microarousals'}
    # clean data inputs
    if type(brainstates) != list:
        brainstates = [brainstates]
    if type(group_labels) != list:
        group_labels = [group_labels]
    if type(add_lines) != list:
        add_lines = [add_lines]
    brainstate_title = ''
    group_title = ''
    
    # create title for brainstate(s) in graph
    if len(brainstates) > 0:
        brainstate_names = []
        for b in brainstates:
            try:
                brainstate_names.append(states[int(b)])
            except:
                brainstate_names.append(b)
        # get rid of duplicate brainstate names and join into title
        brainstate_names = list(dict.fromkeys(brainstate_names).keys())
        brainstate_title = ' vs '.join(brainstate_names)
    # create title for group(s) in graph
    if len(group_labels) > 0:
        group_names = []
        for g in group_labels:
            try: group_names.append('Group ' + str(int(g)))
            except: group_names.append(g)
        # get rid of duplicate group names
        group_names = list(dict.fromkeys(group_names).keys())
        group_title = ' vs '.join(group_names)
    
    # arrange titles in coherent order
    if 'vs' in brainstate_title:
        if group_title == ''        : title = brainstate_title
        else                        : title = brainstate_title + ' (' + group_title + ')'
    else:
        if brainstate_title == ''   : title = group_title
        elif group_title == ''      : title = brainstate_title
        elif 'vs' in group_title    : title = group_title + ' (' + brainstate_title + ')'
        else                        : title = brainstate_title + ' (' + group_title + ')'
    # add custom lines to title
    if len(add_lines) > 0:
        for line in add_lines:
            title = line + '\n' + title
        if title.split('\n')[-1] == '':
            title = title[0:-1]
    return title