#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 16:22:49 2021

@author: nahuel
"""
from libb import *
import numpy as np
import scipy.io
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations

def main():
    trials1, info1 = get_epoch("epochs/", "BCICIV_calib_ds1d")
    print("trials1: ", trials1['left'].shape )
    print("info1: ", type(info1["channel_names"]) )
    
    trials2, info2 = get_data5("data/VICPER/", "S10")
    
    print("trials2: ", trials2['left'].shape )
    print("trials2: ", trials2['right'].shape )
    print("info2: ", type(info2["channel_names"]) )
    
    #print(trials1['left'][0])
    
    #trials1, info1 = get_epoch("epochs/", "Experiment5")
    #trials2, info2 = get_data()
    #trials2, info2 = get_data2()
    #trials2, info2 = get_epoch("epochs/", "Experiment6v4")
    #trials2, info2 = get_epoch("epochs/", "Experiment7v1")
    #trials2, info2 = get_epoch("epochs/", "Experiment4")
    #info2["channel_names"] = ['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2']
    
    #show_info(trials1, info1)
    #report_psd(trials1, info1)
    #calculateML(trials1, info1)
    #print(trials1)
    

    show_info(trials2, info2)
    #report_psd(trials2, info2)
    calculateML(trials2, info2)
    #print(trials2)

    
def show_info(trials, info):
    sample_rate = info['sample_rate']
    cl_lab = info['cl_lab'] 
    channel_names = info['channel_names']
    
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]
    
    ntrials = trials[cl1].shape[0]
    nchannels = trials[cl1].shape[1]
    nsamples = trials[cl1].shape[2]
    nchannels = len(channel_names)
    
    print('Sample rate:', sample_rate)
    print('Number of channels:', nchannels)
    print('Channel names:', channel_names)
    print('Class labels:', cl_lab)
    print('Shape ', cl1,' :', trials[cl1].shape)
    print('Shape ', cl2,' :', trials[cl2].shape)

## Funciones reporte    
def report_psd(trials, info):
    sample_rate = info['sample_rate']
    cl_lab = info['cl_lab'] 
    channel_names = info['channel_names']
    
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]
    
    ntrials = trials[cl1].shape[0]
    nchannels = trials[cl1].shape[1]
    nsamples = trials[cl1].shape[2]
    nchannels = len(channel_names)
    
    # PSD epoch
    psd_l, freqs = psd(trials[cl1], sample_rate)
    psd_r, freqs = psd(trials[cl2], sample_rate)
    trials_PSD = {cl1: psd_l, cl2: psd_r}
    
    plot_psd(
        trials_PSD,
        freqs,
        [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
        chan_lab=['left', 'center', 'right']
        #maxy=500
    )
    
    # PSD epoch bandpass
    trials_filt = {cl1: bandpass(trials[cl1], 8, 15, sample_rate),
                   cl2: bandpass(trials[cl2], 8, 15, sample_rate)}
    
    psd_l, freqs = psd(trials_filt[cl1], sample_rate)
    psd_r, freqs = psd(trials_filt[cl2], sample_rate)
    trials_PSD = {cl1: psd_l, cl2: psd_r}
    
    plot_psd(
        trials_PSD,
        freqs,
        [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
        chan_lab=['left', 'center', 'right']
        #maxy=300
    )
    
    # PSD epoch csp
    W = csp(trials_filt[cl1], trials_filt[cl2])
    
    trials_csp = {cl1: apply_mix(W, trials_filt[cl1]),
                  cl2: apply_mix(W, trials_filt[cl2])}
    
    psd_l, freqs = psd(trials_csp[cl1], sample_rate)
    psd_r, freqs = psd(trials_csp[cl2], sample_rate)
    trials_PSD = {cl1: psd_l, cl2: psd_r}

    plot_psd(
        trials_PSD, 
        freqs, 
        [0, nchannels // 2 ,-1],
        chan_lab=['first component', 'middle component', 'last component']
        #maxy=0.75 
    )
    
    #Logvar
    trials_logvar = {cl1: logvar(trials_filt[cl1]),
                     cl2: logvar(trials_filt[cl2])}
    
    plot_logvar(trials_logvar, nchannels, cl_lab)

    trials_csp_logvar = {cl1: logvar(trials_csp[cl1]),
                     cl2: logvar(trials_csp[cl2])}
    
    plot_logvar(trials_csp_logvar, nchannels, cl_lab)
    
    # Scatterplot
    plot_scatter(trials_csp_logvar[cl1], trials_csp_logvar[cl2], cl_lab)    
    
def calculateML(trials, info):
    sample_rate = info['sample_rate']
    cl_lab = info['cl_lab'] 
    channel_names = info['channel_names']
    
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]
    
    ntrials = trials[cl1].shape[0]
    nchannels = trials[cl1].shape[1]
    nsamples = trials[cl1].shape[2]
    nchannels = len(channel_names)
    
    # PSD epoch bandpass
    trials_filt = {cl1: bandpass(trials[cl1], 8, 15, sample_rate),
                   cl2: bandpass(trials[cl2], 8, 15, sample_rate)}
    
    train, test = prepareData(trials_filt, cl_lab)
    
    #print("train: ", train['left']) 
    #print("test: ", test['left'].shape) 
    
    W,b = train_lda(train[cl1], train[cl2])
    print('W:', W)
    print('b:', b)
        
    # Scatterplot
    #plot_scatter(trials_logvar[cl1], trials_logvar[cl2], cl_lab)
    
    # Scatterplot like before
    plot_scatter(train[cl1], train[cl2], cl_lab)
    
    # Calculate decision boundary (x,y)
    x = np.arange(-5, 1, 0.1)
    y = (b - W[0]*x) / W[1]
    
    # Plot the decision boundary
    plt.plot(x, y, linestyle='--', linewidth=2, color='k')
    #plt.xlim(-5, 1)
    #plt.ylim(-2.2, 1)
    
    plot_scatter(test[cl1], test[cl2], cl_lab)
    #title('Test data')
    plt.plot(x,y, linestyle='--', linewidth=2, color='k')
    #plt.xlim(-5, 1)
    #plt.ylim(-2.2, 1)
    

    # Print confusion matrix
    conf = np.array([
        [(apply_lda(test[cl1].T, W, b) == 1).sum(), (apply_lda(test[cl2].T, W, b) == 1).sum()],
        [(apply_lda(test[cl1].T, W, b) == 2).sum(), (apply_lda(test[cl2].T, W, b) == 2).sum()],
    ])
    
    print('Confusion matrix:')
    print(conf)
    print()
    print('Accuracy: %.3f' % (np.sum(np.diag(conf)) / float(np.sum(conf))))
    print()
    
def prepareData(trials_filt, cl_lab):
    train_percentage = 0.8
    cl1=cl_lab[0]
    cl2=cl_lab[1]
    
    # Calculate the number of trials for each class the above percentage boils down to
    ntrain_l = int(trials_filt[cl1].shape[0] * train_percentage)
    ntrain_r = int(trials_filt[cl2].shape[0] * train_percentage)
    ntest_l = trials_filt[cl1].shape[0] - ntrain_l
    ntest_r = trials_filt[cl2].shape[0] - ntrain_r
    
    
    # Splitting the frequency filtered signal into a train and test set
    train = {cl1: trials_filt[cl1][:ntrain_l,:,:],
             cl2: trials_filt[cl2][:ntrain_r,:,:]}
    
    test = {cl1: trials_filt[cl1][ntrain_l:,:,:],
            cl2: trials_filt[cl2][ntrain_r:,:,:]}
    
    # Train the CSP on the training set only
    W = csp(train[cl1], train[cl2])
    
    print("W:", W.shape)
    
    # Apply the CSP on both the training and test set
    train[cl1] = apply_mix(W, train[cl1])
    train[cl2] = apply_mix(W, train[cl2])
    test[cl1] = apply_mix(W, test[cl1])
    test[cl2] = apply_mix(W, test[cl2])
    
    # Select only the first and last components for classification
    comp = np.array([0,-1])
    train[cl1] = train[cl1][:,comp,:]
    train[cl2] = train[cl2][:,comp,:]
    test[cl1] = test[cl1][:,comp,:]
    test[cl2] = test[cl2][:,comp,:]
    
    # Calculate the log-var
    train[cl1] = logvar(train[cl1])
    train[cl2] = logvar(train[cl2])
    test[cl1] = logvar(test[cl1])
    test[cl2] = logvar(test[cl2])
    
    return train, test
    

### Funciones get_data

def get_data():
    m = scipy.io.loadmat('data/BCICIV_calib_ds1d.mat', struct_as_record=True)
    
    # SciPy.io.loadmat does not deal well with Matlab structures, resulting in lots of
    # extra dimensions in the arrays. This makes the code a bit more cluttered
    
    sample_rate = m['nfo']['fs'][0][0][0][0]
    EEG = m['cnt'].T
    nchannels, nsamples = EEG.shape
    
    channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]
    event_onsets = m['mrk'][0][0][0]
    event_codes = m['mrk'][0][0][1]
    labels = np.zeros((1, nsamples), int)
    labels[0, event_onsets] = event_codes
    
    cl_lab = [s[0] for s in m['nfo']['classes'][0][0][0]]

    nclasses = len(cl_lab)
    nevents = len(event_onsets)
        
    # Dictionary to store the trials in, each class gets an entry
    trials = {}
    info = {}
    
    # The time window (in samples) to extract for each trial, here 0.5 -- 2.5 seconds
    win = np.arange(int(0.5*sample_rate), int(2.5*sample_rate))
    
    # Length of the time window
    nsamples = len(win)
    
    # Loop over the classes (left, right)
    for cl, code in zip(cl_lab, np.unique(event_codes)):
        
        # Extract the onsets for the class
        cl_onsets = event_onsets[event_codes == code]
        
        # Allocate memory for the trials
        trials[cl] = np.zeros((len(cl_onsets),nchannels,nsamples))
        
        # Extract each trial
        for i, onset in enumerate(cl_onsets):
            trials[cl][i,:,:] = EEG[:, win+onset]
      
    # Some information about the dimensionality of the data (channels x time x trials)
    
    info['sample_rate'] = sample_rate
    info['cl_lab'] = cl_lab
    info['channel_names'] = channel_names
    return trials, info

def get_data2():
    runs = [3, 7, 11]  #Motor execution: left vs right hand

    raw_fnames=[]
    for subject in range(1, 10):
        raw_fnames.extend( eegbci.load_data(subject,runs= runs) )
    
    #fnames = eegbci.load_data(subject=1, runs=runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    #raw = concatenate_raws([read_raw_edf(f, preload=True) for f in fnames])
    
    raw.rename_channels(lambda x: x.strip('.'))  # remove dots from channel names
    
    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    
    channel_names = raw.info['ch_names']


    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    
    tmin, tmax = 0.5, 2.5
    event_ids = dict(left=2, right=3)  # map event IDs to tasks
    
    #epochs = mne.Epochs(raw, events, event_ids, tmin - 0.5, tmax + 0.5, baseline=None, preload=True)
    
    epochs = Epochs(raw, events, event_ids, tmin, tmax, proj=True,
                    baseline=None, preload=True)
    
    #epochs.save("epochs/" + "eegbci" + "-epo.fif", overwrite=True)
    
    trials = {}
    epochs2=epochs[epochs.events[:,2]==2]
    epochs3=epochs[epochs.events[:,2]==3]
    
    sample_rate = raw.info['sfreq']
    
    cl_lab=['left', 'right']
    
    trials[cl_lab[0]] = epochs2.get_data(units='uV')
    trials[cl_lab[1]] = epochs3.get_data(units='uV')
    
    info = {}
    info['sample_rate'] = sample_rate
    info['cl_lab'] = cl_lab
    info['channel_names'] = channel_names
    
    return trials, info

def get_data3():
    path="../../LSL-BCI/DATA/Experiment_5/Data/T1/"
    raw = mne.io.read_raw_fif(path + "/data_eeg.fif", preload=True)
    events = mne.read_events(path + "/data-eve.fif")
    events = events[:39]

    trials = {}    
    channel_names = raw.info['ch_names']
    sample_rate = raw.info['sfreq']
    cl_lab=['left', 'right']
    
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    tmin, tmax = 0.5, 2.5
    #event_ids = dict(left=2, right=3)  # map event IDs to tasks
    event_id = {'left': 0, 'right': 1}
    
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, baseline=None, preload=True)
    
    epochs0=epochs[epochs.events[:,2]==0]
    epochs1=epochs[epochs.events[:,2]==1]
       
    trials[cl_lab[0]] = epochs0.get_data(units='uV')
    trials[cl_lab[1]] = epochs1.get_data(units='uV')
    
    return trials, sample_rate, cl_lab, channel_names


def get_data4(path, filename):
    #path="../../LSL-BCI/DATA/Experiment_5/Data/T1/"
    raw = mne.io.read_raw_fif(path + filename + "_eeg.fif", preload=True)
    events = mne.read_events(path + filename + "-eve.fif")
    
    #raw.plot(scalings=None, n_channels=8, events=events)

    trials = {}
    info = {}
    channel_names = raw.info['ch_names']
    sample_rate = raw.info['sfreq']
    cl_lab=['left', 'right']
    
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    tmin, tmax = 0.5, 2.5
    #event_ids = dict(left=2, right=3)  # map event IDs to tasks
    event_id = {'left': 0, 'right': 1}
    
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, baseline=None, preload=True)
    
    epochs.save(path + filename + "-epo.fif", overwrite=True)
    
    epochs0=epochs[epochs.events[:,2]==0]
    epochs1=epochs[epochs.events[:,2]==1]
       
    trials[cl_lab[0]] = epochs0.get_data(units='uV')
    trials[cl_lab[1]] = epochs1.get_data(units='uV')
    
    info['sample_rate'] = sample_rate
    info['cl_lab'] = cl_lab
    info['channel_names'] = channel_names
    return trials, info

def get_data5(path, filename):
    #path="../../LSL-BCI/DATA/Experiment_5/Data/T1/"
    mat = scipy.io.loadmat(path + filename + ".mat")
    data  = mat[filename][0][0][0][0][0]
    events_2d = mat[filename][0][0][0][0][1]

    events = np.squeeze(events_2d)
    print(events)
    valor = events[:]==1
    print(valor[0])
    #print(data)
    #print(data.shape)
    #print(event.shape)
    #reshape = ( 150, 15, 501)
    #epochs = data.reshape(160, 15, 501)
    epochs = np.transpose(data)
    print(epochs.shape)
    
    trials = {}
    info = {}
    channel_names = ['Pz', 'Cz', 'T6', 'T4', 'F8', 'P4', 'C4', 'F4', 'Fz', 'T5', 'T3', 'F7', 'P3', 'C3', 'F3']
    sample_rate = 125
    cl_lab=['left', 'right']
    
    epochs0=epochs[events[:]==1]
    epochs1=epochs[events[:]==2]
       
    trials[cl_lab[0]] = epochs0
    trials[cl_lab[1]] = epochs1
    
    info['sample_rate'] = sample_rate
    info['cl_lab'] = cl_lab
    info['channel_names'] = channel_names
    return trials, info

def get_epoch(path, filename):
    #event_id = {'left': 0, 'right': 1}
    #raw = mne.io.read_raw_fif(path + filename + "_eeg.fif", preload=True)
    #events = mne.read_events(path + filename + "-eve.fif")
       
    epochs=mne.read_epochs(path + filename + "-epo.fif", proj=True, preload=True, verbose=None)
    trials = {}
    info = {}
    channel_names = epochs.info['ch_names']
    sample_rate = epochs.info['sfreq']
    cl_lab=['left', 'right']
    
    epochs0=epochs[epochs.events[:,2]==0]
    epochs1=epochs[epochs.events[:,2]==1]
    
    list_channel=['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2']
    
    trials[cl_lab[0]] = epochs0.get_data(units='uV', picks=list_channel)
    trials[cl_lab[1]] = epochs1.get_data(units='uV', picks=list_channel)
    
    info['sample_rate'] = sample_rate
    info['cl_lab'] = cl_lab
    #info['channel_names'] = channel_names
    info['channel_names'] = list_channel
    
    return trials, info

if __name__ == "__main__":
    main()