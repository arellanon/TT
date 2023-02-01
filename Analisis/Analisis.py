#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 16:22:49 2021

@author: nahuel
"""
from libb import *
import numpy as np
import scipy.io
        
def main():
    print("hola mundo")
    
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
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]
    nclasses = len(cl_lab)
    nevents = len(event_onsets)
    
    print('Shape of EEG:', EEG.shape)
    print('Sample rate:', sample_rate)
    print('Number of channels:', nchannels)
    print('Channel names:', channel_names)
    print('Number of events:', len(event_onsets))
    print('Event codes:', np.unique(event_codes))
    print('Class labels:', cl_lab)
    print('Number of classes:', nclasses)

    
    # Dictionary to store the trials in, each class gets an entry
    trials = {}
    
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
    print('Shape of trials[cl1]:', trials[cl1].shape)
    print('Shape of trials[cl2]:', trials[cl2].shape)
    print('Number of samples: ', trials[cl2].shape[2])
    
    # Apply the function
    psd_l, freqs = psd(trials[cl1], sample_rate)
    psd_r, freqs = psd(trials[cl2], sample_rate)
    trials_PSD = {cl1: psd_l, cl2: psd_r}
    
    plot_psd(
        trials_PSD,
        freqs,
        [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
        chan_lab=['left', 'center', 'right'],
        maxy=500
    )
    
    # Apply the function
    trials_filt = {cl1: bandpass(trials[cl1], 8, 15, sample_rate),
                   cl2: bandpass(trials[cl2], 8, 15, sample_rate)}
    
    
    psd_l, freqs = psd(trials_filt[cl1], sample_rate)
    psd_r, freqs = psd(trials_filt[cl2], sample_rate)
    trials_PSD = {cl1: psd_l, cl2: psd_r}
    
    plot_psd(
        trials_PSD,
        freqs,
        [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
        chan_lab=['left', 'center', 'right'],
        maxy=300
    )
    
    # Apply the function
    trials_logvar = {cl1: logvar(trials_filt[cl1]),
                     cl2: logvar(trials_filt[cl2])}
    
    plot_logvar(trials_logvar, nchannels, cl_lab)
    
    
    # Apply the functions
    W = csp(trials_filt[cl1], trials_filt[cl2])
    
    trials_csp = {cl1: apply_mix(W, trials_filt[cl1]),
                  cl2: apply_mix(W, trials_filt[cl2])}
    
    
    trials_logvar = {cl1: logvar(trials_csp[cl1]),
                     cl2: logvar(trials_csp[cl2])}
    
    plot_logvar(trials_logvar, nchannels, cl_lab)
    
    
    psd_l, freqs = psd(trials_csp[cl1], sample_rate)
    psd_r, freqs = psd(trials_csp[cl2], sample_rate)
    trials_PSD = {cl1: psd_l, cl2: psd_r}

    plot_psd(trials_PSD, freqs, [0,28,-1], chan_lab=['first component', 'middle component', 'last component'], maxy=0.75 )
    
    plot_scatter(trials_logvar[cl1], trials_logvar[cl2], cl_lab)
    
    ##########################################################################
    # IA
    # Percentage of trials to use for training (50-50 split here)
    ##########################################################################
    train_percentage = 0.5 
    
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
    
    W,b = train_lda(train[cl1], train[cl2])
    print('W:', W)
    print('b:', b)
    # Scatterplot like before
    plot_scatter(train[cl1], train[cl2], cl_lab)
    
    # Calculate decision boundary (x,y)
    x = np.arange(-5, 1, 0.1)
    y = (b - W[0]*x) / W[1]
    
    # Plot the decision boundary
    plt.plot(x,y, linestyle='--', linewidth=2, color='k')
    plt.xlim(-5, 1)
    plt.ylim(-2.2, 1)
    
    plot_scatter(test[cl1], test[cl2], cl_lab)
    #title('Test data')
    plt.plot(x,y, linestyle='--', linewidth=2, color='k')
    plt.xlim(-5, 1)
    plt.ylim(-2.2, 1)
    
    
    # Print confusion matrix
    conf = np.array([
        [(apply_lda(test[cl1].T, W, b) == 1).sum(), (apply_lda(test[cl2].T, W, b) == 1).sum()],
        [(apply_lda(test[cl1].T, W, b) == 2).sum(), (apply_lda(test[cl2].T, W, b) == 2).sum()],
    ])
    
    print('Confusion matrix:')
    print(conf)
    print()
    print('Accuracy: %.3f' % (np.sum(np.diag(conf)) / float(np.sum(conf))))
    
if __name__ == "__main__":
    main()