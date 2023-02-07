#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:14:46 2023

@author: nahuel
"""
from libb import *

def main():
    print("Hola Mundo")
    m = scipy.io.loadmat('data/BCICIV_calib_ds1d.mat', struct_as_record=True)
    
    # SciPy.io.loadmat does not deal well with Matlab structures, resulting in lots of
    # extra dimensions in the arrays. This makes the code a bit more cluttered
    
    path = "data/"
    filename = "test5"
    
    sample_rate = m['nfo']['fs'][0][0][0][0]
    EEG = m['cnt'].T
    nchannels, nsamples = EEG.shape
    
    channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]
    event_onsets = m['mrk'][0][0][0]
    event_codes = m['mrk'][0][0][1]
    labels = np.zeros((1, nsamples), int)
    labels[0, event_onsets] = event_codes
    
    cl_lab = [s[0] for s in m['nfo']['classes'][0][0][0]]   
    freq=sample_rate
    #Se carga la matriz de datos
    #data_cnt=data_cnt.transpose()
    #print("data_cnt: ", data_cnt.shape)
    
    #EEG = EEG*0.1
    EEG = EEG / 1000000
    
    print(EEG)
    
    #Se carga los nombre de los caneles
    info = mne.create_info(channel_names, freq, 'eeg')
    raw = mne.io.RawArray(EEG, info, first_samp=0, copy='auto', verbose='critical')
    raw.save(path + filename + "_eeg.fif", overwrite=True)
    
    #print((event_onsets[0]))
    #print((event_codes[0])+1)
    event_onsets = event_onsets[0]
    event_codes = event_codes[0]
    event_codes = (event_codes + 1) / 2
    
    events = np.zeros((len(event_onsets) , 3), int)
    events[:, 0] = event_onsets.astype(int)
    events[:, 2] = event_codes.astype(int)
    
    mne.write_events(path + filename + "-eve.fif", events, overwrite=True)
    
    """
    nclasses = len(cl_lab)
    nevents = len(event_onsets)
        
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
    
    print(trials)
    #sample_rate
    #cl_lab
    #channel_names
    """
    
    """
    #convertir de uV -> V
    data = data / 1000000
    raw = loadDatos(data, 'ch_names.txt')
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage)
    raw.save(path + "/data_eeg.fif", overwrite=True)
    """
         
if __name__ == "__main__":
    main()