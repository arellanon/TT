#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:14:46 2023

@author: nahuel
"""
from libb import *

def main():
    print("Hola Mundo")
    #m = scipy.io.loadmat('data/BCICIV_calib_ds1d.mat', struct_as_record=True)
    
    # SciPy.io.loadmat does not deal well with Matlab structures, resulting in lots of
    # extra dimensions in the arrays. This makes the code a bit more cluttered
    
    print("Inicio...")
    mat = scipy.io.loadmat('data/VICPER/Database-MIOpenBCI/S02.mat')
    data  = mat["S01"][0][0][0][0][0]
    event = mat["S01"][0][0][0][0][1]
    #print(data)
    print(data.shape)
    print(event.shape)
    
    
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
    
if __name__ == "__main__":
    main()