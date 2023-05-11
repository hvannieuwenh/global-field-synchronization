#!/usr/bin/env python3

# Calculating Global Field Synchronization for the Metabolic EEG dataset. 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import mne
import os
import scipy as sp
import scipy.signal
import pandas as pd
import re
import sklearn 
import math

from sklearn.decomposition import PCA
from scipy.signal import get_window
from GFS_analysis_functions import calc_GFS,draw_vector,plot_scatter_PCAcomps
from joblib import Parallel, delayed

sessions=['glu','bhb']
runs=['1','2']
eyes=['EC','EO']

output_dir = '/shared/home/jolien/GFS_outputs_eeg_metabolic/'

#epoch_size = 2 # in seconds, defined instead in loop below

for eye in eyes:
    data_dir='/shared/datasets/private/eeg_metabolic/'

    subjects=[]
    for i in [f.path for f in os.scandir(data_dir) if f.is_dir()]:
        j=i.replace(data_dir,'')
        subjects.append(j)
    subjects=(sorted(subjects))[0:len(subjects)-2] # removes sub-A and sub-B
    
    subjects=['sub-044']
    print(subjects)
    for sub in subjects:
        for ses in sessions:
            for run in runs:
                for epoch_size in [2,10]:
                    try:
                        fname=sub+'_ses-'+ses+'_task-rest'+eye+'_run-'+run+'.set'
                        raw=mne.io.read_raw_eeglab(data_dir+fname,preload=True)

                        channels = raw.ch_names
                        sampling_freq = raw.info['sfreq']
                        #eeg_raw = raw.filter(l_freq, h_freq)
                        amps, times = raw.get_data(return_times=True,picks='eeg')
                        num_timepts = len(raw)
                        # get channel regions and colors for plotting
                        chan_info_df = pd.read_csv('chan_info.csv')
                        chan_info_df = chan_info_df[chan_info_df['Name'].isin(channels)]

                        df_out = pd.DataFrame({'Frequency':[],'GFS':[],'Epoch Start':[],'Epoch End':[]})

                        def parallel_GFS(curr_freq,df):
                            freqs, GFS_vals, win_starts, win_ends = calc_GFS(amps,chan_info_df,times,epoch_size,curr_freq)
                            df_append = pd.DataFrame({'Frequency':freqs,'GFS':GFS_vals,'Epoch Start':win_starts,'Epoch End':win_ends})
                            df = df.append(df_append)
                            return df

                        list_of_df = Parallel(n_jobs=20)(delayed(parallel_GFS)(curr_freq,df_out) for curr_freq in np.arange(1,40,0.1))
                        
                        for i in range(len(list_of_df)):
                            df_out=df_out.append(list_of_df[i])

                        df_out.to_csv(output_dir+re.sub('.set','.csv',fname)+'_epoch{}s'.format(epoch_size))
                        
                    except Exception as e:
                        print('***ERROR***')
                        print(e)
                        continue
                

