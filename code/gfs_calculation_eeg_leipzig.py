#!/usr/bin/env python3

# Calculating Global Field Synchronization for the Leipzig EEG dataset. 

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


eyes=['EC','EO']


output_dir = '/shared/home/jolien/GFS_outputs_eeg_leipzig/'

epoch_size = 2 # in seconds

data_dir='/shared/datasets/private/leipzig_eeg_data/EEG_Preprocessed/'

subjects=[]
for i in [f.path for f in os.scandir(data_dir)]:
    if i.endswith(".set"):
        j=i.replace(data_dir,'')
        subjects.append(j[0:10])
subjects=(sorted(subjects))

print(subjects)

for eye in eyes:
    for sub in subjects:
        try:
            sub_fname=sub+'_'+eye
            fname=data_dir+sub_fname+'.set'
            raw=mne.io.read_raw_eeglab(fname,preload=True)

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

            list_of_df = Parallel(n_jobs=40)(delayed(parallel_GFS)(curr_freq,df_out) for curr_freq in np.arange(1,40,0.1))
            
            for i in range(len(list_of_df)):
                df_out=df_out.append(list_of_df[i])

            df_out.to_csv(output_dir+sub_fname+'.csv')
            
        except Exception as e:
            print('*** ERROR filename: ***')
            print(fname)
            print(e)
            continue
                

