#!/usr/bin/env python3
# coding: utf-8

# Calculating Global Field Synchronization for EEG simulated using the local field potential (LFP) outputs of the Larter-Breakspear neural mass model.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import os
import scipy as sp
import scipy.signal
import pandas as pd
import re
import sklearn 
import math
import mat73
import glob

from sklearn.decomposition import PCA
from scipy.signal import get_window
from scipy import io
from GFS_analysis_functions import calc_GFS
from joblib import Parallel, delayed
from os import listdir
from os.path import isfile, join

leadfield = sp.io.loadmat('mean_leadfield_svd.mat')

for coupling in ['035_dense_sample']:
    for param_var in ['aee','aei','eca','ek','ena','gca','gk','gna']:
        lb_data_dir = '/shared/datasets/public/larter_breakspear_lfp/coupling_{}/{}/'.format(coupling,param_var)

        coupling_dir = '/shared/datasets/private/GFS_outputs_LB_fineres/coupling_{}/'.format(coupling)
        output_dir = '/shared/datasets/private/GFS_outputs_LB_fineres/coupling_{}/{}/'.format(coupling,param_var)

        if os.path.isdir(coupling_dir) == False:
            os.mkdir(coupling_dir)

        if os.path.isdir(output_dir) == False:
            os.mkdir(output_dir)

        for epoch_size in [10]:
            for mat_file in glob.glob(lb_data_dir + '*.mat'):
                try:
                    print('*** Now calculating GFS for {}'.format(mat_file))
                    sampling_freq = 1000
                    
                    lb_mat = mat73.loadmat(mat_file)
                    eeg_signals = np.dot(leadfield['leadfield'],lb_mat['excitatory_signals'])
                    times = np.arange(0,len(eeg_signals[0]))/sampling_freq
                    
                    num_timepts = len(times)

                    df_out = pd.DataFrame({'Frequency':[],'GFS':[],'Epoch Start':[],'Epoch End':[]})

                    def parallel_GFS(curr_freq,df):
                        freqs, GFS_vals, win_starts, win_ends = calc_GFS(eeg_signals,chan_info_df,times,epoch_size,curr_freq)
                        df_append = pd.DataFrame({'Frequency':freqs,'GFS':GFS_vals,'Epoch Start':win_starts,'Epoch End':win_ends})
                        df = df.append(df_append)
                        return df

                    list_of_df = Parallel(n_jobs=40)(delayed(parallel_GFS)(curr_freq,df_out) for curr_freq in np.arange(1,100,0.1))

                    for i in range(len(list_of_df)):
                        df_out=df_out.append(list_of_df[i])

                    df_out.to_csv(output_dir+'{}_epoch{}s.csv'.format(re.sub(lb_data_dir,'',mat_file)[:-4],epoch_size))
                    
                    GFS = []
                    for curr_freq in df_out['Frequency'].unique():
                        GFS.append(np.mean(df_out[df_out['Frequency']==curr_freq]['GFS']))

                    print('*** GFS output saved for {}'.format(mat_file))

                except Exception as e:
                    print('*** ERROR ***')
                    print(e)
                    continue



