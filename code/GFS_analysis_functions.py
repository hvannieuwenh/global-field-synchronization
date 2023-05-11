#!/usr/bin/env python3

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

# perform FFT using parameters from this paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5098962/
# returns complex value from FFT for each channel at particular frequency 

### 

#epoch_size given in seconds

def calc_GFS(amps,chan_info_df,times,epoch_size,curr_freq):

    frequencies,GFS_values,win_starts,win_ends=[],[],[],[]
    num_timepts = int(epoch_size/times[1]-times[0])
    max_epoch_num = math.floor(len(times)/num_timepts)
    #print('Working on {} Hz'.format(str(curr_freq)))

    for epoch in range(max_epoch_num):
        min_timept = epoch*num_timepts
        max_timept = (epoch+1)*num_timepts

        ffts_real_imag=[]
        for chan in range(len(chan_info_df)):
            signal=amps[chan][min_timept:max_timept]
            m = len(signal)
            tukey_window = sp.signal.windows.tukey(m, alpha=0.2, sym=True)
            n = 4096
            signal_fft = np.fft.rfft(signal*tukey_window, n=n)
            freqs = np.fft.rfftfreq(n, d=times[1]-times[0])

            # grabs complex value of FFT associated with wanted frequency
            ffts_real_imag.append(signal_fft[int(curr_freq/(freqs[1]-freqs[0]))])

        needed_fft=np.array(ffts_real_imag)

        # extract real part 
        x = needed_fft.real
        # extract imaginary part
        y = needed_fft.imag

        zipped=np.array(list(zip(x,y)))*1000  #scaled to avoid bug in matplotlib scatter

        # get first two principal components of 
        pca = PCA(n_components=2)
        pca.fit(np.array(zipped))

        evals = pca.explained_variance_

        GFS = (abs(evals[1]-evals[0]))/(evals[1]+evals[0])
        frequencies.append(curr_freq)
        GFS_values.append(GFS)
        win_starts.append(min_timept)
        win_ends.append(max_timept)

    #plot_scatter_PCAcomps(pca,zipped,curr_freq)
    return frequencies, GFS_values, win_starts, win_ends

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
    
def plot_scatter_PCAcomps(pca,zipped,curr_freq):
    plt.figure()
    
    plt.scatter(zipped[:,0],zipped[:,1],c=chan_info_df['Color'])
    
    # plot PCA component vectors
    for length, vector in zip(pca.explained_variance_, pca.components_):
        print(length)
        print(vector)
        v = vector * 3 * np.sqrt(length)
        draw_vector(pca.mean_, pca.mean_ + v)
        
    #print('GFS = {}'.format(GFS))
    #plt.scatter(pca.mean_,c='red')
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    #plt.xticks([])
    #plt.yticks([])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #plt.legend(by_label.values(), by_label.keys(),bbox_to_anchor=(1.3, 1.03))
    plt.title('Frequency = {} Hz'.format(curr_freq))
    plt.axis('equal')
    #plt.tight_layout()
    #plt.savefig('gfs_animation/sub-022_GFS_{}Hz.pdf'.format(str(curr_freq)))
    plt.show()
    
    
