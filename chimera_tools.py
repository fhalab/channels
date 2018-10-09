from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import os
import pandas as pd
import pickle



def chimera_code2seq_convert(file_c,file_n,df_data):
    # Load each position file as a df
    df_c = pd.read_csv(file_c, sep=' ', 
                    names = ['chimera', 'E', 'm', 'seq'])
    df_n = pd.read_csv(file_n, sep=' ', 
                    names = ['chimera', 'E', 'm', 'seq'])
    
    seq_input = []
    for i in df_data.block_k:
        if i[0] == 'c':
            seq_input.append(df_c[df_c.chimera == i].seq.values[0])
        elif i[0] == 'n':
            seq_input.append(df_n[df_n.chimera == i].seq.values[0])
            
    df_data['seq'] = seq_input
    
    return df_data
    
def normalize_(data):
    """
    Normalize data by subtracting the mean and dividing by the std
    Also, make positive
    """
    return (data - np.mean(data)) / np.std(data)
    
def un_normalize_(norm_data, data):
    """
    Normalize data by subtracting the mean and dividing by the std
    Also, make positive
    """
    return norm_data*np.std(data) + np.mean(data)