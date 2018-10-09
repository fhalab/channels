from __future__ import division
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pickle


def one_hot_seq(seq_input):
    # make amino acid directory
    my_dict = {'-':0, 'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,\
                'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,\
                'V':18,'W':19,'Y':20}
                
    #print(seq_input[0])
    L = len(seq_input[0])
    n = len(seq_input)
    
    X = np.zeros((n, len(my_dict)*L))
    
    # loop through each sequence and one_hot encode
    for i, seq in enumerate(seq_input):
        for j, aa in enumerate(seq):
            # fine one index that should be '1'
            aa_indx = my_dict[aa]
            X[i][21*j+aa_indx] = 1
            
    return X
    
def one_hot_contacts(seq_input, ss, contacts):
    # make contact directory 
    my_contact_dict = {}
    AAs = ['-','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S',\
        'T','V','W','Y']
    count = 0
    for k in AAs:
        for j in AAs:
            my_contact_dict[(k, j)] = count
            count += 1
    
    n = len(seq_input)
    
    X = np.zeros((n, len(my_contact_dict)*len(contacts)))
    
    # loop through each sequence and one_hot encode contacts
    for i, seq in enumerate(seq_input):
        for p, (j, k) in enumerate(contacts):
            # find the contact
            contact_index = my_contact_dict[(seq[j],seq[k])]
            X[i][len(my_contact_dict)*p+contact_index] = 1
            
    return X

def one_hot_(seq_input, ss, contacts):
    # reshape to contain both contact and sequence info
    X_seq = one_hot_seq(seq_input)
    X_contact = one_hot_contacts(seq_input, ss, contacts)
    
    X = []
    for i, x in enumerate(X_seq):
        x_c = X_contact[i]
        X.append(np.concatenate((x, x_c)))
    return X
    
