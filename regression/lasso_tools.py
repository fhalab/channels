from __future__ import division
import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


# ML imports
from sklearn import linear_model
from scipy import optimize
import scipy
from sklearn.model_selection import KFold

# custom imports
import encoding_tools as encoding
import GP_tools as GP


def data_format_all(property_, df):
    # remove ChR_29_10 & ChR_30_10 for kinetics and spectra because currents too low for accurate measurements
    if property_ == 'green_norm' or property_ == 'kinetics_off':
        df = df[df.chimera != 'ChR_29_10']
        df = df[df.chimera != 'ChR_30_10']
    
    # make a seperate dataframe for the selected property
    df_select = pd.DataFrame()
    df_select['prop'] = df[str(property_)]
    df_select['seq'] = df['seq']
    df_select['block_k'] = df['block_k']
    df_select['chimera'] = df['chimera']
    df_select.dropna(inplace=True)

    # normalize training data
    log_data = np.log(df_select.prop.values)
    y = (log_data - np.mean(log_data))/np.std(log_data)
    seq = df_select.seq.values
    
    return log_data, y, seq, df_select

def cross_validation(X, log_data, property_):
    path_outputs = 'outputs/'
    
    kf = KFold(n_splits=20) # Define the split
    kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    
    mu_s = []
    var_s = []
    y_s = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        
        log_data_train, log_data_test = log_data[train_index], log_data[test_index]
        
        y_train = (log_data_train - np.mean(log_data_train))/np.std(log_data_train)
        y_test = (log_data_test - np.mean(log_data_train))/np.std(log_data_train)
        
        initial_guess = [0.1,10]
        
        # take the log of the initial guess for optimiziation
        initial_guess_log = np.log(initial_guess)
        
        # optimize to fit model
        result = scipy.optimize.minimize(GP.neg_log_marg_likelihood, initial_guess_log, args=(X_train,y_train), method='L-BFGS-B')#,
        
        # next set of hyper prams
        prams_me = [np.exp(result.x[0])**2, np.exp(result.x[1])]
        
        # next used trained GP model to predict on test data
        mu, var = GP.predict_GP(X_train, y_train, X_test, prams_me)
        
        # append
        mu_s.append(mu)
        var_s.append(var)
        y_s.append(y_test)
    
    # reformat all
    y_s_all = [j for i in y_s for j in i]
    mu_s_all = [j for i in mu_s for j in i]

    # plot results
    plt.figure('GP test set', figsize=(1.5, 1.5))
    plt.plot(y_s_all, mu_s_all, 'o', color='k', ms=3)
    
    # calc correlation
    measured = y_s_all
    predicted = mu_s_all
    
    par = np.polyfit(measured, predicted, 1, full=True)
    slope=par[0][0]
    intercept=par[0][1]
    
    # calc correlation
    variance = np.var(predicted)
    residuals = np.var([(slope*xx + intercept - yy)  for xx,yy in zip(measured, predicted)])
    Rsqr = np.round(1-residuals/variance, decimals=2)
    
    print('20-fold corss validation of GP regression model')
    print('R = %0.2f'% np.sqrt(Rsqr))
    
    max_x = np.max(y_s_all)
    min_x = np.min(y_s_all)
    
    plt.plot([min_x, max_x], [slope*min_x+intercept, slope*max_x+intercept], '-', color='k')
    plt.savefig(path_outputs + str(property_)+'_matern_kernel_LASSO_CV.pdf', bbox_inches='tight', transparent=True)
    plt.show()
    return measured, predicted

def lasso_(alpha_, X, y):
    """ import alpha and full X matrix and y to give limited feature set"""
    clf = linear_model.Lasso(alpha=alpha_)
    
    # fit model with training data
    clf.fit(X,y)
    
    # get the coeff for the input to the next model
    lasso_coeff = clf.coef_
    return lasso_coeff

def lasso_reformat_X(lasso_coeff, X):
    X_lasso = []
    for x in X:
        X_lasso.append(x[lasso_coeff != 0])
    return np.array(X_lasso)


def id_sequence_features(index_seq, seqs):
    # make sequence directory
    my_dict = {'-':0, 'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,\
        'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,\
            'V':18,'W':19,'Y':20}

    # make a vector with amino acids filling
    L = len(seqs[0])
    
    seq_key = np.chararray(len(my_dict)*L)
    
    
    for j in range(L):
        for k, v in my_dict.items():
            seq_key[j*len(my_dict)+v] = k

    amino_acid_numb = np.floor(index_seq / len(my_dict))
    amino_acid = seq_key[index_seq]
    # print(str(amino_acid)+str(int(amino_acid_numb)))
    aa_numb=amino_acid_numb
    aa=amino_acid
    return aa_numb, aa


def id_contact_features(index_contacts, contacts):
    # make contact directory
    my_contact_dict = {}
    AAs = ['-','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    count = 0
    for k in AAs:
        for j in AAs:
            my_contact_dict[(k, j)] = count
            count += 1

    contact_key = np.chararray((len(my_contact_dict), 2))
    
    for k, v in my_contact_dict.items():
        contact_key[v][0] = k[0]
        contact_key[v][1] = k[1]
    
    contact_numbers = []
    contact_amino_acids = []
    
    contact_numb = int(np.floor(index_contacts / len(my_contact_dict)))
    contact_numbers = contacts[contact_numb]
    amino_acid_contact = contact_key[index_contacts % len(my_contact_dict)]
    contact_amino_acids = [amino_acid_contact[0], amino_acid_contact[1]]
    return [amino_acid_contact[0], amino_acid_contact[1]], contacts[contact_numb]

def unique_columns2(data):
    """
    Identify co-varying columns
    """
    dt = np.dtype((np.void, data.dtype.itemsize * data.shape[0]))
    dataf = np.asfortranarray(data).view(dt)
    u,uind = np.unique(dataf, return_inverse=True)
    u = u.view(data.dtype).reshape(-1,data.shape[0]).T
    return (u,uind)

def find_features(df, ss, contacts, coeffs, X, weights):
    # re make the sequence and contact X
    X_seq = encoding.one_hot_seq(df['seq'].values)
    X_contact = encoding.one_hot_contacts(df['seq'].values, ss, contacts)
    X_seq = np.array(X_seq)
    X_contact = np.array(X_contact)
    
    # find the non-zeros features
    index_not_zero = np.where(coeffs != 0)[0]
    
    # find the co-varying
    u,uind = unique_columns2(X)
    vects_covary = uind[index_not_zero]
    lim_set = list(set(vects_covary))
    
    # make dataframe of lasso features with weights
    df_lasso_features = pd.DataFrame()
    df_lasso_features['index_not_zero'] = index_not_zero
    df_lasso_features['vects_covary'] = vects_covary
    df_lasso_features['weights'] = weights
    
    # of the lasso limited set, remove replicated values and ID the covarying set of features for each lasso-limited set
    co_vary_lim_set = []
    co_vary_lim_set_weights = []
    for i in lim_set:
        j, = np.where(uind == i)
        co_vary_lim_set.append(j)
        w = df_lasso_features[df_lasso_features.vects_covary == i].weights.values[0]
        co_vary_lim_set_weights.append(w)
    
    # build list of all features, seperate sequence vs contact features
    all_features = []
    feature_type = []
    weights_ = []
    aa_ = []
    for ind, i in enumerate(co_vary_lim_set):
        for j in i:
            if j < np.shape(X_seq)[1]:
                aa_number, aa = id_sequence_features(j, df['seq'].values)
                all_features.append(aa_number)
                aa_.append(aa)
                weights_.append(co_vary_lim_set_weights[ind])
                feature_type.append('seq')
            
            elif j > np.shape(X_seq)[1]:
                j_contact = j - np.shape(X_seq)[1]
                contact_aas, contact_pos  = id_contact_features(j_contact, contacts)
                all_features.append(contact_pos)
                aa_.append(contact_aas)
                weights_.append(co_vary_lim_set_weights[ind])
                feature_type.append('contact')

    df_features = pd.DataFrame(dtype=object)
    df_features['weights'] = weights_
    df_features['feature'] = all_features
    df_features['type'] = feature_type
    df_features['aa'] = aa_
    
    # define different co-varying groups
    groups_ = list(set(df_features.weights))
    group_number = range(len(groups_))
    
    feature_by_group = []
    for i in df_features.weights:
        feature_by_group.append(groups_.index(i))
    df_features['feature_group'] = feature_by_group
    return df_features

def refromat_feature_numbering(df_features, df_select, property_, lasso_alpha):
    C1C2_seq = df_select[df_select.chimera == 'C1C2'].seq.values[0]
    CheRiff_seq = df_select[df_select.chimera == 'CheRiff'].seq.values[0]
    CsChrim_seq = df_select[df_select.chimera == 'CsChrim'].seq.values[0]
    
    C1C2_seq_numb = range(len(C1C2_seq))
    CheRiff_seq_numb = range(len(CheRiff_seq))
    CsChrim_seq_numb = range(len(CsChrim_seq))
    
    # C1C2 numbering: first drop the gaps in the alignment sequence, but keep proper index
    seq_numb_mod = []
    C1C2_seq_mod = []
    for ind,i in enumerate(C1C2_seq):
        if i != '-':
            seq_numb_mod.append(ind)
            C1C2_seq_mod.append(i)

    gaps = ['-']*49
    C1C2_seq_mod = ''.join(gaps+C1C2_seq_mod)
    C1C2_seq_numb_mod = [-1]*49 + seq_numb_mod
    
    # CheRiff numbering: first drop the gaps in the alignment sequence, but keep proper index
    seq_numb_mod = []
    CheRiff_seq_mod = []
    for ind,i in enumerate(CheRiff_seq):
        if ind == 22:
            seq_numb_mod.append(ind)
            CheRiff_seq_mod.append(i)
        elif ind == 23:
            seq_numb_mod.append(ind)
            CheRiff_seq_mod.append(i)
        elif ind == 24:
            seq_numb_mod.append(ind)
            CheRiff_seq_mod.append(i)
        elif i != '-':
            seq_numb_mod.append(ind)
            CheRiff_seq_mod.append(i)

    gaps = ['-']*74
    CheRiff_seq_mod = ''.join(gaps+CheRiff_seq_mod)
    CheRiff_seq_numb_mod = [-1]*74 + seq_numb_mod
    
    # CsChrim numbering: first drop the gaps in the alignment sequence, but keep proper index
    seq_numb_mod = []
    CsChrim_seq_mod = []
    for ind,i in enumerate(CsChrim_seq):
        if ind == 22:
            seq_numb_mod.append(ind)
            CsChrim_seq_mod.append(i)
        elif i != '-':
            seq_numb_mod.append(ind)
            CsChrim_seq_mod.append(i)

    gaps = ['-']*47
    CsChrim_seq_mod = ''.join(gaps+CsChrim_seq_mod)
    CsChrim_seq_numb_mod = [-1]*47 + seq_numb_mod
    
    path_outputs = 'outputs/'
    
    # go through sequence/contact features and adjust numbering for plotting on 3ug9.pdb
    feature_adjust = []
    aa_feature_adjust = []
    for ind, i in enumerate(df_features.feature):
        if df_features.type[ind] == 'seq':
            feature_adjust.append(C1C2_seq_numb_mod.index(i))
            aa_feature_adjust.append(C1C2_seq_mod[C1C2_seq_numb_mod.index(i)])
        else:
            feature_adjust.append([C1C2_seq_numb_mod.index(i[0]), C1C2_seq_numb_mod.index(i[1])])
            aa_feature_adjust.append([C1C2_seq_mod[C1C2_seq_numb_mod.index(i[0])],
                                      C1C2_seq_mod[C1C2_seq_numb_mod.index(i[1])]])
    df_features['C1C2_features_adjust'] = feature_adjust
    df_features['C1C2_aa_adjust'] = aa_feature_adjust
    
    # go through sequence/contact features and adjust numbering for CheRiff parent
    feature_adjust = []
    aa_feature_adjust = []
    for ind, i in enumerate(df_features.feature):
        if df_features.type[ind] == 'seq':
            feature_adjust.append(CheRiff_seq_numb_mod.index(i))
            aa_feature_adjust.append(CheRiff_seq_mod[CheRiff_seq_numb_mod.index(i)])
        else:
            feature_adjust.append([CheRiff_seq_numb_mod.index(i[0]), CheRiff_seq_numb_mod.index(i[1])])
            aa_feature_adjust.append([CheRiff_seq_mod[CheRiff_seq_numb_mod.index(i[0])],
                                      CheRiff_seq_mod[CheRiff_seq_numb_mod.index(i[1])]])

    df_features['CheRiff_features_adjust'] = feature_adjust
    df_features['CheRiff_aa_adjust'] = aa_feature_adjust
    
    # go through sequence/contact features and adjust numbering for CsChrim parent
    feature_adjust = []
    aa_feature_adjust = []
    for ind, i in enumerate(df_features.feature):
        if df_features.type[ind] == 'seq':
            feature_adjust.append(CsChrim_seq_numb_mod.index(i))
            aa_feature_adjust.append(CsChrim_seq_mod[CsChrim_seq_numb_mod.index(i)])
        else:
            feature_adjust.append([CsChrim_seq_numb_mod.index(i[0]), CsChrim_seq_numb_mod.index(i[1])])
            aa_feature_adjust.append([CsChrim_seq_mod[CsChrim_seq_numb_mod.index(i[0])],
                                      CsChrim_seq_mod[CsChrim_seq_numb_mod.index(i[1])]])
    df_features['CsChrim_features_adjust'] =     feature_adjust
    df_features['CsChrim_aa_adjust'] = aa_feature_adjust
    
    df_features.to_csv(path_outputs+'matern_'+ str(property_) +'_' + str(lasso_alpha) + '_LASSO.csv')
    return df_features