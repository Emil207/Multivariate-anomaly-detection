# Preprocessing
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy import stats

def make_samples(sequence, timesteps):
    ''' Make samples of length 'timesteps' of 'sequence'. '''
    X = list()
    for i in range(len(sequence)-timesteps+1):
        # find the end of this pattern
        end_ix = i + timesteps
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix].to_numpy()
        X.append(seq_x)
    return np.array(X)

def normalize(train, test):
    ''' Normalize train and test set with MinMaxScaler.'''
    norm_scaler = MinMaxScaler()
    train = norm_scaler.fit_transform(train)
    test = norm_scaler.transform(test)
    return train, test

def standardize(train, test):
    ''' Standardize train and test set with StandardScaler.'''
    std_scaler = StandardScaler()
    train = std_scaler.fit_transform(train)
    test = std_scaler.transform(test)
    return train, test

def dim_reduce(train, test, dimensions):
    ''' Dimensionality reduction with PCA. '''
    pca = PCA(n_components=dimensions)
    train = pca.fit_transform(train)
    test = pca.transform(test)
    return train, test, pca.explained_variance_ratio_