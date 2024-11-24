import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def function_pca(data, channels,components):


    # Separate each channel into individual arrays

    # Step 1: Standardize the data (important for PCA)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.T)  # Standardizing each column (channel)

    # Step 2: Apply PCA
    pca = PCA(n_components=components)  # You can choose the number of components based on your needs
    pca_result = pca.fit_transform(data_scaled)
    return (pca_result, pca)


def pca(data):
    data_transp = np.transpose(data)
    data_square = np.dot(data,data_transp)

    eigen_vals, eigen_vects = LA.eig(data_square)



    #eigen values in descending order
    sorted_indexs = np.argsort(-eigen_vals)
    eigen_vals_sorted = eigen_vals[sorted_indexs]
    eigen_vects_sorted = eigen_vects[:, sorted_indexs]

    print(eigen_vals_sorted)
    p = eigen_vects_sorted

    yd = np.dot(data.T,p).T

    return (yd, eigen_vals_sorted)