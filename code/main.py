import torch
from helper import load_data
from solution import PCA, AE, frobeniu_norm_error
import numpy as np
import os


def test_pca(A, p):
    pca = PCA(A, p)
    Ap, G = pca.get_reduced()
    A_re = pca.reconstruction(Ap)
    error = frobeniu_norm_error(A, A_re)
    print('PCA-Reconstruction error for {k} components is'.format(k=p), error)
    return G

def test_ae(A, p):
    model = AE(d_hidden_rep=p)
    model.train(A, A, 128, 300)
    A_re = model.reconstruction(A)
    final_w = model.get_params()
    error = frobeniu_norm_error(A, A_re)
    print('AE-Reconstruction error for {k}-dimensional hidden representation is'.format(k=p), error)
    return final_w

if __name__ == '__main__':
    dataloc = "./data/USPS.mat"
    A = load_data(dataloc)
    A = A.T
    ## Normalize A
    A = A/A.max()

    # Note: You are free to modify your code here for debugging and justifying your ideas for 5(f)
    
    '''
    PART-A-B calculating PCA reconstruction error for p = 32,64,128
    '''
    
    p_array = [32, 64, 128]
    for p in p_array:
        G = test_pca(A, p)
    
    '''    
    #PART-C-D-G calculating AE reconstruction error for d = 32,64,128
    '''
    d_array = [32, 64, 128]
    for d in d_array:
        final_w = test_ae(A, d)
        
    '''
    #PART-E calculating AE reconstruction error for d = 32,64,128
    '''
    pd_array = [32, 64, 128]
    for pd in pd_array:
        G = test_pca(A, pd)
        final_w = test_ae(A, pd)
        
    '''    
    #PART-F relation between G and W
    '''    
    pd_array = [32,64,128]
    
    for pd in pd_array:
        G = test_pca(A,pd)
        W = test_ae(A,pd)  
        R = G.T@W
        u,s,v = np.linalg.svd(R,full_matrices=True)
        R_bar = u@np.eye(u.shape[0],v.shape[0])@v
        G_bar = G@R_bar
        print("G'-W", frobeniu_norm_error(G_bar,W))
        print("R-R'", frobeniu_norm_error(R,R_bar))
        print("G' reconstruction error:", frobeniu_norm_error(A,G_bar@G_bar.T@A))
    
    
    '''
    #PART-H calculating non-linear AE reconstruction error for d = 64
    '''
    d = 64
    final_w = test_ae(A, d)
    
 
