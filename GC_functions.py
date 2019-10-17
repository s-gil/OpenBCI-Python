# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression

def GC_connectivity(data,order):
    """
    The function GC_connectivity computes the Granger Causality (GC) of a given 
    order among the time series contained in the rows of the matrix data. 
    
    Inputs:
        data: Matrix whose rows contain the time series for which GC will be 
        computed (channels x N)
        order: Order of the autoregressive models used to compute GC 
    Output: 
        GC_matrix: Granger causality matrix (channels x channels)
    
    """
    num_ch = np.shape(data)[0] #Number of channels or times series (rows in the matrix data)
    GC_matrix = np.zeros((num_ch,num_ch)) 
    
    for i in range(0,num_ch):
        for j in range(0,num_ch): 
            if i != j:
                x = data[i]
                y = data[j]
                GC_matrix[i,j] = Granger_Causality(x,y,order)
    return GC_matrix 

                        
def Granger_Causality(x,y,order):
    """
    The function Granger_Causality computes the Granger causality of a given order
    between the time series x and y, assuming that x drives y. 
    
    Inputs:
        x: Driving time series (1 x N)
        y: Driven time series (1 x N)
        order: Order of the autoregressive models used to compute GC 
    Output: 
        GC: Granger causality between x and y  
    """
    X = np.zeros((len(x)-order,order))
    Y = np.zeros((len(y)-order,order))
    
    for p in range(0,order):
        X[:,p] = x[p:-(order-p)]
        Y[:,p] = y[p:-(order-p)]
    
    A0 = np.ones((np.shape(Y)[0],1))
    Y_aux = y[order::]
    Y_aux = np.resize(Y_aux,(len(Y_aux),1))
    
    #Autoregressive model 
    regressor_y = LinearRegression()
    regressor_y.fit(np.concatenate((A0,Y),axis=1), Y_aux)
    coeff_y = regressor_y.coef_ #Regression coefficients 
    u_y = Y_aux-np.dot(coeff_y,np.concatenate((A0,Y),axis=1).T).T #Residuals 
    
    #Bivariate autoregressive model 
    regressor_yx = LinearRegression()
    regressor_yx.fit(np.concatenate((A0,Y,X),axis=1), Y_aux)
    coeff_yx = regressor_yx.coef_ #Regression coefficients 
    u_yx = Y_aux-np.dot(coeff_yx,np.concatenate((A0,Y,X),axis=1).T).T #Residuals
    
    GC = np.log(np.var(u_y)/np.var(u_yx)) #Granger Causality 

    if GC < 0:
        GC = 0.0
        
    return GC