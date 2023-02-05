#%% The python script is designed for ridge regression and adaptive ridge regression
# Tuning method: AIC, BIC, LOO-CV


import pandas as pd
import numpy as np
# Sparse Signal Dataset
def gen_beta(p,n):
    beta = np.zeros(p)
    for i in range(1,p):
        if i <= np.sqrt(p):
            beta[i-1] = (2/np.sqrt(n))
    return beta

def gen_xcov(p,rou):
    cov = np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            if i == j:
                cov[i,j] = 1
            else:
                cov[i,j] = rou**abs(i-j)
    return cov

def gen_info(p,rou,n=100):
    miu = np.zeros(p)
    beta = gen_beta(p,n).reshape(-1,1)
    cov = gen_xcov(p,rou)
    var = ((1-0.8)/0.8)*np.dot(np.dot(beta.T,cov),beta)[0][0]
    return miu,beta,cov,var

p = [10,25,50]
rou = [0,0.25,0.5]

np.random.seed(100)

def simulation_x_y(p,rou):
    X_dict = dict()
    y_dict = dict()
    for i in range(len(p)):
        for k in range(len(p)):
            X_data = dict()
            y_data = dict()
            p0 = p[i]
            rou0 = rou[k]
            for j in range(1001):
                miu,beta,cov,var = gen_info(p0,rou0)
                X = np.random.multivariate_normal(miu,cov,size=(120))
                eps = np.random.normal(0,var,size=(120)).reshape(-1,1)
                y = np.dot(X,beta)+eps
                X_data[j] = X
                y_data[j] = y
            X_dict[(p0,rou0)] = X_data
            y_dict[(p0,rou0)] = y_data
    return X_dict,y_dict
X_dict,y_dict = simulation_x_y(p,rou)
alpha = np.logspace(-5,0,10)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def ridge_Loocv(p,rou,X = X_dict,y = y_dict):
    mse_dict = dict()
    for i in range(3):
        for k in range(3):
            mse = []
            for j in range(1000):
                X = X_dict[(p[i],rou[k])][j]
                y = y_dict[(p[i],rou[k])][j]
                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/6,random_state=100)
                clf = RidgeCV(alpha).fit(X_train,y_train)
                y_pred = clf.predict(X_test)
                mse.append(mean_squared_error(y_test,y_pred))
            mse_dict[(p[i],rou[k])] = np.mean(mse)
    return mse_dict
ridge_loocv_mse = ridge_Loocv(p,rou,X_dict,y_dict)

from numpy.linalg import inv
from numpy.random import permutation
def adaptive_ridge(X,y,alpha):
    beta_ols = np.dot(np.dot(inv(np.dot(X.T,X)),X.T),y)
    W = np.diag(1/beta_ols.T[0])
    beta = np.linalg.solve(X.T @ X + alpha * W @ W, X.T @ y)
    return beta

def adaptive_ridge_loocv(X,y,alpha):
    n = y.shape[0]
    test = permutation(n)
    mse_dict = dict()
    for a in alpha:
        test_mse = []
        for test_num in test:
            train_num = [j for j in range(n) if j != test_num]
            X_train,y_train,X_test,y_test = X[train_num,:],y[train_num,:],X[test_num,:],y[test_num,:]
            beta = adaptive_ridge(X_train,y_train,a)
            y_pred = np.dot(X_test.T,beta)
            mse = mean_squared_error(y_test,y_pred)
            test_mse.append(mse)
        mse_dict[a] = np.mean(test_mse)
    return sorted(mse_dict.values())[0]
def adaptive_ridge_loocv_wholesets(p,rou,X_dict,y_dict,alpha):
    mse_dict = dict()
    for i in range(3):
        for k in range(3):
            mse = []
            for j in range(1000):
                X = X_dict[(p[i],rou[k])][j]
                y = y_dict[(p[i],rou[k])][j]
                mse_loocv = adaptive_ridge_loocv(X,y,alpha)
                mse.append(mse_loocv)
            mse_dict[(p[i],rou[k])] = np.mean(mse)
            print(str(p[i])+', '+str(rou[k])+': '+str(np.mean(mse)))
    return mse_dict
alpha = np.logspace(-5,0,10)
adaptive_ridge_loocv_mse = adaptive_ridge_loocv_wholesets(p,rou,X_dict,y_dict,alpha)

from sklearn.linear_model import Ridge

def ABIC(X,y,model):
    error = model.predict(X) - y
    rss = np.dot(error.T,error)
    k = np.count_nonzero(model.coef_)
    n = y.shape[0]
    MLE = n*np.log(rss)
    AIC = 2*k+MLE
    BIC = np.log(n)*k+MLE
    return AIC,BIC

def ridge_aic_bic(X,y,alpha):
    modelA = modelB = None
    minAIC = minBIC = np.inf
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=5/6,random_state=100)
    for a in alpha:
        model = Ridge(a).fit(X_train,y_train)
        AIC,BIC = ABIC(X,y,model)
        if AIC < minAIC:
            minAIC = AIC
            modelA = model
        if BIC < minBIC:
            minBIC = BIC
            modelB = model
    mseA = mean_squared_error(y_test,modelA.predict(X_test))
    mseB = mean_squared_error(y_test,modelB.predict(X_test))
    return mseA,mseB

def ridge_aic_bic_wholesets(p,rou,X_dict,y_dict,alpha):
    mse_aic_dict, mse_bic_dict= dict(),dict()
    for i in range(3):
        for k in range(3):
            mse_aic_lst,mse_bic_lst = [],[]
            for j in range(1000):
                X = X_dict[(p[i],rou[k])][j]
                y = y_dict[(p[i],rou[k])][j]
                mse_aic,mse_bic = ridge_aic_bic(X,y,alpha)
                mse_aic_lst.append(mse_aic)
                mse_bic_lst.append(mse_bic)
            mse_aic_dict[(p[i],rou[k])] = np.mean(mse_aic_lst)
            mse_bic_dict[(p[i],rou[k])] = np.mean(mse_bic_lst)
            print(str(p[i])+', '+str(rou[i])+'||aic: '+str(mse_aic_dict[(p[i],rou[k])]))
            print(str(p[i])+', '+str(rou[i])+'||bic: '+str(mse_bic_dict[(p[i],rou[k])]))
    return mse_aic_dict,mse_bic_dict

alpha = np.logspace(-5,0,10)
mse_aic_dict,mse_bic_dict = ridge_aic_bic_wholesets(p,rou,X_dict,y_dict,alpha)

class AdaptiveRidge:
    def __init__(self,alpha):
        self.alpha = alpha

    def fit(self,X,y):
        beta_ols = np.dot(np.dot(inv(np.dot(X.T,X)),X.T),y)
        W = np.diag(1/beta_ols.T[0])
        self.coef_ = np.linalg.solve(X.T @ X + self.alpha * W @ W, X.T @ y)
        return self

    def predict(self,X):
        return np.dot(X,self.coef_)

def ABIC(X,y,model):
    error = model.predict(X) - y
    rss = np.dot(error.T,error)
    k = np.count_nonzero(model.coef_)
    n = y.shape[0]
    MLE = n*np.log(rss)
    AIC = 2*k+MLE
    BIC = np.log(n)*k+MLE
    return AIC,BIC

def adaptive_ridge_aic_bic(X,y,alpha):
    modelA = modelB = None
    minAIC = minBIC = np.inf
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=5/6,random_state=100)
    # beta_ols = np.dot(np.dot(inv(np.dot(X_train.T,X_train)),X_train.T),y_train)
    # W = np.diag(beta_ols.T[0])
    # X_train = np.dot(X_train,W)
    for a in alpha:
        model = ApativeRidge(a).fit(X_train,y_train)
        AIC,BIC = ABIC(X,y,model)
        if AIC < minAIC:
            minAIC = AIC
            modelA = model
        if BIC < minBIC:
            minBIC = BIC
            modelB = model
    mseA = mean_squared_error(y_test,modelA.predict(X_test))
    mseB = mean_squared_error(y_test,modelB.predict(X_test))
    return mseA,mseB

def adaptive_ridge_aic_bic_wholesets(p,rou,X_dict,y_dict,alpha):
    mse_aic_dict, mse_bic_dict= dict(),dict()
    for i in range(3):
        for k in range(3):
            mse_aic_lst,mse_bic_lst = [],[]
            for j in range(1000):
                X = X_dict[(p[i],rou[k])][j]
                y = y_dict[(p[i],rou[k])][j]
                mse_aic,mse_bic = ridge_aic_bic(X,y,alpha)
                mse_aic_lst.append(mse_aic)
                mse_bic_lst.append(mse_bic)
            mse_aic_dict[(p[i],rou[k])] = np.mean(mse_aic_lst)
            mse_bic_dict[(p[i],rou[k])] = np.mean(mse_bic_lst)
            print(str(p[i])+', '+str(rou[k])+'||aic: '+str(mse_aic_dict[(p[i],rou[k])]))
            print(str(p[i])+', '+str(rou[k])+'||bic: '+str(mse_bic_dict[(p[i],rou[k])]))
    return mse_aic_dict,mse_bic_dict
alpha = np.logspace(-5,0,10)
mse_aic_dict_adp,mse_bic_dict_adp = adaptive_ridge_aic_bic_wholesets(p,rou,X_dict,y_dict,alpha)

