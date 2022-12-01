import numpy as np
from datetime import datetime as dt
from functools import reduce
import matplotlib.pyplot as plt

### set u and sigma
mu_o = np.array([0, 1, 2,4,8,10,20,50,80,100])
sigma_o=np.array([[10,4,8,4,10,4,4,0,6,8],
                [ 4,6,2,4,6,4,4,2,2,4],
                [ 8,2,10,4,10,6,2,0,4,8],
                [ 4,4,4,8,6,8,4,4,2,6],
                [10,6,10,6,18,10,8,4,8,12],
                [ 4,4,6,8,10,12,6,6,4,8],
                [ 4,4,2,4, 8, 6,8,4,6,4],
                [ 0,2,0,4, 4, 6,4,6,2,2],
                [ 6,2,4,2, 8, 4,6,2,8,4],
                [ 8,4,8,6, 12,8,4,2,4,14]])

### generate true data 
n = 3000
X_truth = np.random.multivariate_normal(mu_o, sigma_o, n)

# compute the mean and standard deviation of the relative errors of miu and sigma
# simulating missing value

def simulate_nan(X,nan_rate):
    X_complete = X.copy()
    nr,nc = X_complete.shape
    C = np.random.random(nr*nc).reshape(nr,nc) > nan_rate # correct
   
    checker = np.where(sum(C.T)==0)[0]
    if len(checker)==0:
        X_complete[C==False]=np.nan
        
    else:
        for i in checker:
            reviving_components = np.random.choice(nc,
                                                   int(np.ceil(nc*np.random.random())),
                                                   replace=False)
            
            C[i,np.ix_(reviving_components)]=True        
        X_complete[C==False]=np.nan
        
    result = {'X':X_complete,
              'C':C,
              'nan_rate':nan_rate,
              'nan_rate_actual':np.sum(C==False)/(nr*nc)
              }
    
    return result

def impute_em(X,max_iter=200,eps=0.00005,mu0 = mu_o,sigma0=sigma_o):
    nr,nc=X.shape
    C = np.isnan(X) == False
    
    # remember Missing and observed value
    one_to_nc = np.arange(1,nc+1,step=1)
    M = one_to_nc*(C==False)-1
    O = one_to_nc*C-1 
    
    # generate Mu_o and sigma_o
    observed_rows = np.where(np.isnan(sum(X.T))==False)[0]
    S = np.cov(X[observed_rows,].T)
    Mu = np.mean(X[observed_rows,],axis=0).reshape(10,1)
    S_tilde = {}
    X_tilde = X.copy()
    no_conv = True
    iteration = 0
    while no_conv and iteration < max_iter:
        for i in range(nr):
            S_tilde[i] = np.zeros(nc ** 2).reshape(nc, nc)
            if set(O[i, ]) != set(one_to_nc - 1): # missing component exists
                M_i, O_i = M[i, ][M[i, ] != -1], O[i, ][O[i, ] != -1]
                S_MM = S[np.ix_(M_i, M_i)]
                S_MO = S[np.ix_(M_i, O_i)]
                S_OM = S_MO.T
                S_OO = S[np.ix_(O_i, O_i)]
                
                Mu_O,Mu_M = Mu[np.ix_(O_i)],Mu[np.ix_(M_i)]
                Mu_O = Mu_O.reshape(Mu_O.shape[0],1)
                Mu_M = Mu_M.reshape(Mu_M.shape[0],1)
                x_O = X_tilde[i,O_i].reshape(Mu_O.shape[0],1)
                Mu_cm = Mu_M + np.dot(np.dot(S_MO,np.linalg.inv(S_OO)),(x_O-Mu_O))
                
                X_tilde[i,np.ix_(M_i)] =  Mu_cm.T
                S_MM_O = S_MM - np.dot(np.dot(S_MO,np.linalg.inv(S_OO)),S_OM)
                S_tilde[i][np.ix_(M_i, M_i)] = S_MM_O
                
        Mu_new = np.mean(X_tilde, axis = 0)
        S_new = np.cov(X_tilde.T, bias = 1)+reduce(np.add, S_tilde.values()) / nr
        no_conv = np.linalg.norm(Mu - Mu_new) >= eps or np.sqrt(np.sum((S-S_new)**2))>= eps
        
        Mu = Mu_new
        S = S_new
        iteration += 1
        
    error_mu = np.linalg.norm(Mu-mu0,ord=2)
    error_sigma = np.sqrt(np.sum((S-sigma0)**2))
    
    result = {'mu':Mu,
              'Sigma':S,
              'X_imputed': X_tilde,
              'C':C,
              'iteraction':iteration,
              'mu_error':error_mu,
              'sigma_error':error_sigma }

    return result

# nan_rate = 0.4
mu04,sigma04=[],[]
i=0
while i<10:
    mr04 = simulate_nan(X_truth, nan_rate = .4)
    result4 = impute_em(mr04['X'])
    mu04.append(result4['mu_error'])
    sigma04.append(result4['sigma_error'])
    i +=1

mean_mu04 = np.mean(mu04)
sd_mu04 = np.std(mu04)

mean_sigma04 = np.mean(sigma04)
sd_sigma04 = np.std(sigma04)
end = dt.now()



# nan_rate = 0.3
mu03,sigma03=[],[]
i=0
while i<10:
    mr03 = simulate_nan(X_truth, nan_rate = .3)
    result3 = impute_em(mr03['X'])
    mu03.append(result3['mu_error'])
    sigma03.append(result3['sigma_error'])
    i +=1
    


mean_mu03 = np.mean(mu03)
sd_mu03 = np.std(mu03)

mean_sigma03 = np.mean(sigma03)
sd_sigma03 = np.std(sigma03)



# nan_rate = 0.2
mu02,sigma02=[],[]
i=0
while i<10:
    mr02 = simulate_nan(X_truth, nan_rate = .2)
    result2 = impute_em(mr02['X'])
    mu02.append(result2['mu_error'])
    sigma02.append(result2['sigma_error'])
    i +=1

mean_mu02 = np.mean(mu02)
sd_mu02 = np.std(mu02)

mean_sigma02 = np.mean(sigma02)
sd_sigma02 = np.std(sigma02)


# nan_rate = 0.1
mu01,sigma01=[],[]
i=0
while i<10:
    mr01 = simulate_nan(X_truth, nan_rate = .1)
    result1 = impute_em(mr01['X'])
    mu01.append(result1['mu_error'])
    sigma01.append(result1['sigma_error'])
    i +=1

mean_mu01= np.mean(mu01)
sd_mu01 = np.std(mu01)

mean_sigma01 = np.mean(sigma01)
sd_sigma01 = np.std(sigma01)


# nan_rate = 0.05
mu005,sigma005=[],[]
i=0
while i<10:
    mr005 = simulate_nan(X_truth, nan_rate = .05)
    result05 = impute_em(mr005['X'])
    mu005.append(result05['mu_error'])
    sigma005.append(result05['sigma_error'])
    i +=1
    
mean_mu005= np.mean(mu005)
sd_mu005 = np.std(mu005)

mean_sigma005 = np.mean(sigma005)
sd_sigma005 = np.std(sigma005)


# plot figure
X = np.array([0.05,0.1,0.2,0.3,0.4])
mean_mu = np.array([mean_mu005,mean_mu01,mean_mu02,mean_mu03,mean_mu04])
sd_mu = np.array([sd_mu005,sd_mu01,sd_mu02,sd_mu03,sd_mu04])

mean_sigma = np.array([mean_sigma005,mean_sigma01,mean_sigma02,mean_sigma03,mean_sigma04])
sd_sigma = np.array([sd_sigma005,sd_sigma01,sd_sigma02,sd_sigma03,sd_sigma04])

org_mu = np.linalg.norm(np.mean(X_truth,axis=0)-mu_o,ord=2)
org_sigma = np.sqrt(np.sum((np.cov(X_truth.T)-sigma_o)**2))


plt.errorbar(X, mean_mu,yerr=sd_mu,label='X_imputing')
plt.plot(X,[org_mu]*5,label='X_truth')
plt.legend(loc='upper left')
plt.title('relative errors for miu')
plt.show()

plt.errorbar(X, mean_sigma,yerr=sd_sigma,label='X_imputing')
plt.plot(X,[org_sigma]*5,label='X_truth')
plt.legend(loc='upper left')
plt.title('relative errors for sigma')
plt.show()



