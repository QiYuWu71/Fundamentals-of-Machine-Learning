# Logistic Regression

import numpy as np
from scipy import io
from sklearn import preprocessing

# Data assignment
def read_spam_data():
    raw_data = io.loadmat('spamData.mat')
    train_data = {'X':raw_data['Xtrain'],
                  'y':raw_data['ytrain']}
    test_data = {'X':raw_data['Xtest'],
                  'y':raw_data['ytest']}
    return train_data,test_data

train_data, test_data = read_spam_data()

# Data transformation
def transform_log(x):
    return np.log(x+0.1)

def transform_binary(x):
    return (x > 0).astype(np.int32)

# Cross validation method
train_fold= np.array([val for val in [1,2,3,4,5] for i in range(613)])
test_fold = np.hstack((np.array([val for val in [1,2,3,4,5] for i in range(307)]),
                      np.random.randint(1,5,1)))
np.random.seed(0)
np.random.shuffle(train_fold)
np.random.shuffle(test_fold)

# Gradient Descent:
### Gradient of loss function
def sigmoid(z):
    if z >= 0:
        return 1/(1+np.exp(-z))
    else:
        return np.exp(z)/(1+np.exp(z))
          
def gradient (X,Y,w,c): 
    # c is the penalty 
    lw =np.zeros([58,1])
    for i in range(len(Y)):
        z = np.dot(X[i,:].reshape(1,58),w)
        p = sigmoid(z)
        lw = lw + (p-Y[i])*X[i,:].reshape(58,1)
    return lw + c*w
        
def loss (X,Y,w,c):
    loss=0
    for i in range(len(Y)):
        z= np.dot(X[i,:].reshape(1,58),w)
        p= sigmoid(z)
        if Y[i] == 0:
            p = 1-p
        loss= loss - np.log(p)
    mean_loss = loss/len(Y)
    return mean_loss + (c/2)*np.sum(w**2)

def error_rate(X,o_Y,w,thresh):
    p_Y = []
    for i in range(len(o_Y)):
        p = sigmoid(np.dot(X[i,:].reshape(1,58),w))
        if p>thresh:
            p_Y.append(1)
        else:
            p_Y.append(0)
    p_Y = np.array(p_Y)
    error = np.sum(o_Y!=p_Y.reshape(o_Y.shape[0],1))/len(o_Y)
    return error


# L2-regularizer
grid = np.array([i/10 for i in range(0,50,5)])

def logistic_reg(X,Y,c,itera,testX,testY):
    learning_rate = 0.0001
    n_iter = itera
    w = np.zeros([58,1])
    
    for i in range(n_iter):
        gradient_w = gradient(X,Y,w,c)
        w_new = w - learning_rate * gradient_w
        if np.linalg.norm(w_new-w,ord=1) < 0.00001:
            lg_error = error_rate(testX,testY,w_new,0.5)
            print('gradient descent has converged after '
                  +str(i)+
                  ' iterations')
            return (lg_error)
        w = w_new
    lg_error = error_rate(testX,testY,w,0.5)
    return lg_error


def cross_validation(X,Y,c,itera,fold):
    error_list = []
    for k in range(1,6):
        train_X,train_Y = X[fold!=k],Y[fold!=k]
        test_X,test_Y = X[fold==k],Y[fold==k]
        learning_rate = 0.0001
        n_iter = itera
        w = np.zeros([58,1])
        
        for i in range(n_iter):
            gradient_w = gradient(train_X,train_Y,w,c)
            w_new = w - learning_rate * gradient_w
            if np.linalg.norm(w_new-w,ord=1)<0.00001:
                error_list.append(error_rate(test_X,test_Y,w_new,0.5))
                break
            w = w_new
        error_list.append(error_rate(test_X,test_Y,w,0.5))
    return np.mean(error_list)

grid = np.array([i/10 for i in range(0,50,5)])    


# Train set

# Transformation data
X_train,Y_train = train_data['X'],train_data['y']
X_test, Y_test = test_data['X'],test_data['y']

## Standardize
Xtrain_sd = np.hstack((np.ones([3065,1]),preprocessing.scale(X_train,axis=0)))
Xtest_sd = np.hstack((np.ones([1536,1]),preprocessing.scale(X_test,axis=0)))

error_coll_sd = []
for j in range(10):
    c = grid[j]
    error_coll_sd.append(cross_validation(Xtrain_sd,Y_train,c,1000,train_fold))
error_coll_sd
# After that lambda_sd = 0.5
mean_train_error_sd= error_coll_sd[1]

## Log
Xtrain_log = np.hstack((np.ones([3065,1]),transform_log(X_train)))
Xtest_log = np.hstack((np.ones([1536,1]),transform_log(X_test)))

error_coll_log = []
for j in range(10):
    c = grid[j]
    error_coll_log.append(cross_validation(Xtrain_log,Y_train,c,1000,train_fold))
error_coll_log   
# After that lambda_log = 1
mean_train_error_log = error_coll_log[2]


## binary
Xtrain_bin = np.hstack((np.ones([3065,1]),transform_binary(X_train)))
Xtest_bin = np.hstack((np.ones([1536,1]),transform_binary(X_test)))

error_coll_bin = []   
for j in range(10):
    c = grid[j]
    error_coll_bin.append(cross_validation(Xtrain_bin,Y_train,c,1000,train_fold))
error_coll_bin
# After that lambda_binary = 0
mean_train_error_bin = error_coll_bin[0]

# Train set error
train_error_sd = logistic_reg(Xtrain_sd, Y_train, 0.5, 1000, Xtrain_sd, Y_train)
train_error_log = logistic_reg(Xtrain_log, Y_train, 1, 1000, Xtrain_log, Y_train)
train_error_bin = logistic_reg(Xtrain_bin, Y_train, 0, 1000, Xtrain_bin, Y_train)



# Test set error
test_error_sd = logistic_reg(Xtrain_sd, Y_train, 0.5, 1000, Xtest_sd, Y_test)
test_error_log = logistic_reg(Xtrain_log, Y_train, 1, 1000, Xtest_log, Y_test)
test_error_bin = logistic_reg(Xtrain_bin, Y_train, 0, 1000, Xtest_bin, Y_test)

