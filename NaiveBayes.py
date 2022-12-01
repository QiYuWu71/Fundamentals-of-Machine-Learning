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


## data binding
# binary data
Xtrain_bin = transform_binary(X_train)
Xtest_bin = transform_binary(X_test)

# log data
Xtrain_log = transform_log(X_train)
Xtest_log = transform_log(X_test)


# standard data
Xtrain_sd = preprocessing.scale(X_train,axis=0)
Xtest_sd = preprocessing.scale(X_test,axis=0)


# a. binary naive bayes
def prior(Y):
    prior0 = np.sum(Y==0)/len(Y)
    prior1 = np.sum(Y==1)/len(Y)
    
    return prior0,prior1

def miu(X,Y):
    cla_0 = Y==0
    cla_1 = Y==1
    miu_0,miu_1 = [],[]
    for i in range(X.shape[1]):
        x = X[:,i]
        x0,x1 = x.reshape(x.shape[0],1)[cla_0],x.reshape(x.shape[0],1)[cla_1]
        n0,n1 = np.mean(x0),np.mean(x1)
        miu_0.append(n0) 
        miu_1.append(n1)
    
    return miu_0,miu_1
            

def naive_classifier(X,p,miu):
    poster = 0
    for i in range(X.shape[1]):
        if X[:,i]==0:
            poster += np.log(1-miu[i])
        else:
            poster += np.log(miu[i])
    
    classifier = poster + np.log(p)
    
    return classifier
        

def predict_bin(trainx,trainy,testx,testy):
    p0,p1 = prior(trainy)
    miu_0,miu_1 = miu(trainx,trainy)
    predy = []
    for i in range(len(testy)):
        x = testx[i,:].reshape(1,testx.shape[1])
        classifier_0 = naive_classifier(x,p0,miu_0)
        classifier_1 = naive_classifier(x,p1,miu_1)
        
        if classifier_0 > classifier_1:
            predy.append(0)
        else:
            predy.append(1)
    predy=np.array(predy).reshape(len(predy),1)
    error_rate= np.sum(predy!=testy)/len(testy)
    
    return error_rate

train_error_bin = predict_bin(Xtrain_bin,Y_train,Xtrain_bin,Y_train)
test_error_bin = predict_bin(Xtrain_bin,Y_train,Xtest_bin,Y_test)


# b. Guassian Naive bayes 
def sigma2(X,Y):
    cla_0 = Y==0
    cla_1 = Y==1
    sigma2_0,sigma2_1 = [],[]
    
    for i in range(57):
        x = X[:,i].reshape(X.shape[0],1)
        x0,x1 = x[cla_0],x[cla_1]
        sigma2_0.append(np.var(x0))
        sigma2_1.append(np.var(x1))
    return sigma2_0,sigma2_1

def bayes_classifier(X,p,miu,sigma2):
    poster = 0
    for i in range(57):
        posteri = (X[:,i]- miu[i])**2 / sigma2[i]
        poster = poster + posteri
        
    classifier  = np.log(p)-poster*0.5
    return classifier


def predict_gua(trainx,trainy,testx,testy):
    p0,p1 = prior(trainy)
    miu0,miu1 = miu(trainx,trainy)
    sigma2_0,sigma2_1 = sigma2(trainx,trainy)
    predy = []
    
    for i in range(len(testy)):
        x = testx[i,:].reshape(1,testx.shape[1])
        classifier_0 = bayes_classifier(x,p0,miu0,sigma2_0)
        classifier_1 = bayes_classifier(x,p1,miu1,sigma2_1)
        if classifier_0 > classifier_1:
            predy.append(0)
        else:
            predy.append(1)
            
    predy = np.array(predy).reshape(len(testy),1)
    error_rate = np.sum(predy!=testy)/len(testy)
    return error_rate

train_error_sd = predict_gua(Xtrain_sd,Y_train,Xtrain_sd,Y_train)
test_error_sd = predict_gua(Xtrain_sd,Y_train,Xtest_sd,Y_test)

test_error_log = predict_gua(Xtrain_log,Y_train,Xtrain_log,Y_train)
train_error_log = predict_gua(Xtrain_log,Y_train,Xtest_log,Y_test)

