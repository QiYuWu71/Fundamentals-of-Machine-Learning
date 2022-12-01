import _pickle as cPickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import random

# (a) Implementing multi-class classification using backpropagation
# Initial setting for my neural network:
    #contains 784 neurons for input layer, one hidden layer, 10 neurons for output layer.
    #cost_function is MSE
    
# activation function collection
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))
    
def relu(x):
    return np.maximum(0,x)
    
def relu_prime(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def tanh_prime(x):
    return 1- tanh(x)*tanh(x)

def softmax(x):
    max_x = np.max(x)
    x = x - max_x
    demoni = np.sum(np.exp(x))
    probs = []
    for i in x:
        indi_prob = np.exp(i)/demoni
        probs.append(indi_prob)
    return np.array(probs)

# Construct netural network
class Network(object):
    def __init__(self,sizes,cost): # sizes is a list contains the number of neurons in the respective layers.
        self.num_layers=len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn (y, x) for x,y in zip(sizes[:-1],sizes[1:])]
        self.cost = cost
  
    def activiation(self,names='sigmoid'):
        if names == 'sigmoid':
            self.activiation = sigmoid
            self.prime = sigmoid_prime
        elif names == 'relu':
            self.activiation = relu
            self.prime = relu_prime
            
        else:
            self.activiation = tanh
            self.prime = tanh_prime
    
    def MSE_gradient(self,x,y):
        y_pred = np.argmax(x)
        output_diff = x-y
        softg = []
        for i in range(y.shape[0]):
            if i == y_pred:
                x[i]*(1-x[i])
                softg.append(x[i]*(1-x[i]))
            else:
                softg.append(-x[i]*x[y_pred])
            
        self.delta = output_diff * np.array(softg)
        delta = self.delta
        return delta.reshape(y.shape[0],1)
        
    def cross_gradient(self,x,y):
        return x-y
        
        
    def feedforward(self,a): #softmax for the last layer
        all_a = []
        for b,w in zip(self.biases[:-1],self.weights[:-1]):
            a = self.activiation(np.dot(w,a)+b)
            all_a.append(a)
        b_L,w_L = self.biases[-1],self.weights[-1]
        y = np.dot(w_L,all_a[-1])+b_L
        
        return np.argmax(softmax(y))
    
# needs modification

    def backprop(self,x,y): 
    # feedforward
        input = x
        all_input = [x]
        all_z = []
        
        for b,w in zip(self.biases[:-1],self.weights[:-1]):
            
            z = np.dot(w,input)+b
            all_z.append(z)
            input = self.activiation(z)
            all_input.append(input)
            
        b_l,w_l = self.biases[-1],self.weights[-1]
        z_l = np.dot(w_l,all_input[-1])+b_l
        all_z.append(z_l)
        y_pred = softmax(z_l)
        all_input.append(y_pred)
        self.z = all_input
    # backward
        decre_b = [np.zeros(b.shape) for b in self.biases]
        decre_w = [np.zeros(w.shape) for w in self.weights]
        
    # gradient update
        if self.cost == 'MSE':
            delta = self.MSE_gradient(all_input[-1],y)
        else:
            delta= self.cross_gradient(all_input[-1],y)
            
        decre_b[-1]=delta
        decre_w[-1]=np.dot(delta,all_input[-2].transpose())
        
        for i in range(2,self.num_layers):
            z = all_z[-i]
            ag = self.prime(z)
            delta = np.dot(self.weights[-i+1].transpose(),delta)*ag
            
            decre_b[-i] = delta
            decre_w[-i] = np.dot(delta,all_input[-i-1].transpose())
    
        return (decre_b,decre_w)
    
    def SGD(self,training_data,epochs,mini_batch_size,eta,validation_data,test_data,lmbda=0.0):
        n = len(training_data)
        test_accuracy,validation_accuracy = [],[]
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batche_set = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batche_set:
                self.update_mini_batch(mini_batch,eta,lmbda,n)
            test_accuracy.append(self.evaluate(test_data))
            validation_accuracy.append(self.evaluate(validation_data))
            
        return (np.array(validation_accuracy),np.array(test_accuracy))
                
    def update_mini_batch(self,mini_batch,eta,lmbda,n):
        sum_decre_b = [np.zeros(b.shape) for b in self.biases]
        sum_decre_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_b,delta_w = self.backprop(x,y)
            sum_decre_b = [nb+dnb for nb,dnb in zip(sum_decre_b,delta_b)]
            sum_decre_w = [nw+dnw for nw,dnw in zip(sum_decre_w,delta_w)]
        
        self.biases = [b - (eta/len(mini_batch))*nb
                       for b,nb in zip(self.biases,sum_decre_b)]
        self.weights = [w-(eta/len(mini_batch))*nw-eta*(lmbda/n)*w 
                        for w,nw in zip(self.weights,sum_decre_w)]
        
    def evaluate(self,test_data):
        results = [(self.feedforward(x),y) for (x,y) in test_data]
        logi = [x==y for (x,y) in results]
        return sum(logi) /len(results)
    
    

# (b) Data loader
def load_data():
    f = gzip.open('mnist.pkl.gz','rb')
    training_data,validation_data,test_data=cPickle.load(f,encoding='bytes')
    f.close()
    return (training_data,validation_data,test_data)

# data transformation
def one_k_encode(y):
    k = np.zeros((10,1))
    k[y] = 1
    return k

index = np.random.randint(0,50001,size=5000)

train_data,validation_data,test_data = load_data()
train_x = [x.reshape(784,1) for x in train_data[0]]
train_y = [one_k_encode(y) for y in train_data[1]]


train = list(zip(train_x[0:5000],train_y[0:5000]))

validation_x = [x.reshape(784,1) for x in validation_data[0]]
validation = list(zip(validation_x[0:1000],validation_data[1][0:1000]))

test_x = [x.reshape(784,1) for x in test_data[0]]
test = list(zip(test_x[0:1000],test_data[1][0:1000]))


# (c) Train the neural network using minibatch stochastic gradient descent(SGD)
#Network(sizes,costname)
#SGD(self,training_data,epochs,mini_batch_size,eta,validation_data,test_data,lmbda)

# different number of hidden layers (one or two)
net_layer1 = Network([784,50,10],'C')
net_layer1.activiation('sigmoid')
layer1vc,layer1tc = net_layer1.SGD(train,30,100,1, validation, test)

net_layer2 = Network([784,50,50,10],'C')
net_layer2.activiation('sigmoid')
layer2vc,layer2tc = net_layer1.SGD(train,30,100,1, validation, test)

# plot
epoch = np.linspace(1,30,30)
plt.ylim(0.50,1.00)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(epoch,layer1vc,c='b',label='1 hidden layer (validation)')
plt.plot(epoch,layer1tc,ls = '--',c='b',label='1 hidden layer (test)')
plt.plot(epoch,layer2vc,c='r',label='2 hidden layer (validation)')
plt.plot(epoch,layer2tc,ls = '--',c='r',label = '2 hidden layer (test)')
plt.legend(loc='lower right')
plt.show()


# different activation functions(sigmoid,tanh or relu)
net_sig = Network([784,50,10],'C')
net_sig.activiation('sigmoid')
sigvc,sigtc = net_sig.SGD(train,30,100,1, validation, test)

net_tanh = Network([784,50,10],'C')
net_tanh.activiation('tanh')
tanhvc,tanhtc = net_tanh.SGD(train,30,100,1, validation, test)

net_relu = Network([784,50,10],'C')
net_relu.activiation('relu')
reluvc,relutc = net_relu.SGD(train,30,100,1, validation, test)

#plot
epoch = np.linspace(1,30,30)
plt.ylim(0.50,1.00)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(epoch,sigvc,c='b',label='sigmoid (validation)')
plt.plot(epoch,sigtc,ls = '--',c='b',label='sigmoid (test)')
plt.plot(epoch,tanhvc,c='r',label='tanh (validation)')
plt.plot(epoch,tanhtc,ls = '--',c='r',label='tanh (test)')
plt.plot(epoch,reluvc,c='g',label='relu (validation)')
plt.plot(epoch,relutc,ls = '--',c='g',label='relu (test)')
plt.legend(loc='lower right')
plt.show()


# different hidden nodes in each hidden layer
net_hid50 = Network([784,50,10],'C')
net_hid50.activiation('sigmoid')
hid50vc,hid50tc = net_hid50.SGD(train,30,100,1, validation, test)


net_hid100 = Network([784,100,10],'C')
net_hid100.activiation('sigmoid')
hid100vc,hid100tc = net_hid100.SGD(train,30,100,1, validation, test)

#plot
#3 hidden nodes
epoch = np.linspace(1,30,30)
plt.ylim(0.50,1.00)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(epoch,hid50vc,c='b',label='50 hidden nodes (validation)')
plt.plot(epoch,hid50tc,ls = '--',c='b',label='50 hidden nodes (test)')
plt.plot(epoch,hid100vc,c='r',label='100 hidden nodes (validation)')
plt.plot(epoch,hid100tc,ls = '--',c='r',label='100 hidden nodes (validation)')
plt.legend(loc='lower right')
plt.show()


# different learning rate(0.001, 0.01,0.1,1.0 or 3.0)
net_eta0001 = Network([784,50,10],'C')
net_eta0001.activiation('sigmoid')
eta0001vc,eta0001tc = net_eta0001.SGD(train,30,100,0.001, validation, test)

net_eta001 = Network([784,50,10],'C')
net_eta001.activiation('sigmoid')
eta001vc,eta001tc = net_eta001.SGD(train,30,100,0.01, validation, test)

net_eta01 = Network([784,50,10],'C')
net_eta01.activiation('sigmoid')
eta01vc,eta01tc = net_eta01.SGD(train,30,100,0.1, validation, test)

net_eta1 = Network([784,50,10],'C')
net_eta1.activiation('sigmoid')
eta1vc,eta1tc = net_eta1.SGD(train,30,100,1, validation, test)

net_eta3 = Network([784,50,10],'C')
net_eta3.activiation('sigmoid')
eta3vc,eta3tc = net_eta3.SGD(train,30,100,3, validation, test)

# plot
#4 learning rate
epoch = np.linspace(1,30,30)
plt.ylim(0.00,1.00)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(epoch,eta0001vc,c='b',label='0.001 etha (validation)')
plt.plot(epoch,eta0001tc,ls = '--',c='b',label='0.001 etha (test)')
plt.plot(epoch,eta001vc,c='r',label='0.01 etha (validation)')
plt.plot(epoch,eta001tc,ls = '--',c='r',label='0.01 etha (test)')
plt.plot(epoch,eta01vc,c='g',label='0.1 etha (validation)')
plt.plot(epoch,eta01tc,ls = '--',c='g',label='0.1 etha (test)')
plt.plot(epoch,eta1vc,c='c',label='1 etha (validation)')
plt.plot(epoch,eta1tc,ls = '--',c='c',label='1 etha (test)')
plt.plot(epoch,eta3vc,c='y',label='3 etha (validation)')
plt.plot(epoch,eta3tc,ls = '--',c='y',label='3 etha (test)')
plt.legend(loc='lower right')
plt.show()


# different epochs(30,100)
net_epoch30 = Network([784,50,10],'C')
net_epoch30.activiation('sigmoid')
epoch30vc,epoch30tc = net_epoch30.SGD(train,30,100,1, validation, test)

net_epoch100 = Network([784,50,10],'C')
net_epoch100.activiation('sigmoid')
epoch100vc,epoch100tc = net_epoch100.SGD(train,100,100,1, validation, test)

# plot
#5 epoch 
epoch1 = np.linspace(1,30,30)
epoch2 = np.linspace(1,100,100)
plt.ylim(0.50,1.00)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(epoch1,epoch30vc,c='b',label='30 epoch (validation)')
plt.plot(epoch1,epoch30tc,ls = '--',c='b',label='30 epoch (test)')
plt.plot(epoch2,epoch100vc,c='r',label='100 epoch (validation)')
plt.plot(epoch2,epoch100tc,ls = '--',c='r',label='100 epoch (test)')
plt.legend(loc='lower right')
plt.show()


# different minibatch size
net_minib10 = Network([784,50,10],'C')
net_minib10.activiation('sigmoid')
minib10vc,minib10tc = net_minib10.SGD(train,30,10,1, validation, test)

net_minib50 =  Network([784,50,10],'C')
net_minib50.activiation('sigmoid')
minib50vc,minib50tc = net_minib50.SGD(train,30,50,1, validation, test)

net_minib100 =  Network([784,50,10],'C')
net_minib100.activiation('sigmoid')
minib100vc,minib100tc = net_minib100.SGD(train,30,100,1, validation, test)

#6 minibatch size
epoch = np.linspace(1,30,30)
plt.ylim(0.50,1.00)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(epoch,minib10vc,c='b',label='10 Minibatch Size (validation)')
plt.plot(epoch,minib10tc ,ls = '--',c='b',label='10 Minibatch Size (test)')
plt.plot(epoch,minib50vc,c='r',label='50 Minibatch Size (validation)')
plt.plot(epoch,minib50tc,ls = '--',c='r',label='50 Minibatch Size (test)')
plt.plot(epoch,minib100vc,c='g',label='100 Minibatch Size (validation)')
plt.plot(epoch,minib100tc,ls = '--',c='g',label='100 Minibatch Size (test)')
plt.legend(loc='lower right')
plt.show()


# different cost (crossEntropy or MSE)
net_MSE =  Network([784,50,10],'MSE')
net_MSE.activiation('sigmoid')
MSEvc,MSEtc = net_MSE.SGD(train,50,10,1, validation, test)

net_crossE = Network([784,50,10],'C')
net_crossE.activiation('sigmoid')
crossEvc,crossEtc = net_crossE.SGD(train,50,10,1, validation, test)


#plot
#7 Loss function
epoch = np.linspace(1,50,50)
plt.ylim(0.00,1.00)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(epoch,MSEvc,c='y',label = 'MSE (validation)')
plt.plot(epoch,MSEtc ,ls = '--',c='y', label = 'MSE (test)')
plt.plot(epoch,crossEvc,c='c',label='Cross Entropy (validation)')
plt.plot(epoch,crossEtc,ls = '--',c='c',label='Cross Entropy (test)')
plt.legend(loc='lower right')
plt.show()

# regularization or not (L2 regularization)
network = Network([784,50,10],'C')
network.activiation('sigmoid')
lmbda01vc,lmbda01tc = network.SGD(train,30,100,1, validation, test,0.1)
lmbda05vc,lmbda05tc = network.SGD(train,30,100,1, validation, test,0.5)
noregularvc,noregulartc = network.SGD(train,30,100,1, validation, test,0)
lmbda1vc,lmbda1tc = network.SGD(train,30,100,1, validation, test,1)
lmbda5vc,lmbda5tc = network.SGD(train,30,100,1, validation, test,5)

#8 lambda
epoch = np.linspace(1,30,30)
plt.ylim(0.60,1.00)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(epoch,lmbda01vc,c='b',label='0.1 lambda (validation)')
plt.plot(epoch,lmbda01tc,ls = '--',c='b',label='0.1 lambda (test)')
plt.plot(epoch,lmbda05vc,c='r',label='0.5 lambda (validation)')
plt.plot(epoch,lmbda05tc,ls = '--',c='r',label='0.5 lambda (test)')
plt.plot(epoch,noregularvc,c='g',label='0 lambda (validation)')
plt.plot(epoch,noregularvc,ls = '--',c='g',label='0 lambda (test)')
plt.plot(epoch,lmbda1vc,c='c',label='1 lambda (validation)')
plt.plot(epoch,lmbda1tc,ls = '--',c='c',label='1 lambda (test)')
plt.plot(epoch,lmbda5vc,c='y',label= '5 lambda (validation)')
plt.plot(epoch,lmbda5tc,ls = '--',c='y',label='5 lambda (test)')
plt.legend(loc='lower right')
plt.show()



# (d) Final steps: select the optimal model
# Choose the optimal parameters for the final model.
# optimal choose for learning rate
network = Network([784,50,50,10],'C')
network.activiation('sigmoid')
fetha1vc,fetha1tc = network.SGD(train,50,10,1, validation, test)
fetha3vc,fetha3tc = network.SGD(train,50,10,3, validation, test)

epoch = np.linspace(1,50,50)
plt.ylim(0.70,1.00)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(epoch,fetha1vc,c='y',label = 'fetha1 (validation)')
plt.plot(epoch,fetha1tc,ls = '--',c='y', label = 'fetha1 (test)')
plt.plot(epoch,fetha3vc,c='c',label='fetha3 (validation)')
plt.plot(epoch,fetha3tc,ls = '--',c='c',label='fetha3 (test)')
plt.legend(loc='lower right')
plt.show()

# final option is 3.0

# optimal selection of regularization
network = Network([784,50,50,10],'C')
network.activiation('sigmoid')
lmbda01vc,lmbda01tc = network.SGD(train,50,10,3, validation, test,0.1)
lmbda1vc,lmbda1tc = network.SGD(train,50,10,3, validation, test,1)
lmbda5vc,lmbda5tc = network.SGD(train,50,10,3, validation, test,5)

epoch = np.linspace(1,50,50)
plt.ylim(0.60,1.00)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(epoch,fetha3vc,c='b',label='0 lambda (validation)')
plt.plot(epoch,fetha3tc,ls = '--',c='b',label='0 lambda (test)')
plt.plot(epoch,lmbda01vc,c='r',label='0.1 lambda (validation)')
plt.plot(epoch,lmbda01tc,ls = '--',c='r',label='0.1 lambda (test)')
plt.plot(epoch,lmbda1vc,c='c',label='1 lambda (validation)')
plt.plot(epoch,lmbda1tc,ls = '--',c='c',label='1 lambda (test)')
plt.plot(epoch,lmbda5vc,c='y',label= '5 lambda (validation)')
plt.plot(epoch,lmbda5tc,ls = '--',c='y',label='5 lambda (test)')
plt.legend(loc='lower right')
plt.show()

# Since we prefer stable performance about fitting, we will choose the training model without regularization.

# Final Model with Optimal Parameter :)

Nice_network = Network([784,50,50,10],'C')
Nice_network.activiation('sigmoid')
Optimalvc,Optimaltc = Nice_network.SGD(train,50,10,1, validation, test)

# Accuracy vs. Epoch
epoch = np.linspace(1,50,50)
plt.ylim(0.50,1.00)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.plot(epoch,Optimalvc,label='Validation accuracy')
plt.plot(epoch,Optimaltc,ls = '--',label='Test accuracy')
plt.legend(loc='lower right')
plt.show()



