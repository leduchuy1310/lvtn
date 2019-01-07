import matplotlib.pyplot as plt
import numpy as np
import time
import _pickle as cPickle
from unit import *
seed = 4096
np.random.seed(seed)


def createWeights(size = seed):
    
    ninputs = size
    wl1 = 1024 
    wl2 = 512
    nclass = 10 
    
    w1 = np.random.randn(ninputs,wl1)/ (np.sqrt((ninputs +wl1)/ 2.))
    b1 = np.zeros([1,wl1])
    w2 = np.random.randn(wl1,wl2)/ (np.sqrt((wl1+wl2) / 2.))
    b2 = np.zeros([1, wl2])
    w3 = np.random.randn(wl2,nclass)/ (np.sqrt((wl2 +nclass)/ 2.))
    b3 = np.zeros([1, nclass])
    
    return [w1,w2,w3,b1,b2,b3]


def predict(weights, x, dropout_percent=0):
    
    w1,w2,w3,b1,b2,b3  = weights 
    
    #1-Hidden Layer
    first = relu(np.dot(x,w1)+b1)
    first = dropout(first, dropout_percent)

    #2-Hidden Layer
    second = relu(np.dot(first,w2)+b2)
    second = dropout(second, dropout_percent)
    
    #Output Layer
    return [first, second, softmax(np.dot(second,w3)+b3)]


def accuracy(output, y):
    hit = 0
    output = np.argmax(output, axis=1)
    y = np.argmax(y, axis=1)
    for y in zip(output, y):
        if(y[0]==y[1]):
            hit += 1
    p = (hit*100)/output.shape[0]
    return p


def cost(Y_predict, Y_right, weights, nabla):
    w1,w2,w3,b1,b2,b3  = weights
    weights_sum_square = np.mean(w1**2) + np.mean(w2**2) + np.mean(w3**2)
    Loss = -np.sum(Y_right.T*np.log(Y_predict.T))/Y_right.T.shape[1] + nabla/2 *  weights_sum_square
    return Loss

def SGD_momentum(weights, x, t, outputs, eta, gamma, L2, cache=None):
    
    w1,w2,w3,b1,b2,b3  = weights
    if(cache==None):
            vw1 = np.zeros_like(w1)
            vw2 = np.zeros_like(w2)
            vw3 = np.zeros_like(w3)
            vb1 = np.zeros_like(b1)
            vb2 = np.zeros_like(b2)
            vb3 = np.zeros_like(b3)
    else:
        vw1,vw2,vw3,vb1,vb2,vb3 = cache
    
    first, second, y = outputs
    w3_delta = (t-y)
   
    w2_error = np.dot(w3_delta,w3.T)
    
    w2_delta = w2_error * relu(second,derivative=True)

    w1_error = np.dot(w2_delta, w2.T)
    w1_delta = w1_error * relu(first,derivative=True)
    
    eta = -eta/x.shape[0]
    
    vw3 = gamma*vw3 + eta * (np.dot(second.T, w3_delta)+ L2*w3)
    vb3 = gamma*vb3 + eta * w3_delta.sum(axis=0)

    vw2 = gamma*vw2 + eta * (np.dot(first.T, w2_delta) + L2*w2)
    vb2 = gamma*vb2 + eta * w2_delta.sum(axis=0)

    vw1 = gamma*vw1 + eta * (np.dot(x.T, w1_delta) + L2*w1)
    vb1 = gamma*vb1 + eta * w1_delta.sum(axis=0)
    
    w3 -= vw3
    b3 -= vb3

    w2 -= vw2
    b2 -= vb2

    w1 -= vw1
    b1 -= vb1
    
    weights = [w1,w2,w3,b1,b2,b3]
    cache   = [vw1,vw2,vw3,vb1,vb2,vb3]
    
    return weights, cache


def run(weights, x_train, y_train, x_valid, y_valid, epochs , nbatchs, l_rate, decay , momentum , l2 , dropout_percent ):

    N       = x_train.shape[0]
    index   = np.arange(N)
    cache   = None
    learning_rate   = l_rate
    print("Train data: %d" % (x_train.shape[0]))
    print("Validation data: %d \n" % (x_valid.shape[0]))
    
    max_accuracy    = 0
    weights_res     = []
    history_train   = [[],[]]
    history_valid   = [[],[]]
    interate      = 1
    for i in range(epochs):
    	
        np.random.shuffle(index)
        time_start = time.time()
        iterations = round(N/nbatchs)
        sum_accu = 0
        sum_loss = 0
        print("\nEpochs: %2d \ %2d - L-rate: %.10f "% (i+1,epochs, l_rate))
        num_print = int(iterations/10)

        for j in range(iterations):
            f = j*nbatchs
            l = f+nbatchs
            if(l>(x_train.shape[0]-1)):
                l = x_train.shape[0]
            x = x_train[index[f:l]]
            y = y_train[index[f:l]]
            outputs = predict(weights, x, dropout_percent)
            loss = cost(outputs[-1], y, weights, l2)
            accuracy_t = accuracy(outputs[-1], y)
            sum_accu += accuracy_t
            sum_loss += loss
            accuracy_train = sum_accu/(j+1)
            loss_train = sum_loss/(j+1)

            history_train[0].append(loss_train)
            history_train[1].append(accuracy_train)
            
            weights, cache = SGD_momentum(weights, x, y, outputs, l_rate, momentum, l2, cache)
            interate +=1
            #l_rate = learning_rate/(1+decay*inter_adam)
            print("per: %d/ %d \t loss: %.4f  acc: %.4f   " % (l, N, loss_train, accuracy_train))

            # if j%num_print == 0:
            #     per = j*100/iterations
            #     a_v = predict(weights, x_valid)
            #     l_v = cost(a_v[-1], y_valid, weights, l2)
            #     print("per: %d %% loss: %.4f  acc: %.4f    l_valid: %.4f " % (per, loss_train, accuracy_train, l_v))
            
    
        # if i >= 2 and i%5 == 0:
            
        #     #l_rate = l_rate*(0.1**(j//30))
        #     l_rate = l_rate/2
        
        l_rate = l_rate - l_rate*decay
        #l_rate = l_rate/(1+decay*j)
        # if j > 50 and j < 80:
        #     l_rate = 0.0005
        # elif j > 80 and j < 100:
        #     l_rate = 0.0001
        # elif j > 100:
        #     l_rate = 0.00005

        outputs = predict(weights, x_valid)
        
        loss_valid = cost(outputs[-1], y_valid, weights, l2)
        accuracy_valid = accuracy(outputs[-1], y_valid)
        if accuracy_valid >= max_accuracy:
            max_accuracy    = accuracy_valid
            w1,w2,w3,b1,b2,b3  = weights
            weights_res = [w1.copy(),w2.copy(),w3.copy(),b1.copy(),b2.copy(),b3.copy()]
            print("change!!!---" + str(max_accuracy))

        history_valid[0].append(loss_valid)
        history_valid[1].append(accuracy_valid)
        time_end = time.time()
        print("loss: %.6f  acc: %.6f - lossValid: %.6f  accValid: %.6f " % (loss_train, accuracy_train, loss_valid, accuracy_valid))
        print("Time: %d"
         % (time_end - time_start))
        
    return [weights_res, history_train, history_valid]




# weights = createWeights()
# l_rate = 0.01
# epochs = 150
# #print(count(data_train[2])[0])
# decay = 0.05
# momentum = 0.9
# L2 = 0.0004
# dropout_percent = 0.2
# nbatchs = 50


# data_train  = cPickle.load(open("datatrain_326464","rb"))
# data_test   = cPickle.load(open("datatest_326464","rb"))

# # label_train = one_hot_coding(cPickle.load(open("label_train","rb"))).T
# # label_test  = one_hot_coding(cPickle.load(open("label_test","rb"))).T

# result = run(weights,data_train, label_train,data_test, label_test,
#              epochs = epochs,nbatchs=nbatchs, l_rate = l_rate, decay = decay,
#              momentum =momentum, l2 = L2,  dropout_percent = dropout_percent)
# cPickle.dump(result,open("result","wb"))
# result =  cPickle.load(open("result", "rb"))
# _, h_train, h_valid = result


# step_plot = np.arange(epochs)*int(data_train.shape[0]/nbatchs)

# plt.figure(1)
# print(np.max(h_valid[1]))
# plt.subplot(211)
# plt.title("Loss " + str(l_rate))
# plt.plot(h_train[0])
# plt.plot(step_plot,h_valid[0])
# plt.axis([0, len(h_train[0]), 0, 2.5])
# plt.subplot(212)
# plt.plot(h_train[1])
# plt.plot(step_plot,h_valid[1])
# plt.axis([0, len(h_train[1]), 0, 100])
# plt.title("Accuracy")
# #plt.show()









