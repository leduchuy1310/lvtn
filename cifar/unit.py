import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import _pickle as cPickle
from scipy import sparse
from load_cifar import *
np.random.seed(69)
W1 = cPickle.load(open("data/kernel1","rb"))
b1 = cPickle.load(open("data/bias1","rb"))
W2 = cPickle.load(open("data/kernel2","rb"))
b2 = cPickle.load(open("data/bias2","rb"))
W3 = cPickle.load(open("data/kernel3","rb"))
b3 = cPickle.load(open("data/bias3","rb"))

mean_array    = np.array([[125.30691805],[122.95039414],[113.86538318]])
var_array     = np.array([[3968.14567502],[3855.00761641],[4449.54363607]])
label_name = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

A, B, C, D = load_CIFAR10("data/cifar-10-batches-py")
Weights = [W1, b1, W2, b2, W3, b3]

def conv2d(A, W, b=0, stride = 1, pad = 0):
    #Input
    # A ( H1xW1xD1): feature input
    # W (D2,f f): feature kernel
    # b(k,): bias
    # stride: num of strides
    # pad: num of pads
    #print(A.shape)
    # Ouput: A_res(H2xW2xK)
    #print(A.ndim)
    if A.ndim == 4:
        n_input,H_1, W_1, D_1 = A.shape 
        K = W.shape[0]
        f = W.shape[1]
        A_pad = np.pad(A, pad_width=((0,0),(pad,pad),(pad,pad),(0,0)), mode = 'constant', constant_values = 0)
        H_new = int((H_1 - f + 2*pad)/stride) + 1 
        W_new = int((W_1 - f + 2*pad)/stride) + 1 
        A_res = np.zeros((n_input,H_new, W_new, K))
        for k in range(K):
            for h in range(H_new): 
                for w in range(W_new):
                    h_start = h*stride 
                    h_end = h_start + f
                    w_start = w*stride 
                    w_end = w_start + f
                    a_slide = A_pad[:,h_start: h_end, w_start:w_end,:]
                    A_res[:, h, w, k] = np.sum(a_slide *  W[k], axis = (1,2,3)) + b[k]
        return A_res
    else:
        print("Input must be 4 dims in CON2d!!!")

def one_hot_coding(y, C = 10):
  
    Y = sparse.coo_matrix((np.ones_like(y), 
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y 

def maxpool(A, size = 2, stride = 2):
 
    if A.ndim == 4:
        n_input, H_prev, W_prev , D = A.shape
        H_new = int((H_prev - size)/stride) + 1
        W_new = int((W_prev - size)/stride) + 1
        A_res = np.zeros((n_input,H_new,W_new, D))
        for d in range(D):
            r = 0 # row of feature: Height
            for h in range(0,H_prev-size+1, stride):
                c = 0 # collum of feature : Weight
                for w in range (0,W_prev-size+1, stride):
                    a_slide = A[:, h: h+size, w:w +size, d]
                    A_res[:,r,c,d] = np.max(a_slide, axis=(1,2))
                    c = c+1
                r = r +1
        return A_res
    else:
        print("maxpool fail!!!!!") 

def relu(x, derivative=False):
    if(derivative==False):
        return x*(x > 0)
    else:
        return 1*(x > 0)

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A
    
def dropout(x, dropout_percent = 0):
    mask = np.random.binomial([np.ones_like(x)],(1-dropout_percent))[0]
    return x*mask


def extract_feature(A,Weights = Weights):

    W1, b1, W2, b2, W3, b3= Weights
    Image_input = A
    CON_1 = relu(conv2d(A,W1,b1,1,1))
    #CON_1 = batch_norm(CON_1)
    CON_1 = maxpool(CON_1)
    #print(CON_1.shape)
    #CON_1 = Dropout(CON_1, 0.2)
    #print(CON_1.shape)
    CON_2 = relu(conv2d(CON_1,W2,b2,1,1))
    #CON_2 = batch_norm(CON_2)
    #CON_2 = maxpool(CON_2)
    #CON_2 = Dropout(CON_2, 0.2)
    #print(CON_2.shape)
    CON_3 = relu(conv2d(CON_2,W3,b3,1,1))
    # #CON_3 = batch_norm(CON_3)
    #CON_3 = Dropout(CON_3, 0.2)
    CON_3 = maxpool(CON_3)
    return CON_3.reshape(A.shape[0],-1)


def batch_norm(X):
    gamma   = np.ones(X.shape)
    beta    = np.zeros(X.shape)
    out     = np.zeros(X.shape)
    if len(X.shape) == 4:
        mean_array  = np.zeros((X.shape[-1],1))
        std_array   = np.zeros((X.shape[-1],1))
        for i in range(X.shape[-1]):
            mean_array[i]   = np.mean(X[:,:,:,i])
            std_array[i]    = np.var(X[:,:,:,i])
        for i in range(X.shape[-1]):
            out[:,:,:,i]    = (X[:,:,:,i] - mean_array[i]) / (np.sqrt(std_array[i] + 1e-8))
        out = out*gamma + beta
        
    elif len(X.shape) == 3:
        mean_array  = np.zeros((X.shape[-1],1))
        std_array   = np.zeros((X.shape[-1],1))
        for i in range(X.shape[-1]):
            mean_array[i]   = np.mean(X[:,:,i])
            std_array[i]    = np.var(X[:,:,i])
        for i in range(X.shape[-1]):
            out[:,:,i]      = (X[:,:,i] - mean_array[i]) / (np.sqrt(std_array[i] + 1e-8))
        out = out*gamma + beta
    elif len(X.shape) == 2:
        X_flat  = X.reshape(X.shape[0], -1)
        mean    = np.mean(X_flat, axis = 0)
        var     = np.var(X_flat, axis = 0)
        X_norm  = (X_flat - mean)/np.sqrt(var + 1e-8)
        out     = X_norm*gamma + beta
    return out


def data_extract_feature(A, batch_size,Weights=Weights ):
    batch_size = batch_size
    iterations = round(A.shape[0]/batch_size)
    A_res = np.zeros((A.shape[0],4096))
    for i in range(iterations):
        per = i*100/iterations
        print("per: %d %%" %per)
        f = i*batch_size
        l = f+batch_size
        if(l>(A.shape[0]-1)):
                l = A.shape[0]
        Input = A[f:l]
        A_res[f:l] = extract_feature(Input)
    return A_res


def data_prepocessing(X):
    X_out = np.zeros(X.shape)
    for i in range(X.shape[-1]):
        X_out[:,:,:,i] = (X[:,:,:,i] - mean_array[i]) / (np.sqrt(var_array[i] + 1e-8))   
  
    return X_out

# data_train_extract = data_extract_feature(data_prepocessing(C),batch_size = 200)
# cPickle.dump(data_train_extract, open("datatest","wb" ))