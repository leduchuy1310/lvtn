import _pickle as cPickle
import numpy as np
import random
import os
import gzip
random.seed(1) # set a seed so that the results are consistent

def extractCategories(path, file):
    f     = open(path+file, 'rb')
    dict  = cPickle.load(f)
    return dict['label_names']
    categories  = extractCategories("cifar-10-batches-py/", "batches.meta")
    print(categories)

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = cPickle.load(f,encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3,32, 32).transpose(0,2,3,1).astype("uint8")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f     = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y  = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr     = np.concatenate(xs)
  Ytr     = np.concatenate(ys)
  del X, Y
  Xte, Yte  = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

# def normal_min_max(A):
#   A = A.reshape(A.shape[0], 32*32*3)
#   A_normal = np.zeros(A.shape)
#   for i in range(A.shape[0]):
#     mean = np.max(A[i]) - np.min(A[i])
  
#     for j in range(A.shape[1]):
#       A_normal[i,j] = float(A[i,j]/mean)
#   return A_normal.reshape(A.shape[0],32,32,3)
  
# def normalization(X):
    
#     mean = np.mean(X)
#     var = np.var(X)
#     X_norm = np.zeros_like(X)
#     X_norm = (X - mean)/np.sqrt(var+(1e-5))
#     return X_norm

# def normal_z_core(X):
#   X_res = np.zeros_like(X)
#   for i in range(X.shape[0]):
#     X[i] = normalization(X[i])
#   return X_res


