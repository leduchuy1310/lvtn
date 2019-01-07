import numpy as np 
import sys
import _pickle as cPickle
model_size 	=  np.array([[6,5,3],	
						[16,5,6]]) 
# So luong lop kernel theo thu tu cua model
# [Soluong kernel: chieu rong kernel: chieu input lop truoc]
# print(model_size)


def create_kernel_bias(model_size):
	n_layer = model_size.shape[0]
	print("Create new kernel!!!")
	for i in range(n_layer):
		# Create W kernel
		fan_in 	= model_size[i,1]*model_size[i,1]*model_size[i,2]
		fan_out = model_size[i,0]*model_size[i,1]*model_size[i,1]
		xavie = np.sqrt(6/(fan_in + fan_out))
		#print(xavie)
		temp_kernel 	= np.random.uniform(-xavie, xavie, [model_size[i,0],model_size[i,1],model_size[i,1], model_size[i,2]])# / (np.sqrt(model_size[i,0] / 2.))
		cPickle.dump(temp_kernel ,open("kernel" + '%d' % (i+1, ),'wb'))

		# Create bias kernel
		#temp_bias = np.random.randn(model_size[i,0])/  (np.sqrt(model_size[i,0] / 2.))
		temp_bias 		=np.zeros((model_size[i,0]))
		cPickle.dump(temp_bias, open("bias" + '%d' % (i+1, ),'wb'))

		# Create gamma batch norm
		# temp_gamma 		=  np.ones((1, model_size[i,0]))
		# cPickle.dump(temp_gamma, open("gamma" + '%d' % (i+1, ),'wb'))
		
		# # Create beta batch norm
		# temp_beta 		= np.zeros((1, model_size[i,0]))
		# cPickle.dump(temp_beta, open("beta" + '%d' % (i+1, ),'wb'))
#	
#create_kernel_bias(model_size)




