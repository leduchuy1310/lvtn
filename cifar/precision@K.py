from unit import *
from train import predict, accuracy

from scipy.spatial import distance
result 		=  cPickle.load(open("data/result", "rb"))
weights , h_train, h_valid = result

label_train = one_hot_coding(cPickle.load(open("data/label_train","rb"))).T
label_test  = one_hot_coding(cPickle.load(open("data/label_test","rb"))).T
label_train_original	= cPickle.load(open("data/label_train","rb"))
label_test_original		= cPickle.load(open("data/label_test","rb"))

def create_database_precision(Kprecision):
	conv	= cPickle.load(open("data/datatest_326464","rb"))
	num_images_show = Kprecision
	fully 	= predict(weights, conv)
	pre 	= fully[-1]
	hiden1 	= fully[0]
	hiden2 	= fully[1]
	acc_pre = np.max(pre, axis = 1)
	label_pre 	= np.argmax(pre, axis = 1)
	number_query = label_pre.shape[0]

	sum_all_query = 0
	for i in range(number_query):
		data_pre 		= np.asarray(cPickle.load(open("database/"+"data_F1_"+str(label_pre[i]),"rb")))
		data_softmax 	= np.linalg.norm(data_pre[:,2:data_pre.shape[-1]] - hiden1[i],ord=1, axis = 1)
	

		data_cosin 		= np.append(data_pre[:,:1], data_softmax.reshape(data_softmax.shape[0],1), axis = 1)
		data_select 	= data_cosin[np.argsort(data_cosin[:,-1])][0:num_images_show]
		array_label_select = (data_select[:,0]).astype(int)
		label_select 	= label_train_original[array_label_select]
		if i %1000 == 0:
			print(i*100/number_query)
		sum_one_query = 0
		for j in range(label_select.shape[0]):
			if (label_select[j] == label_test_original[i] ):
				sum_all_query +=1
	sum_sum = sum_all_query/(number_query*label_select.shape[0])
	print(sum_sum)

# create_database_precision(1)
# create_database_precision(10)
# create_database_precision(50)

def confu_matrix():
	res = np.zeros((10,10))
	conv	= cPickle.load(open("data/datatrain_326464","rb"))
	#num_images_show = 100
	fully 	= predict(weights, conv)
	pre 	= fully[-1]
	hiden1 	= fully[0]
	acc_pre = np.max(pre, axis = 1)
	label_pre 	= np.argmax(pre, axis = 1)
	number_query = label_pre.shape[0]
	data_pre 		= np.asarray(cPickle.load(open("database/"+"data","rb")))
	for i in range(label_train_original.shape[0]):
		res[label_train_original[i],label_pre[i]] +=1
	print(res.astype(int))
