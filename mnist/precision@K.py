from unit import *
from train import predict, accuracy
result 		=  cPickle.load(open("data/result", "rb"))
weights , h_train, h_valid = result

label_train = one_hot_coding(cPickle.load(open("data/train_labels","rb"))).T
label_test  = one_hot_coding(cPickle.load(open("data/test_labels","rb"))).T
label_train_original	= cPickle.load(open("data/train_labels","rb"))
label_test_original	= cPickle.load(open("data/test_labels","rb"))

def create_database_precision(kkk):
	conv	= cPickle.load(open("data/test_lenet","rb"))
	num_images_show = kkk
	fully 	= predict(weights, conv)
	pre 	= fully[-1]
	hiden1 	= fully[0]
	hiden2 	= fully[1]
	acc_pre = np.max(pre, axis = 1)
	label_pre 	= np.argmax(pre, axis = 1)
	number_query = label_pre.shape[0]

	
	data_pre 		= np.asarray(cPickle.load(open("database/"+"data_h2","rb")))
	sum_all_query = 0
	for i in range(number_query):
		data_softmax 	= np.linalg.norm(data_pre[:,2:data_pre.shape[-1]] -hiden2[i],ord=1, axis = 1)
		data_cosin 		= np.append(data_pre[:,:1], data_softmax.reshape(data_softmax.shape[0],1), axis = 1)
		data_select 	= data_cosin[np.argsort(data_cosin[:,-1])][0:num_images_show]
		array_label_select = (data_select[:,0]).astype(int)
		label_select 	= label_train_original[array_label_select]
		if i %100 == 0:
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

