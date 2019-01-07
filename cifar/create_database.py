from unit import *
from train import predict, accuracy
result 		=  cPickle.load(open("data/result", "rb"))
weights , h_train, h_valid = result

label_train = one_hot_coding(cPickle.load(open("data/label_train","rb"))).T
label_test  = one_hot_coding(cPickle.load(open("data/label_test","rb"))).T
label_original	= cPickle.load(open("data/label_train","rb"))


database 	= [],[],[],[],[],[],[],[],[],[]
folder_database 	= "database"

def create_database_F3():
	conv	= cPickle.load(open("data/datatrain_326464","rb"))
	fully 	= predict(weights, conv)
	pre 	= fully[-1]
	acc_pre = np.max(pre, axis = 1)
	label_pre 	= np.argmax(pre, axis = 1)
	z = accuracy(label_train, pre)
	for i in range(conv.shape[0]):
		array_temp = np.append((i,label_pre[i]), pre[i])
		database[label_pre[i]].append(array_temp)
		if i%200 == 0:
			print(str(i*100/conv.shape[0]) + "%")
	if not os.path.exists(folder_database):
			os.makedirs (folder_database)
	for i in range(len(database)):
		cPickle.dump(database[i],open(folder_database+"/"+"data_F3_"+str(i),"wb"))


def create_database_F2():
	conv	= cPickle.load(open("data/datatrain_326464","rb"))
	fully 	= predict(weights, conv)
	pre 	= fully[-1]
	hiden2 	= fully[1]
	label_pre 	= np.argmax(pre, axis = 1)
	for i in range(conv.shape[0]):
		array_temp = np.append((i,label_pre[i]), hiden2[i])
		database[label_pre[i]].append(array_temp)
		if i%200 == 0:
			print(str(i*100/conv.shape[0]) + "%")
	if not os.path.exists(folder_database):
			os.makedirs (folder_database)
	for i in range(len(database)):
		cPickle.dump(database[i],open(folder_database+"/"+"data_F2_"+str(i),"wb"))


def create_database_F1():
	conv	= cPickle.load(open("data/datatrain_326464","rb"))
	fully 	= predict(weights, conv)
	pre 	= fully[-1]
	hiden1 	= fully[0]
	label_pre 	= np.argmax(pre, axis = 1)
	for i in range(conv.shape[0]):
		array_temp = np.append((i,label_pre[i]), hiden1[i])
		database[label_pre[i]].append(array_temp)
		if i%200 == 0:
			print(str(i*100/conv.shape[0]) + "%")
	if not os.path.exists(folder_database):
			os.makedirs (folder_database)
	for i in range(len(database)):
		cPickle.dump(database[i],open(folder_database+"/"+"data_F1_"+str(i),"wb"))


#create_database_precision()
# create_database_F2()
#create_database_F1()

