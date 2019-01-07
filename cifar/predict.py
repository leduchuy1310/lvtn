from train import *
from PIL import Image

result 			=  cPickle.load(open("data/result", "rb"))
weights , h_train, h_vaid 	= result

def inputImage(message):
	while True:
		try:
			userInput 	= np.asarray(Image.open("image_test/"+input(message)).resize((32,32)).convert('RGB')) #skimage.io.imread("image_test/"+input(message)).rezise(32,32,3)
		except OSError as e:
			print("Image not found! Try again!")
			continue
		else:
			return userInput 
			break
def inputNumber(message):
	while True:
		try:
			userInput = int(input(message))
		except ValueError:
			print("Not an integer! Try again!")
			continue
		else:
			return userInput 
			break

img = inputImage("Input image name: ")
#img = skimage.transform.resize(skimage.io.imread("image_test/cat_1.png"), ((32,32)))#.reshape(32,32,3)
# # from PIL import Image
# # img = np.asarray(Image.open('image_test/3.jpg').resize((32,32)))
# num_images_show = 10
num_images_show = inputNumber("Choose the number: \n \t0: Show 10 images \n \t1: Show 20 images \nYour choose:")
while num_images_show != 1 and num_images_show != 0:
	num_images_show = inputNumber("Choose the number: \n \t0: Show 10 images \n \t1: Show 20 images\nYour choose:")

if num_images_show == 0:
	num_images_show = 10
else:
	num_images_show = 20

num_plt 		= int(np.sqrt(num_images_show) + 1)
img_preprocessing 	= np.zeros(img.shape)
for i in range(3):
	img_preprocessing[:,:,i] = (img[:,:,i] - mean_array[i])/ (np.sqrt(var_array[i] + (1e-8)))
conv 			= extract_feature(img_preprocessing.reshape(1,32,32,3))
fully			= predict(weights, conv)
pre 			= fully[-1]
hiden2 			= fully[1]
hiden1 			= fully[0]
label_pre 		= np.argmax(pre)
acc_pre 		= np.max(pre)


sencond_pre 	= np.append(np.arange(10).reshape(10, 1), pre.reshape(10,1), axis = 1)
acc_pre_second 	= sencond_pre[np.argsort(sencond_pre[:,-1])][::-1][1,1]
label_pre_second 	= int(sencond_pre[np.argsort(sencond_pre[:,-1])][::-1][1,0])

data_pre 		= np.asarray(cPickle.load(open("database/"+"data_F2_"+str(label_pre),"rb")))

data_softmax 	= np.linalg.norm(data_pre[:,2:data_pre.shape[-1]] - hiden2, axis = 1)
data_cosin 			= np.append(data_pre[:,:1], data_softmax.reshape(data_softmax.shape[0],1), axis = 1)
data_select 	= data_cosin[np.argsort(data_cosin[:,-1])][0:num_images_show]

fig 	= plt.figure(figsize = (8, 8))
fig.suptitle("Label: " + label_name[label_pre]+ " - " +"Accuracy: " + str(round(acc_pre*100,5))+"%")
plt.subplot(num_plt,2,1)
plt.imshow(img.astype(int), 'gray')
plt.xticks([])
plt.yticks([])
plt.subplot(num_plt, 2,2)
array_plot 	= np.append(np.arange(len(pre[0])).reshape(10,1), pre[0].reshape(10,1), axis = 1)
array_plot 	= array_plot[np.argsort(array_plot[:,-1])][::-1][0:3][::-1]

y_pos 		= np.arange(3)
performance = np.round(array_plot[:,1]*100, decimals = 5)
plt.barh(y_pos, performance, align='center', alpha=0.5)
as_int 		= array_plot[:,0].astype(int)
label_X 	= [label_name[index] for index in as_int ]

plt.yticks(y_pos, label_X)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
for i, v in enumerate(performance):
    plt.text(v ,i, str(v)+"%", va='center')
for i in range(len(data_select)):
	fig.add_subplot(num_plt,num_plt,num_plt+i+1)
	plt.imshow(A[int(data_select[i,0])],'gray')
	plt.xticks([])
	plt.yticks([])
plt.show()