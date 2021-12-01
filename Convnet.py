#!/usr/bin/env python3
import numpy as np
from PIL import Image
import os

			#image initialization
def Image_initialization_Batch_Normalization(a,b):
	directory = '/Users/jim/Downloads/Python/Project/Training Set'
	image = []
	for filename in os.listdir(directory):
		if filename.endswith(".png") or filename.endswith(".jpg"):
			temp_Image = Image.open("/Users/jim/Downloads/Python/Project/Training Set/"+filename)
			temp_Image = temp_Image.convert("RGB")
			data = np.array(temp_Image)
			i = np.shape(data)[0]//100
			j = np.shape(data)[1]//100
			data_Processed = np.zeros((100,100,3))
			for m in range(0,100):
				for n in range(0,100):
					data_Processed[m,n] = data[m*i,n*j]
			image.append(data_Processed)
	average = np.zeros((100,100,3))
	standard_Deviation = np.zeros((100,100,3))
	for i in range(len(image)):
		for x in range(100):
			for y in range(100):
				for z in range(3):
					average[x,y,z] += image[i][x,y,z]/len(image)
					standard_Deviation[x,y,z] += (image[i][x,y,z]-average[x,y,z])**2/len(image)
					if i == len(image)-1:
						standard_Deviation[x,y,z] = np.sqrt(standard_Deviation[x,y,z])
						image[i][x,y,z] = (image[i][x,y,z]-average[x,y,z])/np.sqrt(standard_Deviation[x,y,z]**2+0.001)
						image[i][x,y,z] = image[i][x,y,z]*a+b
	return image

def Conv(x,w):
	x_Padded = np.zeros((np.shape(x)[0]+2,np.shape(x)[1]+2,np.shape(x)[2]))
	conv_Outcome = np.zeros((np.shape(x)[0],np.shape(x)[1]))
	x_Padded[1:np.shape(x)[0]+1,1:np.shape(x)[1]+1,:] = x
	for i in range(np.shape(x)[0]):
		for j in range(np.shape(x)[1]):
			for k in range(np.shape(x)[2]):
				for m in range(0,3):
					for n in range(0,3):
						conv_Outcome[i,j] += w[m,n]*(x_Padded[i+m,j+n,k])
	return conv_Outcome

def MaxPooling(x):                                #input must be mean*mean
	pooling_Outcome = np.zeros((np.shape(x)[0]//2,np.shape(x)[1]//2,np.shape(x)[2]))
	for k in range(0,np.shape(x)[2]):
		for i in range(0,np.shape(x)[0],2):
				for j in range(0,np.shape(x)[1],2):
					pooling_Outcome[i//2,j//2,k] = max(x[i,j,k],x[i+1,j,k],x[i,j+1,k],x[i+1,j+1,k])
	return pooling_Outcome

def SVM(x,w):
	x = x.flatten()				#default correct label : first line
	L = 0
	R = 0
	for i in range(1,np.shape(x)[0]):
		L += max(0,x[i]-x[1]+1)
	for i in range(3):
		for j in range(3):
			R += abs(w[i,j])
	Loss = L + R
	return Loss

def ReLU(x):
	for k in range(np.shape(x)[2]):
		for i in range(np.shape(x)[0]):
			for j in range(np.shape(x)[1]):
				x[i,j,k] = max(0,x[i,j,k])
	return x

#init
#Already Xavier Initializtaion
a = 1
b = 0
image = Image_initialization_Batch_Normalization(1, 0) 


def Models_init():
	w11_init = open("/Users/jim/Downloads/Python/Project/Models/w11.json",'r')
	w12_init = open("/Users/jim/Downloads/Python/Project/Models/w12.json",'r')
	w13_init = open("/Users/jim/Downloads/Python/Project/Models/w13.json",'r')
	w21_init = open("/Users/jim/Downloads/Python/Project/Models/w21.json",'r')
	w22_init = open("/Users/jim/Downloads/Python/Project/Models/w22.json",'r')
	w3_init = open("/Users/jim/Downloads/Python/Project/Models/w3.json",'r')
	w11 = np.array(w11_init.read())
	w12 = np.array(w12_init.read())
	w13 = np.array(w13_init.read())
	w21 = np.array(w21_init.read())
	w22 = np.array(w22_init.read())
	w3 = np.array(w3_init.read())
	models = [w11,w12,w13,w21,w22,w3]
	return models

def Models_update(w):
	w11_file = open("/Users/jim/Downloads/Python/Project/Models/w11.json",'w')
	w12_file = open("/Users/jim/Downloads/Python/Project/Models/w12.json",'w')
	w13_file = open("/Users/jim/Downloads/Python/Project/Models/w13.json",'w')
	w21_file = open("/Users/jim/Downloads/Python/Project/Models/w21.json",'w')
	w22_file = open("/Users/jim/Downloads/Python/Project/Models/w22.json",'w')
	w3_file = open("/Users/jim/Downloads/Python/Project/Models/w3.json",'w')
	w11 = w[0]
	w12 = w[1]
	w13 = w[2]
	w21 = w[3]
	w22 = w[4]
	w3 = w[5]
	w11_list = w11.tolist()
	w12_list = w12.tolist()
	w13_list = w13.tolist()
	w21_list = w21.tolist()
	w22_list = w22.tolist()
	w3_list = w3.tolist()
	json.dump(w11_list,w11_file)
	json.dump(w12_list,w12_file)
	json.dump(w13_list,w13_file)
	json.dump(w21_list,w21_file)
	json.dump(w22_list,w22_file)
	json.dump(w3_list,w3_file)
	
	
	

def Compute_Loss(i):			#ith image
	w = Models_init()
	w11 = w[0]
	w12 = w[1]
	w13 = w[2]
	w21 = w[3]
	w22 = w[4]
	w3 = w[5]
	
	#first layer
	image_Processing = image[i]
	conv_Outcome_w11 = Conv(image_Processing,w11)
	conv_Outcome_w12 = Conv(image_Processing,w12)
	conv_Outcome_w13 = Conv(image_Processing,w13)
	conv_Outcome_x2 = np.stack([conv_Outcome_w11,conv_Outcome_w12,conv_Outcome_w13],axis=-1)
	x2 = ReLU(conv_Outcome_x2)
	
	#second layer
	x3 = MaxPooling(x2)
	
	#third layer
	conv_Outcome_w21 = Conv(x3, w21)
	conv_Outcome_w22 = Conv(x3, w22)
	conv_Outcome_x4 = np.stack([conv_Outcome_w21,conv_Outcome_w22],axis=-1)
	x4 = ReLU(conv_Outcome_x4)
	
	#fourth layer
	
	x5 = MaxPooling(x4)
	
	#fifth layer
	
	conv_Outcome_x5 = Conv(x5, w3)
	x6 = ReLU(x5)
	
	#seventh layer
	Loss = SVM(x6, w3)
	return Loss




def Compute_Numerical_Gradient_And_Descent(x):	#xth image		#Adam
	beta1 = 0.9
	beta2 = 0.999
	learning_rate = 0.001
	w = Models_init()
	for k in range(6):
		for i in range(3):
			for j in range(3):
				for t in range(50):			#1000 times training
					first_moment = 0
					second_moment = 0
					Loss_first = Compute_Loss(x)
					w[k][i,j] += 10**-7
					Loss_second = Compute_Loss(x)
					dx = (Loss_second - Loss_first)*10**7
					first_moment = beta1*first_moment + (1-beta1)*dx
					second_moment = beta2*second_moment + (1-beta2)*dx*dx
					first_unbias = first_moment/(1-beta1**t)
					second_unbias = second_moment/(1-beta2**t)
					w[k][i,j] -= learning_rate*first_unbias/(np.sqrt(second_unbias+1e-7))
	Models_update(w)
	
	
Compute_Numerical_Gradient_And_Descent(0)
Compute_Loss(0)
	

				