#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import keras
import random
import math

from sklearn import manifold
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# In[2]:


train_img_rows, train_img_cols = 28,28
train_data = np.load("train.npz")
train_img_data, train_label_data = train_data['image'], train_data['label']
print(train_img_data.shape)
print(train_label_data.shape)


# In[3]:


plt.imshow(train_img_data[0])


# In[4]:


# find the value to do normalization
print(train_img_data.min(), train_img_data.max())
print(train_label_data.min(), train_label_data.max())

x_train = train_img_data.reshape(12000,784)/255
y_train = keras.utils.to_categorical(train_label_data,10) # one-hot encoding
print(x_train.shape)
print(y_train.shape)

# after normalization
print(x_train.min(), x_train.max())
print(y_train.min(), y_train.max())


# In[5]:


test_img_rows, test_img_cols = 28,28
test_data = np.load("test.npz")
test_img_data, test_label_data = test_data['image'], test_data['label']
print(test_img_data.shape)
print(test_label_data.shape)


# In[6]:


plt.imshow(test_img_data[0])


# In[7]:


# find the value to do normalization
print(test_img_data.min(), test_img_data.max())
print(test_label_data.min(), test_label_data.max())

# reshape and normalize
x_test = test_img_data.reshape(5768,784)/255
y_test = keras.utils.to_categorical(test_label_data,10) # one-hot encoding
print(x_train.shape)
print(y_train.shape)

# after normalization
print(x_test.min(), x_test.max())
print(y_test.min(), y_test.max())


# In[8]:


def sigmoid(s):
    return 1/(1 + np.exp(-s))

def sigmoid_derv(s):
    return s * (1 - s)

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def error(pred, real):
    n_samples = real.shape[0]
    
    #print(pred.shape) # pred.shape = (12000, 10)
    #print(pred[np.arange(n_samples), real.argmax(axis=1)]) 
    
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss


# In[9]:


class DNN:
    def __init__(self, x_train, y_train):
        self.x = x_train
        neurons = 128
        self.lr = 0.5
        
        ip_dim = x_train.shape[1]
        op_dim = y_train.shape[1]
        
        # random initializations
        self.w1 = np.random.randn(ip_dim, neurons)
#         self.w1 = np.zeros((ip_dim, neurons))
        self.b1 = np.zeros((1, neurons))
        
        self.w2 = np.random.randn(neurons, neurons)
#         self.w2 = np.zeros((neurons, neurons))
        self.b2 = np.zeros((1, neurons))
        
        self.w3 = np.random.randn(neurons, neurons)
#         self.w3 = np.zeros((neurons, 2))
        self.b3 = np.zeros((1, neurons))
        
        self.w4 = np.zeros((neurons, op_dim))
#         self.w4 = np.random.randn(2, op_dim) # 2 nodes before the output layer

        self.b4 = np.zeros((1, op_dim))
        
        self.y = y_train
        
        self.training_loss = []
        self.training_error = []
        self.latent_features = []
        
        
    def feedforward(self, x, latent_flag):
        self.x = x

        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = sigmoid(z3)
        
        z4 = np.dot(self.a3, self.w4) + self.b4
        self.a4 = softmax(z4)
        
        if latent_flag == 1:
            self.latent_features.append(z4) # latent features 
        
    def backprop(self, pos):
        loss = error(self.a4, self.y)
        
        self.training_loss.append(loss)
        if (pos % 100==0):
            print('Loss :', loss)
        
        
        a4_delta = cross_entropy(self.a4, self.y) # w4
        z3_delta = np.dot(a4_delta, self.w4.T)
        a3_delta = z3_delta * sigmoid_derv(self.a3) # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.a2) # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derv(self.a1) # w1

        self.w4 -= self.lr * np.dot(self.a3.T, a4_delta)
        self.b4 -= self.lr * np.sum(a4_delta, axis=0, keepdims=True)
        
        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0)
        
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

    def predict(self, data):
        self.x = data
        self.feedforward(self.x, 0)
        return self.a4.argmax()
    
    def predict_test(self, data):
        self.x = data
        self.feedforward(self.x, 0)
        return self.a4.argmax(axis=1)


# In[10]:


def get_acc(x, y):
    acc = 0
    for xx,yy in zip(x, y):
        s = model.predict(xx)
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100


# In[11]:


model = DNN(x_train, y_train)

epochs = 3000
batch_size = 100
training_error_rate = []
latent_features = []
testing_error_rate = []

for pos in range(epochs):
    #for i in range(0,len(x_train),batch_size):
    if (pos == 200 or pos == 800):
        latent_flag = 1
    else:
        latent_flag = 0
    
    model.feedforward(x_train, latent_flag)
    model.backprop(pos)     
    
    error_counter_train = 0
    error_counter_test = 0

    predicted_array_train = model.a4.argmax(axis=1)
    predicted_array_test = model.predict_test(x_test)
    
    true_array = model.y.argmax(axis=1)
    true_array_test = y_test.argmax(axis=1)

    for i,j in zip(predicted_array_train, true_array):
        if i!=j:
            #print(i,j)
            error_counter_train += 1
    
    for v,w in zip(predicted_array_test, true_array_test):
        if v!=w:
            error_counter_test += 1
    
    # record for training error rate and testing error rate
    training_error_rate.append(error_counter_train/model.y.shape[0])
    testing_error_rate.append(error_counter_test/y_test.shape[0])


    latent_features.append(model.latent_features)
    
    if (pos % 100 == 0):
        print("Epoch: {0}".format(pos))
        print("Training Error Rate: {0}".format(error_counter_train/model.y.shape[0]))
        print("Testing Error Rate: {0}".format(error_counter_test/y_test.shape[0]))
        print("===============================================")

print("Training accuracy : ", get_acc(x_train, y_train))
print("Test accuracy : ", get_acc(x_test, y_test))


# In[12]:


arr_model_loss = np.array(model.training_loss)[100:]
plt.figure(figsize=(4,4))
plt.xlabel("Number of epochs")
plt.ylabel("Average Cross Entropy")
plt.title("Training Loss")
plt.plot(arr_model_loss)
plt.savefig("training_loss")


# In[13]:


arr_model_training_error_rate = np.array(training_error_rate)[100:]
plt.figure(figsize=(4,4))
plt.xlabel("Number of epochs")
plt.ylabel("Error Rate")
plt.title("Training Error Rate")
plt.plot(arr_model_training_error_rate)
plt.savefig("training_error_rate")


# ### Testing

# In[14]:


arr_model_testing_error_rate = np.array(testing_error_rate)[100:]
plt.figure(figsize=(4,4))
plt.xlabel("Number of epochs")
plt.ylabel("Error Rate")
plt.title("Testing Error Rate")
plt.plot(arr_model_testing_error_rate)
plt.savefig("testing_error_rate")


# In[15]:


from sklearn.metrics import confusion_matrix
confusion_matrix(true_array_test, predicted_array_test)


# ### Latent Features

# In[156]:


# plt.scatter(latent_features[0][:,0],latent_features[0][:,1],c=train_label_data)
# plt.title("2D Feature 20th epoch")
# plt.savefig("latent_20th")


# In[157]:


# plt.scatter(latent_features[1][:,0],latent_features[1][:,1],c=train_label_data)
# plt.title("2D Feature 80th epoch")
# plt.savefig("latent_80th")

