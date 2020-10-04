#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import cv2
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from torch.autograd import Variable 
from sklearn import preprocessing

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ### Image Preprocessing

# In[2]:


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


# In[3]:


df_train.head()


# In[4]:


df_test.head()


# In[5]:


img = cv2.imread("images/11893820-3x2-xlarge.jpg")
b,g,r = cv2.split(img)  
img2 = cv2.merge([r,g,b]) 
plt.imshow(img2)


# In[6]:


def load_images_from_folder(folder):
    images = []
    images_name = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            images_name.append(filename)
    return images, images_name


# In[7]:


images, images_name = load_images_from_folder("images")


# In[8]:


def resize_img(images):
    new_images = []
    for i in range(len(images)):
        resized_images = images[i]
        resized_images = cv2.resize(resized_images, (256,256))
        new_images.append(resized_images)
    return new_images


# In[9]:


new_images = resize_img(images)


# In[10]:


images_list_with_name = [(i,j) for i,j in zip(new_images, images_name)]


# ### Prepare Training Data

# In[11]:


def get_img_train_label(images_list_with_name):
    
    final_label_result = []
    
    for i in range(len(images_list_with_name)):
        """
        find how many labels to crop for each image
        """
        num_to_crop = df_train.loc[df_train['filename']==images_list_with_name[i][1]]
        
        for j in range(len(num_to_crop)):
            good_bad_none_label = '0'
            
            img = cv2.imread(os.path.join("images",images_list_with_name[i][1]))
            #img = images_list_with_name[i][0]

            x_min = df_train.loc[df_train['filename']==images_list_with_name[i][1]]['xmin'].iloc[j]
            x_max = df_train.loc[df_train['filename']==images_list_with_name[i][1]]['xmax'].iloc[j]
            y_min = df_train.loc[df_train['filename']==images_list_with_name[i][1]]['ymin'].iloc[j]
            y_max = df_train.loc[df_train['filename']==images_list_with_name[i][1]]['ymax'].iloc[j]
            
            # plt.imshow(img)
            cropped_img = img[y_min:y_max, x_min:x_max]
            cropped_img_resized = cv2.resize(cropped_img, (256,256))
            
            
            # check good or bad
            if df_train.loc[df_train['filename']==images_list_with_name[i][1]]['label'].iloc[j]=='bad':
                good_bad_none_label = 'bad'
            if df_train.loc[df_train['filename']==images_list_with_name[i][1]]['label'].iloc[j]=='good':
                good_bad_none_label = 'good'
            if df_train.loc[df_train['filename']==images_list_with_name[i][1]]['label'].iloc[j]=='none':
                good_bad_none_label = 'none'
            
            final_label_result.append([cropped_img_resized, good_bad_none_label])
        
        if (i % 100 ==0):
            print("No. of pic: {0}".format(i)) # progress
    
    return final_label_result


# In[12]:


final_label_result = get_img_train_label(images_list_with_name)


# In[13]:


len(final_label_result)


# In[14]:


# tmp = []
# for i in range(len(final_label_result)):
#     if final_label_result[i][1]=='none':
#         tmp.append([final_label_result[i][0], final_label_result[i][1]])
#         final_label_result.append([final_label_result[i][0], final_label_result[i][1]])

# print(len(tmp))


# In[15]:


# save the training label as npz
final_label_result_array = np.array(final_label_result)
np.savez("final_label_result", final_label_result_array)


# In[16]:


plt.imshow(final_label_result[0][0])
print(final_label_result[0][1])


# In[17]:


images_train = []
labels_train = []

for idx, (x,y) in enumerate(final_label_result_array):
    images_train.append(x)
    labels_train.append(y)

images_train = np.array(images_train).reshape(3520,3,256,256)
labels_train = np.array(labels_train)


# In[18]:


images_train.shape


# In[19]:


# numpy to torch tensor
x_train = torch.from_numpy(images_train)
le = preprocessing.LabelEncoder()
labels_train = le.fit_transform(labels_train)
y_train = torch.from_numpy(labels_train)


# In[20]:


x_train.shape


# In[21]:


train_dataset = Data.TensorDataset(x_train, y_train)


# In[22]:


train_loader = Data.DataLoader(
    dataset = train_dataset,      # torch TensorDataset format
    batch_size = 128,             # mini batch size
    shuffle = True,               # 要不要打乱数据 (打乱比较好)
    #num_workers=4,                # 多线程来读数据
)


# ### Prepare Test Data

# In[23]:


def get_img_test_label(images_list_with_name):
    
    final_label_result = []
    
    for i in range(len(images_list_with_name)):
        """
        find how many labels to crop for each image
        """
        num_to_crop = df_test.loc[df_test['filename']==images_list_with_name[i][1]]
        
        for j in range(len(num_to_crop)):
            good_bad_none_label = '0'
            
            img = cv2.imread(os.path.join("images",images_list_with_name[i][1]))
            #img = images_list_with_name[i][0]

            x_min = df_test.loc[df_test['filename']==images_list_with_name[i][1]]['xmin'].iloc[j]
            x_max = df_test.loc[df_test['filename']==images_list_with_name[i][1]]['xmax'].iloc[j]
            y_min = df_test.loc[df_test['filename']==images_list_with_name[i][1]]['ymin'].iloc[j]
            y_max = df_test.loc[df_test['filename']==images_list_with_name[i][1]]['ymax'].iloc[j]
            
            # plt.imshow(img)
            cropped_img = img[y_min:y_max, x_min:x_max]
            cropped_img_resized = cv2.resize(cropped_img, (256,256))
            
            
            # check good or bad
            if df_test.loc[df_test['filename']==images_list_with_name[i][1]]['label'].iloc[j]=='bad':
                good_bad_none_label = 'bad'
            if df_test.loc[df_test['filename']==images_list_with_name[i][1]]['label'].iloc[j]=='good':
                good_bad_none_label = 'good'
            if df_test.loc[df_test['filename']==images_list_with_name[i][1]]['label'].iloc[j]=='none':
                good_bad_none_label = 'none'
            
            final_label_result.append([cropped_img_resized, good_bad_none_label])
        
        if (i % 100 ==0):
            print("No. of pic: {0}".format(i)) # progress
    
    return final_label_result


# In[24]:


final_label_result_test = get_img_test_label(images_list_with_name)


# In[25]:


len(final_label_result_test)


# In[26]:


# tmp = []
# for i in range(len(final_label_result_test)):
#     if final_label_result_test[i][1]=='none':
#         tmp.append([final_label_result_test[i][0], final_label_result_test[i][1]])
#         final_label_result_test.append([final_label_result_test[i][0], final_label_result_test[i][1]])

# print(len(tmp))


# In[27]:


# save the test label as npz
final_label_result_array_test = np.array(final_label_result_test)
np.savez("final_label_result_test", final_label_result_array_test)


# In[28]:


plt.imshow(final_label_result_test[0][0])
print(final_label_result_test[0][1])


# In[29]:


images_test = []
labels_test = []

for idx, (x,y) in enumerate(final_label_result_array_test):
    images_test.append(x)
    labels_test.append(y)

images_test = np.array(images_test).reshape(394,3,256,256)
labels_test = np.array(labels_test)


# In[30]:


images_test.shape


# In[31]:


# numpy to torch tensor
x_test = torch.from_numpy(images_test)
le = preprocessing.LabelEncoder()
labels_test = le.fit_transform(labels_test)
y_test = torch.from_numpy(labels_test)


# In[32]:


test_dataset = Data.TensorDataset(x_test, y_test)


# In[33]:


test_loader = Data.DataLoader(
    dataset = test_dataset,      
    batch_size = 128, # mini-batch size
    shuffle = True,     
    #num_workers=2, # multithread
)


# ### CNN Model

# In[34]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # input shape:(3, 256, 256)
            nn.Conv2d(in_channels=3, 
                      out_channels=32, 
                      kernel_size=5, 
                      stride=1, # if stride = 1, padding = (kernel_size-1)/2
                      padding=2), # (32, 256, 256)
            nn.ReLU(), # (32, 256, 256)
            nn.MaxPool2d(kernel_size=2) # output shape: (32, 128, 128)
        )
        self.conv2 = nn.Sequential( # input shape:(32, 128, 128)
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2), # (64, 128, 128)
            nn.ReLU(), # (64, 128, 128)
            nn.MaxPool2d(kernel_size=2) # output shape: (64, 64, 64)
        )
        self.out = nn.Linear(in_features=64 * 64 * 64, out_features=3)
    
    def forward(self, x):
        x = self.conv1(x.float())
        x = self.conv2(x.float()) # (batch, 64, 64, 64)
        # flatten
        x = x.view(x.size()[0], -1) # (batch size, 64*64*64)
        output = self.out(x)
        return output

cnn = CNN()
print(cnn)


# In[35]:


optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()


# In[38]:


class history_package():
    def __init__(self, net, train_loader, test_loader, EPOCH=30, LR=0.0001):
        self.net = net
        self.optimizer = torch.optim.Adam(net.parameters(), lr = LR)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.EPOCH_ = EPOCH
        self.LR_ = LR

        self.net = self.net.to(device)
        if device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            torch.backends.cudnn.benchmark = True
        
        self.good = 0
        self.bad = 0
        self.none = 0
        self.rgood = 0
        self.rbad = 0
        self.rnone = 0
        
    def start(self):

        history_loss = []
        history_train_acc = []
        history_test_acc = []
        
        for epoch in range(self.EPOCH_):
            print('Epoch:', epoch)
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.test()

            history_loss.append(train_loss)
            history_train_acc.append(train_acc)
            history_test_acc.append(test_acc)
        return history_loss, history_train_acc, history_test_acc

    def train(self):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        self.good = 0
        self.bad = 0
        self.none = 0
        self.rgood = 0
        self.rbad = 0
        self.rnone = 0
        
        for step, (batch_X, batch_y) in enumerate(self.train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            self.optimizer.zero_grad()
            outputs = self.net(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            
            self.good += list(predicted).count(1)
            self.bad += list(predicted).count(0)
            self.none += list(predicted).count(2)

            self.rgood += list(batch_y).count(1)
            self.rbad += list(batch_y).count(0)
            self.rnone += list(batch_y).count(2)

                
                
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()        

        print('【Training】Loss: %.3f | Acc: %.3f%% (%d/%d)' % ( train_loss, 100.*(correct/total), correct, total ))
        print(self.good,self.bad,self.none)
        print(self.rgood, self.rbad, self.rnone)
 
        return train_loss, (correct/total)

    def test(self):
        self.net.eval()

        test_loss = 0
        correct = 0
        total = 0
        
        self.good = 0
        self.bad = 0
        self.none = 0
        self.rgood = 0
        self.rbad = 0
        self.rnone = 0
        
        with torch.no_grad(): 
            for step, (batch_X, batch_y) in enumerate(self.test_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = self.net(batch_X)
                loss = self.criterion(outputs, batch_y)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                
                self.good += list(predicted).count(1)
                self.bad += list(predicted).count(0)
                self.none += list(predicted).count(2)
                
                self.rgood += list(batch_y).count(1)
                self.rbad += list(batch_y).count(0)
                self.rnone += list(batch_y).count(2)

                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()   

        print('【Testing】Loss: %.3f | Acc: %.3f%% (%d/%d)' % ( test_loss, 100.*(correct/total), correct, total ))
        print(self.good,self.bad,self.none)
        print(self.rgood, self.rbad, self.rnone)
 
        return test_loss, (correct/total)


# In[39]:


cnn1_module = history_package(CNN(), train_loader, test_loader)
history_loss, history_train_acc, history_test_acc = cnn1_module.start()


# In[40]:


fig, ax = plt.subplots(1, 2)
fig.set_size_inches(16, 8)
ax[0].set_title('Learning Cruve')
ax[0].set_xlabel('Number of Epochs')
ax[0].set_ylabel('Cross Entropy')
ax[0].plot(history_loss)

ax[1].set_title('Accuracy')
ax[1].set_xlabel('Number of Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].plot(history_train_acc, label = 'train')
ax[1].plot(history_test_acc, label = 'test')
plt.legend(loc=1)

plt.savefig("CNN_LearningCurve_Accuracy")


# ### Result

# In[41]:


plt.imshow(x_test[0].reshape(256,256,3))


# In[44]:


label_name_list = ['1','0','2']#good, bad, none

def raw_plot():
    fig = plt.gcf()
    fig.set_size_inches((15,20))

    with torch.no_grad(): 
        for idx in range(25):
            temp = torch.from_numpy(np.expand_dims(x_test[idx], axis=0)).to(device)
            outputs = cnn1_module.net(temp)
            _, predicted = outputs.max(1)

            ax = plt.subplot(5, 5, idx+1)
            ax.imshow(x_test[idx].reshape(256,256,3))
            
            label = 0
            if label_name_list[y_test[idx]]=='0':
                label = 'good'
            if label_name_list[y_test[idx]]=='2':
                label = 'bad'
            if label_name_list[y_test[idx]]=='1':
                label = 'none'
            
            predicted_result = 0
            if label_name_list[predicted]=='0':
                predicted_result = 'good'
            if label_name_list[predicted]=='2':
                predicted_result = 'bad'
            if label_name_list[predicted]=='1':
                predicted_result = 'none'
            
                
            title = 'label:' + label + ', predict:' + predicted_result
            if(y_test[idx].item() == predicted.item()):
                ax.set_title(title, fontsize=10)
            else:
                ax.set_title(title, fontsize=10, color='red')


# In[45]:


raw_plot()


# In[ ]:




