#!/usr/bin/env python
# coding: utf-8

# Building the CNN  model by using keras.....
# 
# 4-Convolution layers
# 
# 2-Dense for nn-hidden and output

# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import Sequential
from keras.preprocessing import image


# In[ ]:


from keras.layers import LeakyReLU


# In[ ]:


# Building CNN model using keras

model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#model.add(Conv2D(128,(3,3),activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))

# now 4 convolution layers are  added


# now adding  flatten layer for 1D input NN
model.add(Flatten())

#model.summary()

#Adding  Dense or hidden layers for  our  NN (Fully connected layers)



model.add(Dense(64,activation='relu'))
model.add(Dropout(0.50))

# adding  output  layer with 1 output  since it is binary classification

model.add(Dense(1,activation='sigmoid'))

#compling model with all layers

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


#To make data ready for keras
# Training data images of x-ray 
# Normalizing data  by rescaling size of imageand adding some augmnentation to train data like shear,zoom etc
# by keeping horizontal_flip = True becoz not invert the x-ray

train_datagen= image.ImageDataGenerator(
               rescale=1./255,
               shear_range=0.2,
               zoom_range=0.2,
               horizontal_flip=True
               )

#just applying rescaling

test_dataset=image.ImageDataGenerator(rescale=1./255)


# In[ ]:


#import os
#import shutil
# to remove extra class indices '.ipynb_checkpoints'
#shutil.rmtree('output/train/.ipynb_checkpoints')
#shutil.rmtree('output/val/.ipynb_checkpoints')


# In[ ]:


# actual application of generators 
# on objects  (here train_datagen)
train_generator = train_datagen.flow_from_directory(
    'output/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
 )



# In[ ]:


train_generator.class_indices


# In[ ]:


# for test dataset 

validation_generator= test_dataset.flow_from_directory(
    'output/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
    
)


# In[ ]:


validation_generator.class_indices


# In[ ]:


hist=model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=2
)


# loss is  decreasing 
# 
# ## Saving the model

# In[ ]:


model.save('cnn0.h5')


# In[ ]:


acc=model.evaluate_generator(train_generator)


# In[ ]:


print ("Train accuracy is :")

acc[1]


# In[ ]:


model.evaluate_generator(validation_generator)


# ## Testing  the images

# In[ ]:


# load and evaluate a saved model
from keras.models import load_model


model= load_model('cnn0.h5')


# In[ ]:


train_generator.class_indices


# In[ ]:


y_actual= []
y_test = []


# In[ ]:


for i in os.listdir("./output/val/Normal/"):
    img= image.load_img("./output/val/Normal/"+i,target_size=(224,224))
    img= image.img_to_array(img)
    img= np.expand_dims(img,axis=0)
    p= model.predict_classes(img)
    y_test.append(p[0,0])
    y_actual.append(1)
    


# In[ ]:


for i in os.listdir("./output/val/Covid/"):
    img= image.load_img("./output/val/Covid/"+i,target_size=(224,224))
    img= image.img_to_array(img)
    img= np.expand_dims(img,axis=0)
    p= model.predict_classes(img)
    y_test.append(p[0,0])
    y_actual.append(0)


# In[ ]:


y_actual=np.array(y_actual)
y_test= np.array(y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix

cm= confusion_matrix(y_actual,y_test)


# In[ ]:


import seaborn as sns

sns.heatmap(cm,cmap='plasma',annot=True)


# train loss ,acc[0.2863135039806366, 0.9783549904823303] 
# test loss ,acc  [0.13886258006095886, 0.982758641242981]
# 
# 
# 
# cm [[28, 1],
#     [0,29]]
# # Building CNN model using keras
# 
# model=Sequential()
# model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
# 
# model.add(Conv2D(64,(2,2),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# 
# 
# model.add(Conv2D(32,(3,3),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# 
# model.add(Flatten())
# 
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.50))
# 
# # adding  output  layer with 1 output  since it is binary classification
# 
# model.add(Dense(1,activation='sigmoid'))
# 
# #compling model with all layers
# 
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# 
# 
# 
# 
# 
# 

# In[ ]:




