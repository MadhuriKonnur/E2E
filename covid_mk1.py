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
import tensorflow.keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend
#import os
#import shutil


# In[ ]:





# Building CNN model using keras

model=Sequential()

# now adding convolution layers are  added
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#model.add(Conv2D(64,(3,3),activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))




model.add(Flatten())
# now added  flatten layer for 1D input NN

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
    steps_per_epoch=8,
    epochs=8,
    validation_data=validation_generator,
    validation_steps=8
)


# loss is  decreasing 
# 
# ## Evaluating model accuracy

# In[ ]:
#Saving  model
model.save('mk_cnn.h5')


acc=model.evaluate(train_generator)


# In[ ]:


ta=acc[1]


# In[ ]:


with open('accuracy.txt', 'w') as myFile:
    print( ta, file=myFile)


# In[ ]:


model.evaluate(validation_generator)


# # Testing model 
# #Once we got  desired  accuracy 
# #test the  model.predict( test_image)
# #for  chest x-ray
# 
# #<certain level preprocessing of  test_image  is required  before predicting>
# 
# 

# # For further  use , we can directly  store  weights  by  saving  the  model
# 
# #load and evaluate a saved model
# #from keras.models import load_model
# 
# 
# #model= load_model('cnnmk.h5')

# In[ ]:




