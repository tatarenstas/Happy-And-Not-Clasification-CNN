import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('Dataset/Train',target_size=(200,200),batch_size=3,class_mode='binary')
validation_dataset = validation.flow_from_directory('Dataset/Validation',target_size=(200,200),batch_size=3,class_mode='binary')
train_dataset.class_indices

model = tf.keras.Sequential([
                             Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
                             MaxPool2D(2,2),
                             Conv2D(32,(3,3),activation='relu'),
                             MaxPool2D(2,2),
                             Conv2D(64,(3,3),activation='relu'),
                             MaxPool2D(2,2),
                             Flatten(),
                             Dense(512,activation='relu'),
                             Dense(1,activation='sigmoid')
])
model.compile(loss = 'binary_crossentropy',
              optimizer = RMSprop(lr=0.001),
              metrics = ['accuracy'])
model_fit = model.fit(train_dataset,steps_per_epoch=5,epochs=30,validation_data=validation_dataset)

dir_path = 'Dataset/Test'

for i in os.listdir(dir_path):
  img = image.load_img(dir_path+'//'+i,target_size=(200,200))
  x = image.img_to_array(img)
  x = np.expand_dims(x,axis=0)
  images = np.vstack([x])
  val = model.predict(images)
  if val == 1:
    print('happy')
  else:
    print('not happy')
  plt.imshow(img)
