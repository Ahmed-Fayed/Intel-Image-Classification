# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 16:24:56 2021

@author: ahmed
"""


# Import Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import random
from tqdm import tqdm

import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical


# Reading Training Classes
train_classes_path = "E:/Software/professional practice projects/In progress/seg_train/seg_train"
classes_list = os.listdir(train_classes_path)

# Exploring Random Images from Random CLasses
plt.figure(figsize=(15, 10))
plt.tight_layout()
counter = 0
for class_name in classes_list:
    class_path = os.path.join(train_classes_path, class_name)
    
    random_img_name = random.choice(os.listdir(class_path))
    img_path = os.path.join(class_path, random_img_name)
    img = cv2.imread(img_path)
    
    counter += 1
    # We only have 6 class so plotting img for each class
    plt.subplot(2, 3, counter)
   
    plt.imshow(img)
    plt.xlabel(img.shape[1])
    plt.ylabel(img.shape[0])
    plt.title(class_name)

plt.show()
# Create a autograph pre-processing function to resize and normalize an image

img_width = 150
img_height = 150
def format_image(img_path):
    img = cv2.imread(img_path)
    
    img = tf.image.resize(img, (img_width, img_height))
    img /= 255.0
    
    return img
    
# Function to Load datsets

def load_dataset(dataset_path):
    images = []
    labels = []
    label = 0
    for class_name in classes_list:
        class_path = os.path.join(dataset_path, class_name)
        print('{} class'.format(class_name))
        
        for img_name in tqdm(os.listdir(class_path)):
            img_path = os.path.join(class_path, img_name)
            img = format_image(img_path)
            
            images.append(img)
            labels.append(label)
        
        label += 1
        
    return images, labels



# Loading data
print('Loading Training data..')
train_dataset, train_labels = load_dataset(train_classes_path)

print('loading testing data..')
test_classes_path = "E:/Software/professional practice projects/In progress/seg_test/seg_test"
test_dataset, test_labels = load_dataset(test_classes_path)


# Splitting data into training and validation
train_dataset = np.array(train_dataset)
train_labels = np.array(train_labels)
x_train, x_val, y_train, y_val = train_test_split(train_dataset, train_labels, test_size=0.2, shuffle=True)


y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

num_classes = 6

# Building a functional model
inputs = Input(shape=(img_width, img_height, 3))
x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)
x = Dropout(0.2)(x)

x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

x = Flatten()(x)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

classifier = Dense(num_classes, activation='softmax')(x)


model = Model(inputs=inputs, outputs=classifier)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=30, validation_data=(x_val, y_val),
          callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True), 
                     CSVLogger('train.csv')])



model.save('intel_image_classifier.h5')
model.save_weights('intel_image_weights.h5')

json_model = model.to_json()
with open("E:/Software/professional practice projects/In progress/intel_image_model.json", 'w') as json_file:
    json_file.write(json_model)



def plot_accuray(history):
    plt.figure(figsize=(12, 8))
    plt.title("Intel Image Model Accuracy")
    plt.plot(history.history['accuracy'], color='g')
    plt.plot(history.history['val_accuracy'], color='b')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(['train', 'val'], loc="lower right")
    plt.show()


def plot_loss(history):
    plt.figure(figsize=(12, 8))
    plt.title("Intel Image Model Loss")
    plt.plot(history.history['loss'], color='g')
    plt.plot(history.history['val_loss'], color='b')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(['train', 'val'], loc="lower right")
    plt.show()



history = model.history

plot_accuray(history)
plot_loss(history)


# measure accuracy and loss in test dataset
x_test = np.array(test_dataset)
y_test = np.array(test_labels)
y_test = to_categorical(y_test)

loss, accuracy = model.evaluate(x_test, y_test)


y_test = np.array(test_labels)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)


# y_test type is array
# y_pred type is array
CM = confusion_matrix(y_test, y_pred)
ax = plt.axes()
sns.heatmap(CM, annot=True, 
           annot_kws={"size": 10}, 
           xticklabels=classes_list, 
           yticklabels=classes_list, ax = ax)
ax.set_title('Confusion matrix')
plt.show()

