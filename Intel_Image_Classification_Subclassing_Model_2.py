# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 15:33:37 2021

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
from tensorflow.keras.preprocessing import image_dataset_from_directory


# Reading Training Classes
train_classes_path = "E:/Software/professional practice projects/Done/seg_train/seg_train"
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


img_width = 150
img_height = 150
batch_size = 64


# Load Dataset
train_dataset = image_dataset_from_directory(
    train_classes_path,
    label_mode='categorical',
    class_names=classes_list,
    batch_size=batch_size,
    image_size=(img_width, img_height),
    seed=123,
    validation_split=0.2, 
    subset="training")

val_dataset = image_dataset_from_directory(
    train_classes_path, 
    label_mode='categorical',
    class_names=classes_list,
    batch_size=batch_size,
    image_size=(img_width, img_height),
    seed=123,
    validation_split=0.2,
    subset="validation")


# Visualizing images in the training dataset
plt.figure(figsize=(10, 6))
for images, labels in train_dataset.take(1):
    for i in range(1, 7):
        plt.subplot(2, 3, i)
        plt.tight_layout()
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(classes_list[np.argmax(labels[i])])


# Visualizaing images in the val dataset
plt.figure(figsize=(8, 6))
for images, labels in val_dataset.take(1):
    for i in range(1, 7):
        plt.subplot(2, 3, i)
        plt.tight_layout()
        plt.imshow(images[i].numpy().astype("uint"))
        plt.title(classes_list[np.argmax(labels[i])])


# exploring batches shape in training dataset
for x, y in train_dataset.take(1):
    print("batch images {}".format(x.shape))
    print("batch labels {}".format(y.shape))


# exploring batches shape in val dataset
for x, y in val_dataset.take(1):
    print("batch images {}".format(x.shape))
    print("batch labels {}".format(y.shape))



# Standarize the images to be for 0 to 1
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

image_batch, labels_batch = next(iter(train_dataset))

first_image = image_batch[0]
first_label = labels_batch[0]

print(np.min(first_image), ' --> ', np.max(first_image))
print(first_label)



image_batch_val, labels_batch_val = next(iter(val_dataset))

first_image_val = image_batch_val[0]
first_label_val = labels_batch_val[0]

print(np.min(first_image_val), ' --> ', np.max(first_image_val))
print(first_label_val)




# Configure the dataset for performance
AutoTune = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AutoTune)
val_dataset = val_dataset.cache().prefetch(buffer_size=AutoTune)


num_classes = 6


class IntelImageModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(IntelImageModel, self).__init__()
        
        self.conv1 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(img_width, img_height, 3))
        self.conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv3 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv4 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')
        
        self.btn1 = BatchNormalization()
        self.btn2 = BatchNormalization()
        self.btn3 = BatchNormalization()
        self.btn4 = BatchNormalization()
        self.btn5 = BatchNormalization()
        self.btn6 = BatchNormalization()
        
        self.mxpool = MaxPooling2D(pool_size=(2, 2))
        self.drop = Dropout(0.25)
        
        self.flat = Flatten()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.classifier = Dense(num_classes, activation='softmax')
    
    
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.btn1(x)
        x = self.mxpool(x)
        x = self.drop(x)
        
        x = self.conv2(x)
        x = self.btn2(x)
        x = self.mxpool(x)
        x = self.drop(x)
        
        x = self.conv3(x)
        x = self.btn3(x)
        x = self.mxpool(x)
        x = self.drop(x)
        
        x = self.conv4(x)
        x = self.btn4(x)
        x = self.mxpool(x)
        x = self.drop(x)
        
        x = self.flat(x)
        
        x = self.dense1(x)
        x = self.btn5(x)
        x = self.drop(x)
        
        x = self.dense2(x)
        x = self.btn6(x)
        x = self.drop(x)
        
        return self.classifier(x)


model = IntelImageModel(num_classes)


patience = 0

def DetectOverFittingCallback(threshold, train_accuracy, val_accuracy):
    ratio = train_accuracy / val_accuracy
    print("  ratio = ", ratio)
    
    global patience
    stop_training = False
    
    if ratio > threshold and patience < 3:
        patience += 1
    elif ratio < threshold:
        patience = 0
    elif ratio > threshold and patience == 3:
        stop_training = True
    
    return stop_training



# Instantiating object from Adam Class
optimizer = tf.keras.optimizers.Adam()


# Intantating objects from categorical Loss Class
train_loss = tf.keras.losses.CategoricalCrossentropy()
val_loss = tf.keras.losses.CategoricalCrossentropy()



# Intantating objects from categorical Accuracy Class
train_accuracy = tf.keras.metrics.CategoricalAccuracy()
val_accuracy = tf.keras.metrics.CategoricalAccuracy()


# this code uses GPU if available otherwise uses a CPU
device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
Epochs = 9



# Custome Training Loop
def train_one_step(model, optimizer, x, y, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        # check for debuging
        y_true = y
        # y = np.expand_dims(y, axis=0)
        loss = train_loss(y, y_pred)
    
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    train_accuracy(y, y_pred)
    
    return loss



# @tf.function
def train(model, optimizer, epochs, device, train_dataset, val_dataset, train_loss, val_loss, train_accuracy, val_accuracy, DetectOverFittingCallback):
    loss = 0.0
    
    # step = 0
    for epoch in tqdm(range(epochs)):
        for (x, y) in tqdm(train_dataset):
            # x = np.expand_dims(x, axis=0)
            # x = np.array(x)
            # y = np.array(y)
            
            # step += 1
            with tf.device(device_name=device):
                loss = train_one_step(model,optimizer, x, y, train_loss, train_accuracy)
                # tf.print()
                # tf.print("step: {},  train loss:  {},  train accuracy:  {}".format(step, loss, train_accuracy.result()))
        
        with tf.device(device_name=device):
            for (x, y) in val_dataset:
                # x = np.expand_dims(x, axis=0)
                # x = np.array(x)
                # y = np.array(y)
                
                y_pred = model(x)
                v_loss = val_loss(y, y_pred)
                val_accuracy(y, y_pred)
        
        tf.print('epoch {}: train loss {} ; val_loss {} ; train accuracy {} ; val_accuracy {}'.format(epoch, loss, v_loss, train_accuracy.result(), val_accuracy.result()))
        
        stop_training = DetectOverFittingCallback(1.3, train_accuracy.result(), val_accuracy.result())
        
        if stop_training:
            print("stop training overfitting..")
            break



train(model, optimizer, Epochs, device, train_dataset, val_dataset, train_loss, val_loss, train_accuracy, val_accuracy, DetectOverFittingCallback)











