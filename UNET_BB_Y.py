# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 05:55:26 2020

@author: emrev
"""
import keras
import tensorflow as tf
from keras.optimizers import Adam

#import optimizer
import os
import random
import numpy as np
import cv2
from tqdm import tqdm 
import tensorflow.keras.backend as K
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

import csv


print('Keras        :', keras.__version__)
print('Tensorflow   :', tf.__version__)


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

IM_HEIGHT = 708
IM_WIDTH = 990 

IMG_NEW_SIZE = 256
IMG_CHANNELS = 3

def read_rectangles(fname):
    positions = []
    with open(fname, mode='r') as csv_file:        
        for row in csv.reader(csv_file, delimiter = ' '):
            row = row[4:8]        
            row = [int(x) for x in row]                        
            positions.append(row)
    return(positions)    


def count_rectangles(fname):
    l = 1    
    with open(fname, mode='r') as csv_file:        
        for row in csv.reader(csv_file, delimiter = ' '):
            l = l + 1
    return(l)    

def resize_position(xy):
    x1,y1,x2,y2 = xy
    x1_r = x1/(IM_WIDTH/IMG_NEW_SIZE)
    y1_r = y1/(IM_HEIGHT/IMG_NEW_SIZE)
    x2_r = x2/(IM_WIDTH/IMG_NEW_SIZE)    
    y2_r = y2/(IM_HEIGHT/IMG_NEW_SIZE)
    
    xy_r = [x1_r, y1_r, x2_r, y2_r]
    
    xy_r = [round(x) for x in xy_r]
    
    return(xy_r)
     

def create_bbox_mask(positions,IMG_NEW_SIZE):

    mask = np.zeros((IMG_NEW_SIZE,IMG_NEW_SIZE,1), dtype=np.bool)
    for row in positions:
        mask[row[1]:row[3] + 1,row[0]:row[2] + 1] = 1
    return(mask)


def show_bbox(img,pos):
    for p in pos:
        cv2.rectangle(img,(p[0],p[1]),(p[2],p[3]),(0,255,0),2)
    return img


def check_plot_bb():
    images = os.listdir(IMAGE_DIR)
    bbs = [x.split('.')[0] + ".txt" for x in images] 
    cr = 0
    for i in range(len(images)):
        bb = BB_DIR +  bbs[i]
        cr = cr + count_rectangles(bb)    
        ss = read_rectangles(bb)
        image = cv2.imread(IMAGE_DIR  + images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape
        if not height == IM_HEIGHT:
            print(height)
        if not width == IM_WIDTH:
            print(width)
  
        #image = resize(image, (IMG_NEW_SIZE, IMG_NEW_SIZE), mode='constant', preserve_range=False)
        bb = BB_DIR +  bbs[i]
        ss = read_rectangles(bb)
        cr = cr + count_rectangles(bb)
        ss2 =  [resize_position(x) for x in ss]
        img = show_bbox(image,ss)
        
        #img = show_bbox(image,ss2)
        
        
        arr = np.asarray(img)
        plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
        plt.show()
        mask = create_bbox_mask(ss2,IMG_NEW_SIZE)            
        input("Press Enter to continue...")
        arr = np.asarray(mask)
        plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
        plt.show()
        print(bb)
        input("Press Enter to continue...")


        
seed = 42
np.random.seed = seed

INPUT_PATH = 'D:\\Dewpoint\\Input\\BB\\'

IMAGE_DIR = INPUT_PATH + 'images\\'
BB_DIR = INPUT_PATH + 'positions\\'

images = os.listdir(IMAGE_DIR)
bbs = [x.split('.')[0] + ".txt" for x in images] 

X = np.zeros((len(images), IMG_NEW_SIZE, IMG_NEW_SIZE, IMG_CHANNELS),dtype=np.uint8)
Y = np.zeros((len(images), IMG_NEW_SIZE, IMG_NEW_SIZE, 1),  dtype=np.bool)

for i, image_name in tqdm(enumerate(images),total=len(images)):   
    #if i > 100:
     #   break
    image = cv2.imread(IMAGE_DIR  + image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize(image, (IMG_NEW_SIZE, IMG_NEW_SIZE), mode='constant', preserve_range=True)
    X[i] = image     
    
    bb = BB_DIR +  bbs[i] 
    rectlist = read_rectangles(bb)
    rectlist_r =  [resize_position(x) for x in rectlist]
    mask = create_bbox_mask(rectlist_r ,IMG_NEW_SIZE)     
    Y[i] = mask


#np.save(file='Y',arr = Y)
#np.save(file='X',arr = X)

#X = np.load('X.npy')
#Y = np.load('Y.npy')



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=42)


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


#Build the model
import keras.backend.tensorflow_backend as K
K.set_image_dim_ordering("th")



def dice_coef(y_true, y_pred, smooth = 0):        
 
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f )
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

inputs = tf.keras.layers.Input((IMG_NEW_SIZE, IMG_NEW_SIZE, IMG_CHANNELS))
#s = inputs
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#s = inputs

cc  = 4

#Contraction path
c1 = tf.keras.layers.Conv2D(16*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
#c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
#c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
#c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
#c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
#c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128*cc, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
#c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64*cc, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
#c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32*cc, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
#c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16*cc, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
#c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16*cc, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 

################################
#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_astrocyte.h5', verbose=1, save_best_only=True)

np.save(file='X_train_1',arr = X_train)
np.save(file='X_test_1',arr = X_test)

np.save(file='Y_train_1',arr = Y_train)
np.save(file='Y_test_1',arr = Y_test)


callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

A = tf.keras.optimizers.Adam(lr=0.0001)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer = A, loss=dice_coef_loss, metrics=[dice_coef])
model.summary()

#model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.summary()

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=32, epochs=200, callbacks=callbacks)

####################################

history = results 
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('model dice_coef')
plt.ylabel('dice_coef')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



#################################### TEST 
import random as r
pred = model.predict(X_test, verbose=1)




fig, axarr = plt.subplots(2, 4, figsize=(20, 10))

  
for j in range(4):
     
     n = int(r.random() * pred.shape[0])

     #plt.subplot(131)
     #plt.title('Input'+str(i))
     #plt.imshow(X_test[i, :, :, 0])
     
     ind = n
 
     from skimage.segmentation import mark_boundaries
     ytestbound = mark_boundaries(X_test[ind],np.squeeze(Y_test[ind]),color = (0,0,0) )
     
     axarr[0,j].axis('off')
     axarr[0,j].set_title('Ground Truth')
     axarr[0,j].imshow(np.squeeze(ytestbound))
            
     predbound = mark_boundaries(X_test[ind],np.squeeze(pred[ind]>0.5),color = (0,0,0) )
     
     axarr[1,j].axis('off')
     axarr[1,j].set_title('Prediction')
     axarr[1,j].imshow(np.squeeze(predbound))
     #plt.imshow(pred[i, :, :,0]>0.5)

plt.show()
 

def cal_dice(A, B):
    intersection = np.logical_and(A, B)
    union = np.logical_or(A, B)
    iou = 2*np.sum(intersection > 0) / (np.sum(A) + np.sum(B))
    return iou

idx = random.randint(0, len(X_train))


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

iou=[]
for i in range(len(Y_train[int(Y_train.shape[0]*0.9):])):
    iou.append(cal_dice(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):]), np.squeeze(preds_val_t[i])))
print('Average Validate IOU: {}'.format(round(np.mean(iou),2)))

iou=[]
for i in range(len(Y_test)):
    iou.append(cal_dice(np.squeeze(Y_test[i]), np.squeeze(preds_test_t[i])))
print('Average Validate IOU: {}'.format(round(np.mean(iou),2)))


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()




# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()

