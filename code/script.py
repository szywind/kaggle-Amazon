import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import imutils

import keras as k
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

import cv2
from tqdm import tqdm

import resnet
import newnet
import random

def comp_mean(imglist):
    mean = [0, 0, 0]
    for img in imglist:
        mean += np.mean(np.mean(img, axis=0), axis=0)
    return mean/len(imglist)

rescaled_dim = 64
batch_size = 128

x_train = []
x_test = []
y_train = []
df_train = pd.read_csv('../data/train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.imread('../data/train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1

    # loop over the rotation angles
    for angle in np.arange(0, 360, 90):
        rotated = imutils.rotate(img, angle)
        new_img = cv2.resize(rotated, (rescaled_dim, rescaled_dim))
        x_train.append(new_img)
        y_train.append(targets)

        x_train.append(cv2.flip(new_img, 1))
        y_train.append(targets)
    
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.uint8)
# x_train = np.array(x_train, np.float16) / 255.


print(x_train.shape)
print(y_train.shape)

del df_train

mean = np.round(comp_mean(x_train)).astype('uint8')
# mean = comp_mean(x_train)

split = 35000*8
# x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]

N = len(x_train)

all = list(range(N))
np.random.shuffle(all)
train_index = all[:split]
val_index = all[split:]
assert(len(train_index) + len(val_index) == N)

for i in xrange(N):
    x_train[i] -= mean

x_valid = x_train[val_index]
x_train = x_train[train_index]
y_valid = y_train[val_index]
y_train = y_train[train_index]

# x_train, x_valid, y_train, y_valid = x_train[train_index], x_train[val_index], y_train[train_index], y_train[val_index]

# model = newnet.run_normal(rescaled_dim)

# modiel = resnet.ResnetBuilder.build_resnet_18((3, rescaled_dim, rescaled_dim), 17)

base_model = VGG16(weights=None, include_top=False)
base_model.load_weights('/home/szywind/Desktop/tmp/vgg16.h5')

# base_model = VGG16(weights='/home/szywind/Desktop/tmp/vgg16.h5', include_top=False)
predictions = Dense(17, activation='softmax')(base_model.output)
model = Model(inputs=base_model.input, outputs=predictions)

K.set_image_dim_ordering('tf')




model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
            optimizer='adam',
            metrics=['accuracy'])

model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=40,
        verbose=1,
        validation_data=(x_valid, y_valid))

from sklearn.metrics import fbeta_score

p_valid = model.predict(x_valid, batch_size=batch_size)
print(y_valid)
print(p_valid)
avg_sample_score = fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')
print(avg_sample_score)



df_sub = pd.read_csv('../data/sample_submission_v2.csv')
x_sub = []
for name in df_sub['image_name'].values:
    if name.startswith('test'):
        img = cv2.imread('../data/test-jpg/{}.jpg'.format(name))
    else:
        img = cv2.imread('../data/test-jpg-additional/{}.jpg'.format(name))
    x_sub.append(cv2.resize(img, (rescaled_dim, rescaled_dim)))

x_sub = np.array(x_sub, np.float16)

for i in xrange(len(x_sub)):
    x_sub[i] -= mean

# x_sub = np.array(x_sub, np.float16) / 255.
p_sub = model.predict(x_sub, batch_size=batch_size)
all_test_tags = []
for index in range(p_sub.shape[0]):
    all_test_tags.append(' '.join(list(map(lambda x: inv_label_map[x], *np.where(p_sub[index, :] > 0.5)))))

df_sub['tags'] = all_test_tags
df_sub.head()
df_sub.to_csv('../ovr_f2_{}.csv'.format(avg_sample_score), index=False)

