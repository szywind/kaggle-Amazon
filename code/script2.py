# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../data"]).decode("utf8"))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import cv2
from tqdm import tqdm
from keras import optimizers

from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import fbeta_score
import time
import newnet
import imutils

global input_dim, batch_size, nfolds, model

input_dim = 64
batch_size = 128
nfolds = 5

model = newnet.model3(input_dim)

# model = newnet.model11(input_dim)
# model = newnet.denseNet(input_dim)

# def data_augmentation(img):


def comp_mean(imglist):
    mean = [0, 0, 0]
    for img in imglist:
        mean += np.mean(np.mean(img, axis=0), axis=0)
    return mean/len(imglist)

def find_f_measure_threshold2(probs, labels, num_iters=100, seed=0.21):
    _, num_classes = labels.shape[0:2]
    best_thresholds = [seed] * num_classes
    best_scores = [0] * num_classes
    for t in range(num_classes):

        thresholds = list(best_thresholds)  # [seed]*num_classes
        for i in range(num_iters):
            th = i / float(num_iters)
            thresholds[t] = th
            f2 = fbeta_score(labels, probs > thresholds, beta=2, average='samples')
            if f2 > best_scores[t]:
                best_scores[t] = f2
                best_thresholds[t] = th
        print('\t(t, best_thresholds[t], best_scores[t])=%2d, %0.3f, %f' % (t, best_thresholds[t], best_scores[t]))
    print('')
    return best_thresholds, best_scores


''' ---------------------------- training phase ---------------------------- '''
def train():
    df_train = pd.read_csv('../data/train_v2.csv')

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

    global label_map, inv_label_map
    label_map = {l: i for i, l in enumerate(labels)}
    inv_label_map = {i: l for l, i in label_map.items()}

    print ("inv_label_map:\n{}".format(inv_label_map))
    x_train = []
    y_train = []
    for f, tags in tqdm(df_train.values, miniters=1000):
        img = cv2.imread('../data/train-jpg/{}.jpg'.format(f))
        targets = np.zeros(17)
        Tags = tags.split(' ')
        aug_flip = False
        for t in Tags:
            targets[label_map[t]] = 1
            if not aug_flip and t in ['conventional_mine', 'blow_down', 'slash_burn', 'blooming', 'artisinal_mine', 'selective_logging', 'bare_ground']:
                aug_flip = True


        img = cv2.resize(img, (input_dim, input_dim))
        # loop over the rotation angles
        for angle in np.arange(0, 360, 90):
            new_img = imutils.rotate(img, angle)
            x_train.append(new_img)
            y_train.append(targets)
            if aug_flip:
                x_train.append(cv2.flip(new_img, 1))
                y_train.append(targets)

            # if len(x_train) > 1000:
            #     break

    y_train = np.array(y_train, np.uint8)
    x_train = np.array(x_train, np.float16)/255.0

    # mean = np.round(comp_mean(x_train)).astype('uint8')
    # for i in xrange(len(x_train)):
    #     x_train[i] -= mean

    print(x_train.shape)
    print(y_train.shape)

    num_fold = 0
    sum_score = 0
    yfull_train =[]

    kf = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=1)
    # kf = StratifiedKFold(y_train, n_folds=nfolds, shuffle=True, random_state=1)

    thres = []
    val_score = 0
    for train_index, test_index in kf:
        start_time_model_fitting = time.time()

        X_train = x_train[train_index]
        Y_train = y_train[train_index]
        X_valid = x_train[test_index]
        Y_valid = y_train[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')

        # epochs_arr = [50, 50, 50]
        # learn_rates = [0.001, 0.0001, 0.0001]

        epochs_arr = [50]
        learn_rates = [0.0001]

        for learn_rate, epochs in zip(learn_rates, epochs_arr):
            opt  = optimizers.Adam(lr=learn_rate)
            model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                          optimizer=opt,
                          metrics=['accuracy'])
            # callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
            # ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

            callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6),
                         EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=8, verbose=0),
                         ModelCheckpoint(kfold_weights_path, monitor='val_acc', save_best_only=True, verbose=0)]

            model.fit(x = X_train, y= Y_train, validation_data=(X_valid, Y_valid),
                  batch_size=batch_size,verbose=2, epochs=epochs,callbacks=callbacks,shuffle=True)

        if os.path.isfile(kfold_weights_path):
            model.load_weights(kfold_weights_path)

        p_valid = model.predict(X_valid, batch_size = batch_size, verbose=2)

        ## find best thresholds for each class
        best_threshold, best_scores = find_f_measure_threshold2(p_valid, Y_valid)

        thres.append(best_threshold)
        val_score += best_scores[-1]
        # print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

        # p_train = model.predict(x_train, batch_size =128, verbose=2)
        # yfull_train.append(p_train)

    # del df_train
    # del x_train
    # del y_train
    # del X_train
    # del Y_train
    # del X_valid
    # del Y_valid
    return thres, val_score/float(nfolds)

'''-------------------------- test phase -----------------------------'''
def test(thres, val_score, early_fusion=True, augmentation = 1, mirror=True):
    x_test = []
    df_test = pd.read_csv('../data/sample_submission_v2.csv')

    for f, tags in tqdm(df_test.values, miniters=1000):
        if f.startswith('test'):
            img = cv2.imread('../data/test-jpg/{}.jpg'.format(f))
            img = cv2.resize(img, (input_dim, input_dim))
            for angle in np.arange(0, augmentation*90, 90):
                new_img = imutils.rotate(img, angle)
                x_test.append(new_img)
                if mirror:
                    x_test.append(cv2.flip(new_img, 1))
        else:
            img = cv2.imread('../data/test-jpg-additional/{}.jpg'.format(f))
            img = cv2.resize(img, (input_dim, input_dim))
            for angle in np.arange(0, augmentation*90, 90):
                new_img = imutils.rotate(img, angle)
                x_test.append(new_img)
                if mirror:
                    x_test.append(cv2.flip(new_img, 1))

    x_test  = np.array(x_test, np.float16)/255.0
    # for i in xrange(len(x_test)):
    #     x_test[i] -= mean

    ntimes = augmentation * (1 + mirror)
    nsamples = len(df_test)

    yfull_test = []

    for i in xrange(nfolds):
        kfold_weights_path = os.path.join('', 'weights_kfold_' + str(i+1) + '.h5')
        if os.path.isfile(kfold_weights_path):
            model.load_weights(kfold_weights_path)
            p_test = model.predict(x_test, batch_size = batch_size, verbose=2)
            if ntimes > 1:
                for j in xrange(nsamples):
                    p_test[j,:] = np.mean(p_test[j*ntimes:(j+1)*ntimes], 0)

            yfull_test.append(p_test[:nsamples])

    raw_result = np.zeros(yfull_test[0].shape)
    if early_fusion:
        thresh = np.zeros([1, len(thres[0])])
        for i in xrange(nfolds):
            raw_result += yfull_test[i]
            thresh += thres[i]
        raw_result /= float(nfolds)
        thresh /= float(nfolds)
        result = (raw_result > thresh)
    else:
        for i in xrange(nfolds):
            raw_result += (yfull_test[i] > thres[i])
        result = raw_result / float(nfolds)


    # thres = [0.07, 0.17, 0.2, 0.04, 0.23, 0.33, 0.24, 0.22, 0.1, 0.19, 0.23, 0.24, 0.12, 0.14, 0.25, 0.26, 0.16]
    # thres /= float(nfolds)
    # result = (result > thres)

    global inv_label_map
    preds = []
    for index in range(result.shape[0]):
        pred = ' '.join(list(map(lambda x: inv_label_map[x], *np.where(result[index, :] == 1))))
        if len(pred) == 0:
            if early_fusion:
                pred = ' '.join(list(map(lambda x: inv_label_map[x], *np.argmax(raw_result[index, :] - thresh))))
            else:
                pred = ' '.join(list(map(lambda x: inv_label_map[x], *np.argmax(raw_result[index, :]))))
        preds.append(pred)

    df_test['tags'] = preds
    df_test.to_csv('../submission_keras_5_fold_CV_{}_LB_.csv'.format(val_score), index=False)


if __name__ == "__main__":

    #
    # labels = ['blow_down',
    #  'bare_ground',
    #  'conventional_mine',
    #  'blooming',
    #  'cultivation',
    #  'artisinal_mine',
    #  'haze',
    #  'primary',
    #  'slash_burn',
    #  'habitation',
    #  'clear',
    #  'road',
    #  'selective_logging',
    #  'partly_cloudy',
    #  'agriculture',
    #  'water',
    #  'cloudy']
    #
    # label_map = {'agriculture': 14,
    #  'artisinal_mine': 5,
    #  'bare_ground': 1,
    #  'blooming': 3,
    #  'blow_down': 0,
    #  'clear': 10,
    #  'cloudy': 16,
    #  'conventional_mine': 2,
    #  'cultivation': 4,
    #  'habitation': 9,
    #  'haze': 6,
    #  'partly_cloudy': 13,
    #  'primary': 7,
    #  'road': 11,
    #  'selective_logging': 12,
    #  'slash_burn': 8,
    #  'water': 15}

    inv_label_map = {0: 'slash_burn', 1: 'clear', 2: 'blooming', 3: 'primary', 4: 'cloudy', 5: 'conventional_mine', 6: 'water',
     7: 'haze', 8: 'cultivation', 9: 'partly_cloudy', 10: 'artisinal_mine', 11: 'habitation', 12: 'bare_ground',
     13: 'blow_down', 14: 'agriculture', 15: 'road', 16: 'selective_logging'}
    #
    # thresh = [[0.26, 0.15, 0.25, 0.23, 0.07, 0.13, 0.17, 0.26, 0.21, 0.09, 0.28, 0.19, 0.16, 0.41, 0.17, 0.14, 0.18],
    #           [0.21, 0.21, 0.14, 0.22, 0.08, 0.23, 0.17, 0.16, 0.20, 0.08, 0.15, 0.19, 0.19, 0.31, 0.20, 0.15, 0.16],
    #           [0.22, 0.16, 0.12, 0.23, 0.10, 0.30, 0.17, 0.20, 0.19, 0.12, 0.27, 0.19, 0.24, 0.26, 0.13, 0.18, 0.18],
    #           [0.19, 0.21, 0.18, 0.22, 0.08, 0.23, 0.14, 0.18, 0.25, 0.22, 0.33, 0.21, 0.23, 0.46, 0.16, 0.16, 0.18],
    #           [0.20, 0.23, 0.13, 0.24, 0.15, 0.25, 0.17, 0.24, 0.22, 0.20, 0.27, 0.21, 0.20, 0.29, 0.18, 0.25, 0.17]]
    #
    # val_score = 0.953954
    # thresh, val_score = train()
    thresh = [[0.19, 0.12, 0.2, 0.15, 0.12, 0.16, 0.15, 0.18, 0.19, 0.17, 0.23, 0.15, 0.16, 0.13, 0.23, 0.19, 0.09], [0.15, 0.18, 0.29, 0.19, 0.16, 0.11, 0.17, 0.16, 0.22, 0.13, 0.34, 0.26, 0.14, 0.12, 0.23, 0.18, 0.16], [0.2, 0.18, 0.19, 0.18, 0.09, 0.1, 0.17, 0.2, 0.23, 0.18, 0.33, 0.16, 0.16, 0.12, 0.16, 0.19, 0.23], [0.18, 0.17, 0.14, 0.18, 0.14, 0.1, 0.15, 0.18, 0.23, 0.15, 0.33, 0.18, 0.14, 0.15, 0.22, 0.2, 0.17], [0.16, 0.13, 0.17, 0.19, 0.07, 0.22, 0.14, 0.22, 0.22, 0.2, 0.22, 0.17, 0.18, 0.2, 0.2, 0.17, 0.09]]
    val_score = 0.92654
    print("thresh:\n{}".format(thresh))
    print("val_score:", val_score)
    test(thres=thresh, val_score=val_score, early_fusion=True, augmentation=4, mirror=True)
