import os
import keras as k

from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
import densenet


def model1(input_dim):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(input_dim, input_dim, 3)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))
    return model

def model2(input_dim):
    model_ = Sequential()
    model_.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(input_dim, input_dim, 3)))

    model_.add(Conv2D(64, (3, 3), activation='relu'))
    model_.add(MaxPooling2D(pool_size=(2, 2)))

    model_.add(Conv2D(128, (3, 3), activation='relu'))

    model_.add(Conv2D(128, (3, 3), activation='relu'))
    model_.add(MaxPooling2D(pool_size=(2, 2)))
    model_.add(Dropout(0.25))
    model_.add(Flatten())
    model_.add(Dense(128, activation='relu'))
    model_.add(Dropout(0.5))
    model_.add(Dense(17, activation='sigmoid'))
    return model_

def model11(input_dim):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(input_dim, input_dim, 3)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))
    return model
def model3(input_dim):
    model = Sequential()
    # preprocess
    model.add(BatchNormalization(input_shape=(input_dim, input_dim, 3)))
    model.add(Conv2D(16, kernel_size=(1, 1), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, (1, 1), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, (1, 1), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(BatchNormalization(input_shape=(input_dim, input_dim, 3)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))
    return model

def pynet12(input_dim):
    model = Sequential()
    # preprocess
    model.add(BatchNormalization(input_shape=(input_dim, input_dim, 3)))
    model.add(Conv2D(16, kernel_size=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, (1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, (1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # conv1d
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # conv2d
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))
    return model

def vgg16():
    base_model = VGG16(weights=None, include_top=False, input_shape = (224,224,3))
    # Classification block
    x = Flatten(name='flatten', input_shape=base_model.output_shape[1:])(base_model.output)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dense(17, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model

def vgg19():
    base_model = VGG19(weights=None, include_top=False, input_shape = (224,224,3))
    x = Flatten(name='flatten')(base_model.output)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dense(17, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def resNet50():
    base_model = ResNet50(weights=None, include_top=False, input_shape = (224,224,3))
    x = Flatten(name='flatten')(base_model.output)
    x = Dense(17, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def inceptionV3():
    base_model = InceptionV3(weights=None, include_top=False, input_shape = (224,224,3))
    # Classification block
    x = InceptionV3.GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    x = Dense(17, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def denseNet(input_dim):
    base_model = densenet.DenseNet(input_shape=(input_dim, input_dim, 3), classes=17, dropout_rate=0.2, weights=None, include_top=False)
    x = Dense(17, activation='softmax', kernel_regularizer=regularizers.L1L2(l2=1E-4),
              bias_regularizer=regularizers.L1L2(l2=1E-4))(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)

    # Load model
    weights_file = "../weights/DenseNet-40-12CIFAR10-tf.h5"
    if os.path.exists(weights_file):
        model.load_weights(weights_file)
        print("Model loaded.")
    return model