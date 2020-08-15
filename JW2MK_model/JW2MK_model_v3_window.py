from tensorflow.python.keras.applications import ResNet50, densenet
from tensorflow.python.keras.layers import Dense, Flatten, BatchNormalization, Activation
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.backend import categorical_crossentropy
from collections import namedtuple
import tensorflow as tf

JW2MK_parameters = namedtuple('parameters',
                              'main_name, '
                              'base_path,'
                              'test_image_path,'
                              'test_result_path,'
                              'checkpoint_path,'
                              'batch_size,'
                              'learning_rate,'
                              'epochs,'
                              'num_threads,')


def resnet_windows():
    model = Sequential()
    # densenet121 = densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(256, 256, 3), pooling='avg')
    resnet = ResNet50(include_top=False, pooling='avg', weights='imagenet', input_shape=(256, 256, 3))
    model.add(resnet)
    # model.add(densenet121)
    # model.add(Flatten())
    # model.add(BatchNormalization())

    model.add(Dense(2048, activation=None))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1024, activation=None))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(512, activation=None))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(7, activation='softmax'))
    # model.add(BatchNormalization())
    # model.add(Activation('softmax'))

    # resnet = ResNet50(include_top=False, pooling='avg', weights='imagenet', input_shape=(256, 256, 3))
    # model.add(resnet)
    # model.add(Flatten())
    # model.add(BatchNormalization())
    # model.add(Dense(2048, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(1024, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(7, activation='softmax'))

    # model.layers[0].trainable = False

    return model