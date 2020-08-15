from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dense, Flatten, BatchNormalization, Conv2D, Activation, add, ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.backend import image_data_format

from collections import namedtuple

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

def resnet50(params):
    model = Sequential()
    resnet = ResNet50(include_top=False, pooling='avg', weights='imagenet', input_shape=(256, 256, 3))
    model.add(resnet)
    model.add(Flatten())
    model.add(BatchNormalization())

    model.add(Dense(2048, activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1024, activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(512, activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(7, activation=None))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    opt = Adam(lr=params.learning_rate, decay=params.learning_rate/params.epochs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model