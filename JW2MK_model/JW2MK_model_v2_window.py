from tensorflow.python.keras.applications import ResNet50, DenseNet121, inception_resnet_v2
from tensorflow.python.keras.layers import Dense, Flatten, BatchNormalization, Activation, Add
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import Model
from collections import namedtuple

JW2MK_parameters = namedtuple('parameters',
                              'width,'
                              'height,'
                              'main_name, '
                              'pre_main_name,'
                              'base_path,'
                              'checkpoint_path,'
                              'batch_size,' 
                              'learning_rate,'
                              'epochs,'
                              'num_threads,')


def incept_resnet_v2_windows(params):
    InceptResNetV2 = inception_resnet_v2.InceptionResNetV2(include_top=False, pooling='avg', weights='imagenet',
                                                           input_shape=(params.width, params.height, 3))
    x = InceptResNetV2.output
    # x = Dense(2048, activation=None)(x)
    # x = Activation('relu')(x)
    # x = Dense(1024, activation=None)(x)
    # x = Activation('relu')(x)
    # x = Dense(512, activation=None)(x)
    # x = Activation('relu')(x)
    x = Dense(7, activation='softmax', name='dense_softmax')(x)
    InceptResNetV2 = Model(InceptResNetV2.input, x)

    return InceptResNetV2

def densenet_windows(params):
    densenet = DenseNet121(include_top=False, pooling='avg', weights='imagenet',
                           input_shape=(params.width, params.height, 3))

    x = densenet.output
    x = Dense(2048, activation=None)(x)
    x = Activation('relu')(x)
    x = Dense(1024, activation=None)(x)
    x = Activation('relu')(x)
    x = Dense(512, activation=None)(x)
    x = Activation('relu')(x)
    x = Dense(7, activation='softmax', name='dense_softmax')(x)
    densenet = Model(densenet.input, x)

    return densenet

def resnet_windows(params):
    resnet = ResNet50(include_top=False, pooling='avg', weights='imagenet',
                      input_shape=(params.width, params.height, 3))

    x = resnet.output
    x = Dense(2048, activation=None)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(1024, activation=None)(x)
    x = Activation('relu')(x)
    # x = BatchNormalization()(x)

    x = Dense(512, activation=None)(x)
    x = Activation('relu')(x)
    # x = BatchNormalization()(x)

    x = Dense(7, activation='softmax', name='dense_softmax')(x)
    resnet = Model(resnet.input, x)

    return resnet
