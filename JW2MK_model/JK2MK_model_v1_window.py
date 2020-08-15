from tensorflow.python.keras.applications import ResNet50, DenseNet121, inception_resnet_v2
from tensorflow.python.keras.layers import Dense, Flatten, BatchNormalization, Activation, Add
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from collections import namedtuple

JW2MK_parameters = namedtuple('parameters',
                              'width,'
                              'height,'
                              'main_name, '
                              'pre_main_name,'
                              'base_path,'
                              'test_image_path,'
                              'test_result_path,'
                              'checkpoint_path,'
                              'batch_size,' 
                              'learning_rate,'
                              'epochs,'
                              'num_threads,')
def resnet_window():
    model = Sequential()
    resnet = ResNet50(include_top=False, pooling='avg', weights='imagenet', input_shape=(256, 256, 3))

    model.add(resnet)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    # model.add(Dense(13, activation='relu'))
    model.add(Dense(2, activation=None))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    opt = Adam(lr=1e-4, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model