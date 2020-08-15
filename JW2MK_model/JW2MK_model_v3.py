from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dense, Flatten, BatchNormalization, Conv2D, Activation, add, ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.backend import image_data_format

from collections import namedtuple
import os
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

# def identity_block(input_tensor, kernel_size, filters, stage, block):
#     filters1, filters2, filters3 = filters
#     if image_data_format() == 'channels_last':
#         bn_axis = 3
#     else:
#         bn_axis = 1
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = Conv2D(filters1, (1, 1),
#                       kernel_initializer='he_normal',
#                       name=conv_name_base + '2a')(input_tensor)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters2, kernel_size,
#                       padding='same',
#                       kernel_initializer='he_normal',
#                       name=conv_name_base + '2b')(x)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters3, (1, 1),
#                       kernel_initializer='he_normal',
#                       name=conv_name_base + '2c')(x)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
#
#     x = add([x, input_tensor])
#     x = Activation('relu')(x)
#     return x
#
# def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
#
#     filters1, filters2, filters3 = filters
#     if image_data_format() == 'channels_last':
#         bn_axis = 3
#     else:
#         bn_axis = 1
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
#
#     shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '1')(input_tensor)
#     shortcut = BatchNormalization( axis=bn_axis, name=bn_name_base + '1')(shortcut)
#
#     x = add([x, shortcut])
#     x = Activation('relu')(x)
#     return x
#
# def ResNet50_MK(include_top=True,
#              weights='imagenet',
#              input_tensor=None,
#              input_shape=None,
#              pooling=None,
#              classes=7):
#
#     global backend, layers, models, keras_utils
#     backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
#
#     if not (weights in {'imagenet', None} or os.path.exists(weights)):
#         raise ValueError('The `weights` argument should be either '
#                          '`None` (random initialization), `imagenet` '
#                          '(pre-training on ImageNet), '
#                          'or the path to the weights file to be loaded.')
#
#     if weights == 'imagenet' and include_top and classes != 1000:
#         raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
#                          ' as true, `classes` should be 1000')
#
#     # Determine proper input shape
#     input_shape = _obtain_input_shape(input_shape,
#                                       default_size=224,
#                                       min_size=32,
#                                       data_format=backend.image_data_format(),
#                                       require_flatten=include_top,
#                                       weights=weights)
#
#     if input_tensor is None:
#         img_input = layers.Input(shape=input_shape)
#     else:
#         if not backend.is_keras_tensor(input_tensor):
#             img_input = layers.Input(tensor=input_tensor, shape=input_shape)
#         else:
#             img_input = input_tensor
#     if backend.image_data_format() == 'channels_last':
#         bn_axis = 3
#     else:
#         bn_axis = 1
#
#     x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
#     x = Conv2D(64, (7, 7),
#                       strides=(2, 2),
#                       padding='valid',
#                       kernel_initializer='he_normal',
#                       name='conv1')(x)
#     x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
#     x = Activation('relu')(x)
#     x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
#     x = MaxPooling2D((3, 3), strides=(2, 2))(x)
#
#     x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
#
#     x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
#
#     x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
#
#     x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
#
#     if include_top:
#         x = GlobalAveragePooling2D(name='avg_pool')(x)
#         x = Dense(classes, activation='softmax', name='fc1000')(x)
#     else:
#         if pooling == 'avg':
#             x = GlobalAveragePooling2D()(x)
#         elif pooling == 'max':
#             x = GlobalMaxPooling2D()(x)
#         else:
#             warnings.warn('The output shape of `ResNet50(include_top=False)` '
#                           'has been changed since Keras 2.2.0.')
#
#     # Ensure that the model takes into account
#     # any potential predecessors of `input_tensor`.
#     if input_tensor is not None:
#         inputs = keras_utils.get_source_inputs(input_tensor)
#     else:
#         inputs = img_input
#     # Create model.
#     model = models.Model(inputs, x, name='resnet50')
#
#     # Load weights.
#     if weights == 'imagenet':
#         if include_top:
#             weights_path = keras_utils.get_file(
#                 'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
#                 WEIGHTS_PATH,
#                 cache_subdir='models',
#                 md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
#         else:
#             weights_path = keras_utils.get_file(
#                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
#                 WEIGHTS_PATH_NO_TOP,
#                 cache_subdir='models',
#                 md5_hash='a268eb855778b3df3c7506639542a6af')
#         model.load_weights(weights_path)
#         if backend.backend() == 'theano':
#             keras_utils.convert_all_kernels_in_model(model)
#     elif weights is not None:
#         model.load_weights(weights)
#
#     return model

def resnet50():
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
    # model.add(Flatten())
    # model.add(BatchNormalization())
    # model.add(Dense(2048, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(1024, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(7, activation='softmax'))

    # model.layers[0].trainable = False

    return model