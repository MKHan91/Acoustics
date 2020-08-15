import tensorflow as tf
import os
import numpy as np
from glob import glob
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn import model_selection

def custom_labeling(base_path):
    class_list = ['normal', 'fault0', 'fault1', 'fault2', 'fault3', 'fault4', 'fault5']
    validation_path = os.path.join(base_path, 'LG_CNS_data', 'By_datasetMK', '20190417_validation_data')
    training_path = os.path.join(base_path, 'LG_CNS_data', 'By_datasetMK', '20190417_training_data')

    class_array_training = np.array([])
    class_array_validation = np.array([])

    """training data labeling"""
    for idx, class_name in enumerate(class_list):
        if class_name == 'normal':
            training_class = np.asarray([idx] * len(os.listdir(os.path.join(training_path, 'normal'))))
            class_array_training = np.concatenate([class_array_training, training_class], 0)
        else:
            training_class = np.asarray([idx] * len(os.listdir(os.path.join(training_path, class_name))))
            class_array_training = np.concatenate([class_array_training, training_class], 0)

    """validation data labeling"""
    for index, class_name in enumerate(class_list):
        if class_name == 'normal':
            training_class = np.asarray([index] * len(os.listdir(os.path.join(validation_path, 'normal'))))
            class_array_validation = np.concatenate([class_array_validation, training_class], 0)
        else:
            training_class = np.asarray([index] * len(os.listdir(os.path.join(validation_path, class_name))))
            class_array_validation = np.concatenate([class_array_validation, training_class], 0)

    training_label = tf.keras.utils.to_categorical(class_array_training, num_classes=len(class_list))
    validation_label = tf.keras.utils.to_categorical(class_array_validation, num_classes=len(class_list))

    return training_label, validation_label

class DataGenerator(object):

    def __init__(self, params):
        self.params = params
        self.train_generator, self.validation_generator = self.read_image(params.base_path)

    def read_image(self, image_paths):
        # training_path = os.path.join(image_paths, 'By_datasetMK', '201905', '20190508_training_data_M3_2')
        training_path = os.path.join(image_paths, '01 By_datasetMK', '201907', '20190717_training_data')

        data_generation = ImageDataGenerator(validation_split=0.3, rescale=1./255)
        train_generator = data_generation.flow_from_directory(directory=training_path,
                                                              target_size=(self.params.width, self.params.height),
                                                              batch_size=self.params.batch_size,
                                                              class_mode='categorical',
                                                              shuffle=True,
                                                              subset='training')

        validation_generator = data_generation.flow_from_directory(directory=training_path,
                                                                   target_size=(self.params.width, self.params.height),
                                                                   batch_size=self.params.batch_size,
                                                                   class_mode='categorical',
                                                                   shuffle=True,
                                                                   subset='validation')
        return train_generator, validation_generator
