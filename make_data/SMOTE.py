from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE
from kmeans_smote import KMeansSMOTE
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.manifold import TSNE
from scipy import misc
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd
import os


def custom_labeling(base_path):
    class_list = ['fault0', 'fault1', 'fault2', 'fault3', 'fault4', 'fault5']
    fault_path = os.path.join(base_path, 'By_datasetMK', '201904', '20190411_fault_image')

    class_array_fault = np.array([])

    """fault data labeling"""
    for idx, class_name in enumerate(class_list):
        fault_class = np.asarray([idx] * len(os.listdir(os.path.join(fault_path, class_name))))
        class_array_fault = np.concatenate([class_array_fault, fault_class], 0)

    fault_label = tf.keras.utils.to_categorical(class_array_fault, num_classes=len(class_list))

    return fault_label

def fault_data(dir):
    fault_label = custom_labeling(base_path=dir)

    fault_target = np.array([])
    for idx in range(len(fault_label)):
        pos = np.argmax(fault_label[idx])
        fault_target = np.concatenate([fault_target, [pos]])

    fault_path = os.path.join(dir, 'By_datasetMK', '201904', '20190411_fault_image')
    fault_image = np.zeros((len(fault_label), 196608))

    for Ftype in sorted(os.listdir(fault_path)):
        images = glob(os.path.join(fault_path, Ftype, "*.png"))
        for batch, img in enumerate(sorted(images)):
            image = misc.imread(img)
            image = image[:, :, 0:3]
            # image = image.ravel()
            image = image.flatten()
            fault_image[batch, :] = image

    X = np.vstack([fault_image[fault_target == i] for i in range(6)])
    y = np.hstack([fault_target[fault_target == i] for i in range(6)])

    return X, y

def t_sne(fault_image):
    tsne = TSNE(n_components=2, verbose=1, n_iter=1000)
    fault_tsne = tsne.fit_transform(X=fault_image)

    target_ids = range(6)
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, np.unique(np.int32(fault_label))):
        plt.scatter(fault_tsne[fault_label == i, 0], fault_tsne[fault_label == i, 1], c=c, label='Fault{}'.format(label))

    plt.legend(loc='best')
    plt.show()

def smote():
    sm = SMOTE(sampling_strategy='auto', random_state=2, kind='regular')
    fdata_res, flabel_res = sm.fit_sample(X=fault_image, y=fault_label[:, 0])

    x = fdata_res.reshape(len(fdata_res), 256, 256, 3)[0, :, :, :]
    plt.imshow(x)
    plt.show()

if __name__ == '__main__':
    fault_image, fault_label = fault_data(dir='D:\\Onepredict_MK\\LG CNS')
    t_sne(fault_image)