import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

def random_noise(x):
    x = x + (np.random.random(x.shape) - 0.5) / 20
    return x

def data_generator(image_path):
    # image_datagen = ImageDataGenerator()

    save_path = os.path.join(base_path, 'LG_CNS_data', 'By_datasetMK', '20190520_training_data_M3_2')
    for index in range(6):
        image_list = sorted(glob(os.path.join(image_path, 'fault{}'.format(index), '*.png')))
        # modi_batch_img = np.zeros((len(image_list), 256, 256, 3))

        if index == 0:
            print('================================= Fault type: ', index,'==========================================')
            for idx in range(1, 119):
                for batch, img in enumerate(image_list):
                    image = load_img(path=img)
                    image = img_to_array(image)
                    # image = image.reshape((1,) + image.shape)
                    modi_image = random_noise(image)

                    if os.path.exists(os.path.join(save_path, 'fault{}').format(index)) != True:
                        os.mkdir(os.path.join(save_path, 'fault{}').format(index))

                    modi_image_pil = array_to_img(modi_image)
                    modi_image_pil.save(os.path.join(save_path, 'fault{}', 'Augmented_' + img.split('/')[-1][:-4] + '_{:04d}.png').format(index, idx))
                    print('save the image: {}/{}, count index: {}/{}'.format(batch + 1, len(image_list), idx, 118))

        elif index == 1:
            print('================================= Fault type: ', index,'==========================================')
            for idx in range(1, 2):
                for batch, img in enumerate(image_list):
                    if not '21h16m33s' in img:
                        continue
                    image = load_img(path=img)
                    image = img_to_array(image)
                    # image = image.reshape((1,) + image.shape)
                    modi_image = random_noise(image)

                    if os.path.exists(os.path.join(save_path, 'fault{}').format(index)) != True:
                        os.mkdir(os.path.join(save_path, 'fault{}').format(index))

                    modi_image_pil = array_to_img(modi_image)
                    modi_image_pil.save(os.path.join(save_path, 'fault{}', 'Augmented_' + img.split('/')[-1][:-4] + '_{:04d}.png').format(index, idx+82))
                    print('save the image: {}/{}, count index: {}/{}'.format(batch + 1, len(image_list), idx, 1))

        elif 2 <= index <= 3:
            print('================================= Fault type: ', index,'==========================================')
            for idx in range(1, 2500):
                for batch, img in enumerate(image_list):
                    image = load_img(path=img)
                    image = img_to_array(image)
                    # image = image.reshape((1,) + image.shape)
                    modi_image = random_noise(image)

                    if os.path.exists(os.path.join(save_path, 'fault{}').format(index)) != True:
                        os.mkdir(os.path.join(save_path, 'fault{}').format(index))

                    modi_image_pil = array_to_img(modi_image)
                    modi_image_pil.save(os.path.join(save_path, 'fault{}', 'Augmented_' + img.split('/')[-1][:-4] + '_{:04d}.png').format(index, idx))
                    print('save the image: {}/{}, count index: {}/{}'.format(batch + 1, len(image_list), idx, 2499))

        elif index == 4:
            print('================================= Fault type: ', index,'==========================================')
            for idx in range(1, 417):
                for batch, img in enumerate(image_list):
                    image = load_img(path=img)
                    image = img_to_array(image)
                    # image = image.reshape((1,) + image.shape)
                    modi_image = random_noise(image)

                    if os.path.exists(os.path.join(save_path, 'fault{}').format(index)) != True:
                        os.mkdir(os.path.join(save_path, 'fault{}').format(index))

                    modi_image_pil = array_to_img(modi_image)
                    modi_image_pil.save(os.path.join(save_path, 'fault{}', 'Augmented_' + img.split('/')[-1][:-4] + '_{:04d}.png').format(index, idx))
                    print('save the image: {}/{}, count index: {}/{}'.format(batch + 1, len(image_list), idx, 416))

        elif index == 5:
            print('================================= Fault type: ', index,'==========================================')
            for idx in range(1, 1250):
                for batch, img in enumerate(image_list):
                    image = load_img(path=img)
                    image = img_to_array(image)
                    # image = image.reshape((1,) + image.shape)
                    modi_image = random_noise(image)

                    if os.path.exists(os.path.join(save_path, 'fault{}').format(index)) != True:
                        os.mkdir(os.path.join(save_path, 'fault{}').format(index))

                    modi_image_pil = array_to_img(modi_image)
                    modi_image_pil.save(os.path.join(save_path, 'fault{}', 'Augmented_' + img.split('/')[-1][:-4] + '_{:04d}.png').format(index, idx))
                    print('save the image: {}/{}, count index: {}/{}'.format(batch + 1, len(image_list), idx, 1249))

base_path = '/home/onepredict/Myungkyu/LG_CNS'
base_path_windows = 'D:\\Onepredict_MK\\LG CNS'

training_path = os.path.join(base_path, 'LG_CNS_data', 'By_datasetMK', '20190520_training_data_M3_2')
temp_path_windows = os.path.join(base_path_windows, 'By_datasetMK', '20190425_training_data_M3')

training_image = data_generator(training_path)
# training_image = data_generator_M3_windows(temp_path_windows)
