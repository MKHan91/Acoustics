import os
import random
import shutil
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array

def random_noise(x):
    x = x + (np.random.random(x.shape) - 0.5) / 20
    return x

def computation(images):
    num = len(images)
    # x = 3500 // num
    # remain = 0
    #
    # if mul < 3500:
    #     remain = 3500 - mul
    Rval = 3500 - num
    x = Rval // num
    remain = Rval % num

    mul = num * x
    print('Predicted total number: {}'.format(mul+len(images)+remain))

    return x, remain

def check_dir_or_create(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def remain_data_generation(remain, images, type_):
    remain_images = random.sample(sorted(images), remain)

    for batch, img in enumerate(sorted(remain_images)):
        image = load_img(path=img)
        image = img_to_array(image)
        image = image / 255.0
        modi_image = random_noise(image)

        check_dir_or_create(dir=save_path + '/20190717_generated_data')
        check_dir_or_create(dir=save_path + '/20190717_generated_data' + '/{}'.format(type_))

        mimage_pil = array_to_img(modi_image)
        mimage_pil.save(os.path.join(save_path, '20190717_generated_data', type_, '{}.png')
                        .format('Augmented_remain_' + img.split('\\')[-1][:-4]))

        print('---------- data type: {}, ---------- Generated image: {}/{}'.format(type_, batch + 1, len(images)))

def data_generator_windows(image_path, save_path):
    for type_ in os.listdir(image_path):
        images = glob(os.path.join(image_path, type_, '*.png'))

        if 'fault' in type_:
            x, remain = computation(images)
            remain_data_generation(remain, images, type_)
            for num in range(x):
                for batch, img in enumerate(sorted(images)):
                    image = load_img(path=img)
                    image = img_to_array(image)
                    image = image / 255.0
                    modi_image = random_noise(image)

                    check_dir_or_create(dir=save_path + '/20190717_generated_data')
                    check_dir_or_create(dir=save_path + '/20190717_generated_data' + '/{}'.format(type_))

                    mimage_pil = array_to_img(modi_image)
                    mimage_pil.save(os.path.join(save_path, '20190717_generated_data', type_, '{}_{:04d}.png')
                                    .format('Augmented_' + img.split('\\')[-1][:-4], num))

                    print('---------- data type: {}, --------- The number per a image: {}, ---------- Generated image: {}/{}'
                          .format(type_, num, batch + 1, len(images)))

        elif type_ == 'normal':
            normal_imgs = glob(os.path.join(save_path, '20190717_normal_image', '*.png'))
            rand_norm_imges = random.sample(sorted(normal_imgs), 3500)

            for batch, img in enumerate(sorted(rand_norm_imges)):
                shutil.copy(img, os.path.join(data_path, 'normal', '{}.png'.format(img.split('\\')[-1][:-4])))
                print('----------------- copying! {}/{}'.format(batch + 1, len(rand_norm_imges)))

def temp():
    from imblearn.over_sampling import SMOTE

    SM = SMOTE(random_state=2)

if __name__ == '__main__':
    data_path = 'D:\\Onepredict_MK\\LG CNS\\By_datasetMK\\201907\\20190717_training_data'
    save_path = 'D:\\Onepredict_MK\\LG CNS\\By_datasetMK\\201907'
    data_generator_windows(data_path, save_path)