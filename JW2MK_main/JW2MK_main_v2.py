from MK_version.JW2MK_dataloader.JW2MK_dataloader_v2_window import *
from MK_version.JW2MK_model.JW2MK_model_v2 import *
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from glob import glob
from scipy import misc
from datetime import datetime
from pytz import timezone
import shutil
import argparse
import time

from sklearn.model_selection import KFold, train_test_split

parser = argparse.ArgumentParser('LG CNS training and test code of Myungkyu')
parser.add_argument('--mode',               type=str,       help='choosing either training or test', default='train')
parser.add_argument('--test_mode',          type=str,       help='choosing either training or test', default='train_test')
parser.add_argument('--main_name',          type=str,       help='current main file name', default='20190524_JW2MK_mainV2_JK_NoAug')
parser.add_argument('--base_path',          type=str,       help='path containing LG_CNS files',
                    default='/home/onepredict/Myungkyu/LG_CNS')
parser.add_argument('--test_image_path',    type=str,       help='path containing LG_CNS files',
                    default='/home/onepredict/Myungkyu/LG_CNS/JW_will/20190416_test_3sec_image')
                    # default='/home/onepredict/Myungkyu/LG_CNS/JW_will/20190403_test_CSV')
parser.add_argument('--test_result_path',    type=str,       help='path containing LG_CNS files',
                    default='/home/onepredict/Myungkyu/LG_CNS/JW_will/Test_result/20190523_big_test_result_JK_ep80')
parser.add_argument('--checkpoint_path',    type=str,       help='path containing LG_CNS files',
                    default='/home/onepredict/Myungkyu/LG_CNS/JW_will/MK_version/TrainedModel')
parser.add_argument('--learning_rate',      type=float,     help='initial learning rate', default=1e-4)
parser.add_argument('--batch_size',         type=int,       help='training image batch size', default=8)
parser.add_argument('--epochs',             type=int,       help='the number of training', default=80)
parser.add_argument('--num_threads',        type=int,       help='the number of cpu core', default=16)
parser.add_argument('--retrain',            type=int,       help='if used with checkpoint_path, will restart training from step zero',
                    default=True)
args = parser.parse_args()


def train(params):
    resnet_model50 = resnet50(params)
    print(resnet_model50.summary())

    tensorboard = TensorBoard(log_dir=os.path.join(params.base_path, 'JW_will', 'MK_version', 'Tensorboard', params.main_name),
                              write_images=True, batch_size=8)

    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(params.checkpoint_path, params.main_name+'_model', 'model-{epoch:04d}.ckpt'),
                                          save_weights_only=True,
                                          verbose=1)

    print('======================== Data Loading ======================== ')
    dataloader = DataGenerator(params)
    train_generator, validation_generator = dataloader.train_generator, dataloader.validation_generator

    print('======================== Training ======================== ')
    resnet_model50.fit_generator(generator=train_generator, steps_per_epoch=len(train_generator),
                                 epochs=params.epochs, validation_data=validation_generator,
                                 validation_steps=len(validation_generator),
                                 callbacks=[tensorboard, checkpoint_callback], workers=params.num_threads)

    print('======================== Evaluation loss and metrics !! ========================')
    loss_and_metrics = resnet_model50.evaluate_generator(
        generator=validation_generator, steps=len(validation_generator), workers=params.num_threads)
    print("%s: %.2f%%" % (resnet_model50.metrics_names[1], loss_and_metrics[1] * 100))
    print('========================================================================')

def test_v2(params):
    start = time.time()
    loaded_model = resnet50()

    model_name = 'model-0080.ckpt'
    best_model_path = os.path.join(params.checkpoint_path, params.main_name+'_model', model_name)
    loaded_model.load_weights(best_model_path)

    raw_data = glob(params.test_image_path + '/*.csv')

    for num, file in enumerate(sorted(raw_data)):
        data = np.genfromtxt(file, delimiter='. ')
        # f, t, Zxx = signal.stft(data, fs=128000, nperseg=512, noverlap=210)
        f, t, Zxx = signal.stft(data, fs=25600, nperseg=512, noverlap=10)
        Zxx = Zxx[1:, :]

        Sxx = np.abs(Zxx)
        Sxx = np.maximum(Sxx, 1e-8)
        mel = 20 * np.log10(Sxx)
        img = (mel + 160) / 160
        img = Image.fromarray(img, 'RGB')
        image = np.expand_dims(img, axis=0)
        """Class prediction"""
        class_pred = loaded_model.predict_classes(x=image, batch_size=1)
        time_sofar = (time.time() - start) / 3600
        print(time_sofar, class_pred)

def csv_test(params):
    loaded_model = resnet50()

    model_name = 'model-0080.ckpt'
    best_model_path = os.path.join(params.checkpoint_path, params.main_name + '_model', model_name)
    loaded_model.load_weights(best_model_path)

    test_csv = sorted(glob(args.test_csv_file + '\\*.csv'))

    for num, csv in enumerate(test_csv):
        data = np.genfromtxt(csv, delimiter=',')

        start = time.time()
        f, t, Zxx = signal.stft(data, fs=25600, nperseg=512, noverlap=10)
        Zxx = Zxx[1:, :]

        Sxx = np.abs(Zxx)
        Sxx = np.maximum(Sxx, 1e-8)
        mel = 20 * np.log10(Sxx)
        img = (mel + 160) / 160

        img = Image.fromarray(img, 'RGB')
        # array_img = np.asarray(img)
        image = np.expand_dims(img, axis=0)
        """Class prediction"""
        class_pred = loaded_model.predict_classes(x=image, batch_size=1)
        time_sofar = time.time() - start
        now = datetime.now()

        print('Current Time: {}-{}-{} {}:{}:{}.{}, Predicted class: {}, Checking test Image: {}/{}, Time elapsed: {:f}s'.format(
            now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond, class_pred, num+1, len(test_csv), time_sofar))

def Mando_test(params):
    start = time.time()
    loaded_model = resnet50()

    # model_name = 'model-0080.ckpt'
    # best_model_path = os.path.join(params.checkpoint_path, params.main_name+'_model', model_name)
    best_model_path = '/home/onepredict/JaeKyung/Mando+LGCNS_result/model-0063.ckpt'
    loaded_model.load_weights(best_model_path)

    Mando_images = sorted(glob('/home/onepredict/Myungkyu/Mando/Mando_data/유형별 정리_v3/10.작동음초과/*.png'))
    # test_images = sorted(glob(params.test_image_path + '/*.png'))

    for idx in range(13):
        if os.path.exists(os.path.join(params.test_result_path, 'test_class{}'.format(idx))) != True:
            os.makedirs(os.path.join(params.test_result_path, 'test_class{}'.format(idx)))

    for num, img in enumerate(Mando_images):
        image = misc.imread(img)
        image = np.expand_dims(image, axis=0)
        """Class prediction"""
        class_pred = loaded_model.predict_classes(x=image[:,:,:, 0:3], batch_size=1)
        time_sofar = (time.time() - start) / 3600

        print('Predicted class: {}, Checking test Image: {}/{}, Time elapsed: {:f}s'.format(class_pred, num+1, len(Mando_images), time_sofar))
        shutil.copy(img, os.path.join(params.test_result_path, 'test_class{}'.format(*class_pred)))

    """Plot the prediction"""
    test_batch = np.zeros((len(Mando_images), 256, 256, 3))
    for batch, image in enumerate(sorted(Mando_images)):
        """img_to_array?????????????????????????????????"""
        image = misc.imread(image)
        image = image[:, :, 0:3]
        test_batch[batch, :, :, :] = image

    prediction = loaded_model.predict(test_batch, batch_size=params.batch_size, workers=params.num_threads)
    plt.figure(figsize=(12, 12))
    plt.plot(np.argmax(prediction, 1), '.')
    plt.ylim([-0.5, 10.5])
    plt.show()

def test(params):
    start = time.time()
    loaded_model = resnet50()

    model_name = 'model-0080.ckpt'
    best_model_path = os.path.join(params.checkpoint_path, params.main_name+'_model', model_name)
    loaded_model.load_weights(best_model_path)

    test_images = sorted(glob(params.test_image_path + '/*.png'))

    for idx in range(7):
        if os.path.exists(os.path.join(params.test_result_path, 'test_class{}'.format(idx))) != True:
            os.makedirs(os.path.join(params.test_result_path, 'test_class{}'.format(idx)))

    for num, img in enumerate(test_images):
        if not '_3sec_nonhighpass' in img:
            continue
        image = misc.imread(img)
        image = np.expand_dims(image, axis=0)
        """Class prediction"""
        class_pred = loaded_model.predict_classes(x=image[:,:,:, 0:3], batch_size=1)
        time_sofar = (time.time() - start)

        print('Predicted class: {}, Checking test Image: {}/{}, Time elapsed: {:f}s'.format(class_pred, num+1, len(test_images), time_sofar))
        shutil.copy(img, os.path.join(params.test_result_path, 'test_class{}'.format(*class_pred)))

    """Plot the prediction"""
    test_batch = np.zeros((1924, 256, 256, 3))
    test_img = glob(params.test_image_path + '/*_3sec_nonhighpass.png')
    for batch, image in enumerate(sorted(test_img)):
        """img_to_array?????????????????????????????????"""
        image = misc.imread(image)
        image = image[:, :, 0:3]
        test_batch[batch, :, :, :] = image

    prediction = loaded_model.predict(test_batch, batch_size=params.batch_size, workers=params.num_threads)
    plt.figure(figsize=(12, 12))
    plt.plot(np.argmax(prediction, 1), '.')
    plt.ylim([-0.5, 6.5])
    plt.show()

def fault_test(params):
    loaded_model = resnet50()

    model_name = 'model-0079.ckpt'
    best_model_path = os.path.join(params.checkpoint_path, params.main_name + '_model', model_name)
    loaded_model.load_weights(best_model_path)

    fault_path = '/home/onepredict/Myungkyu/LG_CNS/LG_CNS_data/By_datasetMK/20190430_Fault_op_image'
    # fault_path = '/home/onepredict/Myungkyu/LG_CNS/LG_CNS_data/By_datasetMK/20190411_fault_image'

    for idx in range(4):
        if os.path.exists(os.path.join(params.test_result_path, 'test_class{}'.format(idx))) != True:
            os.makedirs(os.path.join(params.test_result_path, 'test_class{}'.format(idx)))

    for num in range(4):
        fault_images = glob(os.path.join(fault_path, 'fault{}'.format(num), '*.png'))
        for fault_img in sorted(fault_images):
            fault_image = misc.imread(fault_img)
            fault_image = np.expand_dims(fault_image, axis=0)
            fault_class_pred = loaded_model.predict_classes(x=fault_image[:,:,:, 0:3], batch_size=1)

            print('Predicted class: {}, Checking test Image: {}/{}'.format(fault_class_pred, num + 1, len(fault_images)))
            shutil.copy(fault_img, os.path.join(params.test_result_path, 'test_class{}'.format(*fault_class_pred)))

    """Plot the prediction"""
    fault_batches = ImageDataGenerator().flow_from_directory(directory=fault_path,
                                                            target_size=(256, 256),
                                                            batch_size=params.batch_size,
                                                            shuffle=False)

    prediction = loaded_model.predict_generator(fault_batches, verbose=1, workers=params.num_threads)
    plt.figure(figsize=(12, 12))
    plt.plot(np.argmax(prediction, 1), '.')
    plt.ylim([-0.5, 4.5])
    plt.show()

def train_val_test(params):
    loaded_model = resnet50()

    model_name = 'model-0079.ckpt'
    best_model_path = os.path.join(params.checkpoint_path, params.main_name+'_model', model_name)
    loaded_model.load_weights(best_model_path)

    if args.test_mode == 'train_test':
        training_path = os.path.join(params.base_path, 'LG_CNS_data', 'By_datasetMK', '20190508_training_data_M3_2')

        for idx in range(7):
            if os.path.exists(os.path.join(params.test_result_path, 'test_class{}'.format(idx))) != True:
                os.makedirs(os.path.join(params.test_result_path, 'test_class{}'.format(idx)))

        for num in range(6):
            fault_images = glob(os.path.join(training_path, 'fault{}'.format(num), '*.png'))
            for num_fault, fault_img in enumerate(sorted(fault_images)):
                fault_image = misc.imread(fault_img)
                fault_image = np.expand_dims(fault_image, axis=0)
                fault_class_pred = loaded_model.predict_classes(x=fault_image[:, :, :, 0:3], batch_size=1)

                print('Predicted class: {}, Checking test Image: {}/{}'.format(fault_class_pred, num_fault + 1,
                                                                            len(fault_images)))
                shutil.copy(fault_img, os.path.join(params.test_result_path, 'test_class{}'.format(*fault_class_pred)))

        normal_images = glob(os.path.join(training_path, 'normal', '*.png'))
        for num, normal_img in enumerate(sorted(normal_images)):
            normal_image = misc.imread(normal_img)
            normal_image = np.expand_dims(normal_image, axis=0)
            normal_class_pred = loaded_model.predict_classes(x=normal_image[:, :, :, 0:3], batch_size=1)

            print('Predicted class: {}, Checking test Image: {}/{}'.format(normal_class_pred, num + 1, len(normal_images)))
            shutil.copy(normal_img, os.path.join(params.test_result_path, 'test_class{}'.format(*normal_class_pred)))

        train_batches = ImageDataGenerator().flow_from_directory(directory=training_path,
                                                                 target_size=(256, 256),
                                                                 batch_size=params.batch_size,
                                                                 shuffle=False)
        prediction = loaded_model.predict_generator(train_batches, verbose=1, workers=params.num_threads)
        plt.figure(figsize=(12, 12))
        plt.plot(np.argmax(prediction, 1), '.')
        plt.ylim([-0.5, 6.5])
        plt.show()

    elif args.test_mode == 'val_test':
        validation_path = '/home/onepredict/Myungkyu/LG_CNS/LG_CNS_data/By_datasetMK/20190430_validation_data_M4'

        for idx in range(5):
            if os.path.exists(os.path.join(params.test_result_path, 'test_class{}'.format(idx))) != True:
                os.makedirs(os.path.join(params.test_result_path, 'test_class{}'.format(idx)))

        val_images = glob(validation_path + '/*.png')
        for num, val_img in enumerate(sorted(val_images)):
            val_image = misc.imread(val_img)
            val_image = np.expand_dims(val_image, axis=0)
            val_class_pred = loaded_model.predict_classes(x=val_image[:, :, :, 0:3], batch_size=1)

            print('Predicted class: {}, Checking test Image: {}/{}'.format(val_class_pred, num + 1, len(val_images)))
            shutil.copy(val_img, os.path.join(params.test_result_path, 'test_class{}'.format(*val_class_pred)))

        """Plot the prediction"""
        val_test_batch = np.zeros((3965, 256, 256, 3))
        for batch, image in enumerate(sorted(val_images)):
            image = misc.imread(image)
            image = image[:, :, 0:3]
            val_test_batch[batch, :, :, :] = image

        prediction = loaded_model.predict(val_test_batch, batch_size=params.batch_size, workers=params.num_threads)
        plt.figure(figsize=(12, 12))
        plt.plot(np.argmax(prediction, 1), '.')
        plt.ylim([-0.5, 5.5])
        plt.show()

def main():
    params = JW2MK_parameters(
        main_name=args.main_name,
        base_path=args.base_path,
        test_image_path=args.test_image_path,
        test_result_path=args.test_result_path,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        num_threads=args.num_threads,
    )

    if args.mode == 'train':
        if os.path.exists(os.path.join(params.base_path, 'JW_will', 'MK_version', 'Tensorboard', params.main_name)) != True:
            os.mkdir(os.path.join(params.base_path, 'JW_will', 'MK_version', 'Tensorboard', params.main_name))
        if os.path.exists(os.path.join(params.checkpoint_path, params.main_name + '_model')) != True:
            os.mkdir(os.path.join(params.checkpoint_path, params.main_name + '_model'))
        """dataloader, main, model code save"""
        shutil.copy(os.path.join(params.base_path, 'JW_will', 'MK_version', 'JW2MK_dataloader', 'JW2MK_dataloader_v3.py'),
                    os.path.join(params.base_path, 'JW_will', 'MK_version', 'Tensorboard', params.main_name))

        shutil.copy(os.path.join(params.base_path, 'JW_will', 'MK_version', 'JW2MK_main', 'JW2MK_main_v3.py'),
                    os.path.join(params.base_path, 'JW_will', 'MK_version', 'Tensorboard', params.main_name))

        shutil.copy(os.path.join(params.base_path, 'JW_will', 'MK_version', 'JW2MK_model', 'JW2MK_model_v3.py'),
                    os.path.join(params.base_path, 'JW_will', 'MK_version', 'Tensorboard', params.main_name))

        train(params)
    elif args.mode == 'test':
        test(params)
    elif args.mode == 'fault_test':
        fault_test(params)
    elif args.mode == 'train_val_test':
        train_val_test(params)
    elif args.mode == 'csv_test':
        csv_test(params)

if __name__ == '__main__':
    main()