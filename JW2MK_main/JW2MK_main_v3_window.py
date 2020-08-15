# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from MK_version.JW2MK_dataloader.JW2MK_dataloader_v3_window import *
from MK_version.JW2MK_model.JW2MK_model_v3_window import *
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import array_to_img
from pytz import timezone
from datetime import datetime
from glob import glob
from scipy import misc, signal
from PIL import Image
import shutil
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

parser = argparse.ArgumentParser('LG CNS training and test code of Myungkyu')
parser.add_argument('--mode',           type=str,       help='choosing either training or test', default='test')
parser.add_argument('--test_mode',      type=str,   help='choosing either training or test', default='train_test')
parser.add_argument('--main_name',      type=str,       help='current main file name', default='20190613_JW2MK_mainV3-9_M3_2')
parser.add_argument('--base_path',      type=str,       help='path containing LG_CNS files', default='D:\\Onepredict_MK\\LG CNS')
parser.add_argument('--test_image_path',    type=str,       help='path containing LG_CNS files',
                    default='D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\20190416_test_3sec_image')
parser.add_argument('--test_result_path',    type=str,       help='path containing LG_CNS files',
                    default='D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\Test_result\\201906\\20190613_JW2MK_mainV3-9_M3_2_ep72')
parser.add_argument('--checkpoint_path',    type=str,       help='path containing LG_CNS files',
                    default='D:\\Onepredict_MK\\LG CNS\\Training_result\\TrainedModel')
parser.add_argument('--learning_rate',      type=float,     help='initial learning rate', default=1e-4)
parser.add_argument('--batch_size',     type=int,       help='training image batch size',   default=8)
parser.add_argument('--epochs',         type=int,       help='the number of training', default=80)
parser.add_argument('--num_threads',    type=int,       help='the number of cpu core', default=12)
parser.add_argument('--retrain',        type=str,       help='if used with checkpoint_path, will restart training from step zero',
                    default=False)
parser.add_argument('--test_csv_file',  type=str,       help='Read LG CNS test csv file',
                    default='D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\20190403_test_csv')
args = parser.parse_args()

def code_save(params):
    """dataloader, main, model code save"""
    shutil.copy('C:\\Users\\audrb\\PycharmProjects\\LG_CNS\\JW_will\\MK_version\\JW2MK_dataloader\\JW2MK_dataloader_v3_window.py',
                os.path.join(params.base_path, 'Training_result', 'Tensorboard', '201906_Graph', params.main_name))

    shutil.copy('C:\\Users\\audrb\\PycharmProjects\\LG_CNS\\JW_will\\MK_version\\JW2MK_main\\JW2MK_main_v3_window.py',
                os.path.join(params.base_path, 'Training_result', 'Tensorboard', '201906_Graph', params.main_name))

    shutil.copy('C:\\Users\\audrb\\PycharmProjects\\LG_CNS\\JW_will\\MK_version\\JW2MK_model\\JW2MK_model_v3_window.py',
                os.path.join(params.base_path, 'Training_result', 'Tensorboard', '201906_Graph', params.main_name))

class TensorBoardHistory(TensorBoard):
    def __init__(self, log_dir):
        super().__init__(log_dir=log_dir)

    def on_epoch_begin(self, epoch, logs=None):
        # get values
        learning_rate = float(K.get_value(self.model.optimizer.lr))

        # compute learning rate
        learning_rate = learning_rate * (1 - epoch/args.epochs)**0.9
        K.set_value(self.model.optimizer.lr, learning_rate)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['learning_rate'] = K.get_value(self.model.optimizer.lr)

        super().on_epoch_end(epoch, logs)


def train(params):
    resnet_model50 = resnet_windows()

    if args.retrain:
        model_name = 'model-0001.ckpt'
        best_model_path = os.path.join(params.checkpoint_path, '201906_Model', params.main_name + '_model', model_name)
        resnet_model50.load_weights(best_model_path)

    tb_callback = TensorBoardHistory(log_dir=os.path.join(params.base_path, 'Training_result', 'Tensorboard', '201906_Graph', params.main_name))
    # lr_decay = LearningRateScheduler(schedule=lambda epoch: params.learning_rate * (0.9 ** params.epochs))
    # lr_decay = LearningRateScheduler(schedule=lambda epoch: params.learning_rate * (0.9 ** epochs))
    resnet_model50.compile(loss='categorical_crossentropy',
                           optimizer=Adam(lr=params.learning_rate),
                           metrics=['accuracy'])

    # tensorboard = TensorBoard(log_dir=os.path.join(params.base_path, 'Training_result', 'Tensorboard', params.main_name),
    #                           write_images=True, write_graph=True, batch_size=params.batch_size)

    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(params.checkpoint_path, '201906_Model', params.main_name+'_model', 'model-{epoch:04d}.ckpt'),
                                          save_weights_only=True,
                                          verbose=1)

    print('======================== Data Loading ======================== ')
    dataloader = DataGenerator(params)
    train_generator, validation_generator = dataloader.train_generator, dataloader.validation_generator

    print('======================== Training ======================== ')
    resnet_model50.fit_generator(generator=train_generator, steps_per_epoch=len(train_generator),
                                 epochs=params.epochs, validation_data=validation_generator,
                                 validation_steps=len(validation_generator),
                                 callbacks=[tb_callback, checkpoint_callback], workers=params.num_threads)

    print('======================== Evaluation loss and metrics !! ========================')
    loss_and_metrics = resnet_model50.evaluate_generator(
        generator=validation_generator, steps=len(validation_generator), workers=params.num_threads)
    print("%s: %.2f%%" % (resnet_model50.metrics_names[1], loss_and_metrics[1] * 100))
    print('========================================================================')

def test(params):
    loaded_model = resnet_windows()

    model_name = 'model-0072.ckpt'
    best_model_path = os.path.join(params.checkpoint_path, '201906_Model', params.main_name + '_model', model_name)
    # best_model_path = 'D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\Mando_LG_CNS_result\\model-0149.ckpt'
    loaded_model.load_weights(best_model_path)

    test_images = sorted(glob(params.test_image_path + '/*.png'))

    for idx in range(7):
        if os.path.exists(os.path.join(params.test_result_path, 'test_class{}'.format(idx))) != True:
            os.makedirs(os.path.join(params.test_result_path, 'test_class{}'.format(idx)))

    for num, img in enumerate(test_images):
        start = time.time()
        image = misc.imread(img)
        image = np.expand_dims(image, axis=0)
        """Class prediction"""
        class_pred = loaded_model.predict_classes(x=image[:,:,:, 0:3], batch_size=1)
        time_sofar = time.time() - start
        now = datetime.now()

        print('Current Time: {}-{}-{} {}:{}:{}.{}, Predicted class: {}, Checking test Image: {}/{}, Time elapsed: {:f}s'.format(
            now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond, class_pred, num+1, len(test_images), time_sofar))
        shutil.copy(img, os.path.join(params.test_result_path, 'test_class{}'.format(*class_pred)))

        if str(class_pred) == '[1]' or str(class_pred) == '[0]':
            if os.path.exists(os.path.join(params.test_result_path, 'test_class{}'.format(*class_pred), '20190613_V3-9_M3_2_ep72')) != True:
                os.makedirs(os.path.join(params.test_result_path, 'test_class{}'.format(*class_pred), '20190613_V3-9_M3_2_ep72'))
            shutil.copy('D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\20190416_test_3sec_signal\\' + img.split('\\')[-1][:-21] + '_3sec.wav',
                        os.path.join(params.test_result_path, 'test_class{}'.format(*class_pred), '20190613_V3-9_M3_2_ep72'))

    """Plot the prediction"""
    test_batch = np.zeros((1924, 256, 256, 3))
    # test_img = glob(params.test_image_path + '/*_3sec_nonhighpass.png')
    test_img = glob(params.test_image_path + '/*.png')
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
    loaded_model = resnet_windows()

    model_name = 'model-0080.ckpt'
    best_model_path = os.path.join(params.checkpoint_path, params.main_name + '_model', model_name)
    loaded_model.load_weights(best_model_path)

    # fault_path = 'D:\\Onepredict_MK\\LG CNS\\By_datasetMK\\20190430_Fault_op_image'
    fault_path = 'D:\\Onepredict_MK\\LG CNS\\By_datasetMK\\20190411_fault_image'
    for idx in range(7):
        if os.path.exists(os.path.join(params.test_result_path, 'test_class{}'.format(idx))) != True:
            os.makedirs(os.path.join(params.test_result_path, 'test_class{}'.format(idx)))

    for num in range(6):
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
    plt.ylim([-0.5, 6])
    plt.show()

def train_val_test(params):
    loaded_model = resnet_windows()

    model_name = 'model-0080.ckpt'
    best_model_path = os.path.join(params.checkpoint_path, params.main_name+'_model', model_name)
    loaded_model.load_weights(best_model_path)

    if args.test_mode == 'train_test':
        training_path = os.path.join(params.base_path, 'By_datasetMK', '20190430_training_data_M4_2')

        for idx in range(5):
            if os.path.exists(os.path.join(params.test_result_path, 'test_class{}'.format(idx))) != True:
                os.makedirs(os.path.join(params.test_result_path, 'test_class{}'.format(idx)))

        for num in range(4):
            fault_images = glob(os.path.join(training_path, 'fault{}'.format(num), '*.png'))
            for fault_num, fault_img in enumerate(sorted(fault_images)):
                fault_image = misc.imread(fault_img)
                fault_image = np.expand_dims(fault_image, axis=0)
                fault_class_pred = loaded_model.predict_classes(x=fault_image[:, :, :, 0:3], batch_size=1)

                print('Predicted class: {}, Checking test Image: {}/{}'.format(fault_class_pred, fault_num + 1,
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
        plt.ylim([-0.5, 6])
        plt.show()

    elif args.test_mode == 'val_test':
        validation_path = 'D:\\Onepredict_MK\\LG CNS\\By_datasetMK\\20190430_validation_data_M4_2'

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
        if os.path.exists(os.path.join(params.base_path, 'Training_result', 'Tensorboard', '201906_Graph', params.main_name)) != True:
            os.mkdir(os.path.join(params.base_path, 'Training_result', 'Tensorboard',  '201906_Graph', params.main_name))
        if os.path.exists(os.path.join(params.checkpoint_path, '201906_Model', params.main_name + '_model')) != True:
            os.mkdir(os.path.join(params.checkpoint_path, '201906_Model', params.main_name + '_model'))

        code_save(params)
        train(params)

    elif args.mode == 'test':
        test(params)
    elif args.mode == 'fault_test':
        fault_test(params)
    elif args.mode == 'train_val_test':
        train_val_test(params)

if __name__ == '__main__':
    # data_generator_M3_windows(args.base_path)
    main()