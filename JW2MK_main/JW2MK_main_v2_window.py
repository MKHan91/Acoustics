from MK_version.JW2MK_dataloader.JW2MK_dataloader_v2_window import *
from MK_version.JW2MK_model.JW2MK_model_v2_window import *
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from sklearn.utils import class_weight
from datetime import datetime
from glob import glob
from scipy import misc, signal, ndimage
from PIL import Image
import cv2
import shutil
import argparse
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser('LG CNS training and test code of Myungkyu')
parser.add_argument('--mode',           type=str,       help='choosing either training or test', default='train')
parser.add_argument('--test_mode',      type=str,   help='choosing either training or test', default='train_test')
parser.add_argument('--main_name',      type=str,       help='current main file name', default='20190816_JW2MK_main')
parser.add_argument('--pre_main_name',  type=str,       help='current main file name', default='20190624_JW2MK_mainV4-1_M3_2')
parser.add_argument('--base_path',      type=str,       help='path containing LG_CNS files', default='D:\\Onepredict_MK\\05 LG CNS')
parser.add_argument('--checkpoint_path',    type=str,       help='path containing LG_CNS files',
                    default='D:\\Onepredict_MK\\05 LG CNS\\02 Training_result\\TrainedModel')
parser.add_argument('--width',              type=int,       help='initial learning rate', default=256)
parser.add_argument('--height',             type=int,       help='initial learning rate', default=256)
parser.add_argument('--learning_rate',      type=float,     help='initial learning rate', default=1e-4)
parser.add_argument('--batch_size',         type=int,       help='training image batch size',   default=8)
parser.add_argument('--epochs',             type=int,       help='the number of training', default=50)
parser.add_argument('--num_threads',        type=int,       help='the number of cpu core', default=12)
parser.add_argument('--retrain',            type=bool,      help='if used with checkpoint_path, will restart training from step zero',
                    default=True)
parser.add_argument('--test_csv_file',  type=str,       help='Read LG CNS test csv file',
                    default='D:\\Onepredict_MK\\05 LG CNS\\20190417_LG_CNS_test_mode\\20190403_test_csv')
args = parser.parse_args()

def code_save(params):
    """dataloader, main, model code save"""
    dir = os.path.dirname(os.getcwd())

    main_code = sys.argv[0].split('/')[-1]
    x = main_code.split('_')
    x.remove('main')
    x.insert(1, 'dataloader')
    dataloader_code = '_'.join(x)

    x.remove('dataloader')
    x.insert(1, 'model')
    model_code = '_'.join(x)

    shutil.copy(os.path.join(dir, 'JW2MK_dataloader', dataloader_code),
                os.path.join(params.base_path, Training_result, 'Tensorboard', str(year)+'0{}'.format(month)+'_Graph', params.main_name))

    shutil.copy(os.path.join(dir, 'JW2MK_main', main_code),
                os.path.join(params.base_path, Training_result, 'Tensorboard', str(year)+'0{}'.format(month)+'_Graph', params.main_name))

    shutil.copy(os.path.join(dir, 'JW2MK_model', model_code),
                os.path.join(params.base_path, Training_result, 'Tensorboard', str(year)+'0{}'.format(month)+'_Graph', params.main_name))

def check_dir_or_create(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

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

def transfer_learning(network):
    for layer in network.layers:
        layer.trainable = False

    network._layers.pop(-1)
    x = network.layers[-1].output
    predictions = Dense(7, activation='softmax', name='transfer_softmax')(x)

    fine_tune = Model(inputs=network.input, outputs=predictions)
    print(fine_tune.summary())
    return fine_tune

def Transfer_Learning(params):
    network = load_model(os.path.join(params.checkpoint_path, str(year)+'0{}'.format(month)+'_Model', params.pre_main_name+'_model',
                                      params.pre_main_name + '.h5'))

    tb_callback = TensorBoardHistory(log_dir=os.path.join(params.base_path, Training_result, 'Tensorboard', str(year)+'0{}'.format(month)+'_Graph', params.main_name))

    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(params.checkpoint_path, str(year)+'0{}'.format(month)+'_Model', params.main_name+'_model', 'model-{epoch:04d}.ckpt'),
                                          save_weights_only=True,
                                          verbose=1)

    print('======================== Data Loading ======================== ')
    dataloader = DataGenerator(params)
    train_generator, validation_generator = dataloader.train_generator, dataloader.validation_generator

    # fine_network = transfer_learning(network=network)

    print('======================== Fine Training ======================== ')
    network.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=params.learning_rate),
                    metrics=['accuracy'])

    network.fit_generator(generator=train_generator, steps_per_epoch=len(train_generator),
                               epochs=params.epochs, validation_data=validation_generator,
                               validation_steps=len(validation_generator),
                               callbacks=[tb_callback, checkpoint_callback], workers=params.num_threads)

    network.save(filepath=os.path.join(params.checkpoint_path, str(year)+'0{}'.format(month)+'_Model', params.main_name+'_model',
                                       params.main_name + '.h5'))

    print('======================== Evaluation loss and metrics !! ========================')
    loss_and_metrics = network.evaluate_generator(
        generator=validation_generator, steps=len(validation_generator), workers=params.num_threads)
    print("%s: %.2f%%" % (network.metrics_names[1], loss_and_metrics[1] * 100))

def train(params):
    network = resnet_windows(params)
    # network = DiagnosticsNet.densenet_windows(params)
    # network = DiagnosticsNet.incept_resnet_v2_windows(params)

    tb_callback = TensorBoardHistory(log_dir=os.path.join(params.base_path, Training_result, 'Tensorboard', str(year)+'0{}'.format(month)+'_Graph', params.main_name))

    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(params.checkpoint_path, 'Tensorboard', str(year)+'0{}'.format(month)+'_Model', params.main_name+'_model', 'model-{epoch:04d}.ckpt'),
                                          save_weights_only=True,
                                          verbose=1)

    print('======================== Data Loading ======================== ')
    dataloader = DataGenerator(params)
    train_generator, validation_generator = dataloader.train_generator, dataloader.validation_generator
    # class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.labels), y=train_generator.labels)

    print('======================== Training ======================== ')
    network.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=params.learning_rate),
                    metrics=['accuracy'])

    network.fit_generator(generator=train_generator, steps_per_epoch=len(train_generator),
                          epochs=params.epochs, validation_data=validation_generator,
                          validation_steps=len(validation_generator),
                          callbacks=[tb_callback, checkpoint_callback], workers=params.num_threads)

    network.save(filepath=os.path.join(params.checkpoint_path, str(year)+'0{}'.format(month)+'_Model', params.main_name + '_model',
                                       params.main_name + '.h5'))

    print('======================== Evaluation loss and metrics !! ========================')
    loss_and_metrics = network.evaluate_generator(
        generator=validation_generator, steps=len(validation_generator), workers=params.num_threads)
    print("%s: %.2f%%" % (network.metrics_names[1], loss_and_metrics[1] * 100))

def test(params):
    By_datasetMK = os.listdir(params.base_path)[0]
    test_mode = os.listdir(params.base_path)[2]
    Test_result = os.listdir(params.base_path)[0]

    test_images = sorted(glob(os.path.join(params.base_path, By_datasetMK, '201905', '20190508_generated_image', 'fault5', '*.png')))
    test_result_path = os.path.join(params.base_path, test_mode, Test_result, "201907", "20190722_fault5_generated_image_ep80V2")

    # LOAD THE MODEL
    loaded_model = load_model(os.path.join(params.checkpoint_path, str(year)+'0{}'.format(month)+'_Model', params.pre_main_name + '_model',
                                           params.pre_main_name + '.h5'))
    model_name = 'model-0050.ckpt'
    best_model_path = os.path.join(params.checkpoint_path, str(year)+'0{}'.format(month)+'_Model', params.pre_main_name + '_model', model_name)
    loaded_model.load_weights(best_model_path)

    for idx in range(7):
        check_dir_or_create(dir=os.path.join(test_result_path, 'test_class{}'.format(idx)))

    for num, img in enumerate(test_images):
        start = time.time()
        image = misc.imread(img)
        image = image/image.max()

        image_tensor = np.expand_dims(image, axis=0)
        image_tensor = image_tensor[:, :, :, 0:3]
        prediction = loaded_model.predict(x=image_tensor)[0]
        class_pred = np.argmax(prediction)
        # class_pred = loaded_model.predict_classes(x=image_tensor, batch_size=1)
        time_sofar = time.time() - start
        now = datetime.now()

        print('Current Time: {}-{}-{} {}:{}:{}.{}, Predicted class: {}, Checking test Image: {}/{}, Time elapsed: {:f}s'.format(
            now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond, class_pred, num+1, len(test_images), time_sofar))
        shutil.copy(img, os.path.join(test_result_path, 'test_class{}'.format(class_pred)))

        # SAVE THE .WAV FILE
        # date = params.main_name.split('_')[0]
        # ver = params.main_name.split('_')[2]
        # if str(class_pred) == '1' or str(class_pred) == '0':
        #     check_dir_or_create(dir=os.path.join(test_result_path, 'test_class{}'.format(class_pred), date+'_'+ver+'_wavfile'))
        #     shutil.copy('D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\20190416_test_3sec_signal\\' + img.split('\\')[-1][:-21] + '_3sec.wav',
        #                 os.path.join(test_result_path, 'test_class{}'.format(class_pred), date+'_'+ver+'_wavfile'))

    # PLOT THE RESULT
    test_batch = np.zeros((len(test_images), params.height, params.width, 3))
    for batch, image in enumerate(sorted(test_images)):
        image = misc.imread(image)
        image = image[:, :, 0:3]
        image = image/image.max()
        # RESCALE THE IMAGE
        # image = image/255.0
        test_batch[batch, :, :, :] = image

    prediction = loaded_model.predict(test_batch, batch_size=params.batch_size, workers=params.num_threads)
    plt.figure(figsize=(12, 12))
    plt.plot(np.argmax(prediction, 1), '.', )
    plt.ylim([-0.5, 6.5])
    # plt.title(params.main_name + '-ep12')
    plt.show()

def fault_test(params):
    loaded_model = resnet_windows(params)
    loaded_model.compile(loss='categorical_crossentropy',
                         optimizer=Adam(lr=params.learning_rate),
                         metrics=['accuracy'])

    model_name = 'model-0080.ckpt'
    best_model_path = os.path.join(params.checkpoint_path, '201906_Model', params.main_name + '_model', model_name)
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
            fault_image = fault_image / 255.0
            # fault_class_pred = loaded_model.predict_classes(x=fault_image[:,:,:, 0:3], batch_size=1)
            fault_class_pred = loaded_model.predict(x=fault_image[:, :, :, 0:3], batch_size=1)
            fault_pred = np.argmax(fault_class_pred[0])

            print('Predicted class: {}, Checking test Image: {}/{}'.format(fault_pred, num + 1, len(fault_images)))
            shutil.copy(fault_img, os.path.join(params.test_result_path, 'test_class{}'.format(fault_pred)))

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
    loaded_model = densenet_windows(params)

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
    if args.mode == 'train':
        check_dir_or_create(dir=os.path.join(params.base_path, Training_result, 'Tensorboard', str(year) + '0{}'.format(month) + '_Graph'))
        check_dir_or_create(dir=os.path.join(os.path.join(params.checkpoint_path, str(year) + '0{}'.format(month) + '_Model')))
        check_dir_or_create(dir=os.path.join(params.base_path, Training_result, 'Tensorboard', str(year)+'0{}'.format(month)+'_Graph', params.main_name))
        check_dir_or_create(dir=os.path.join(os.path.join(params.checkpoint_path, str(year)+'0{}'.format(month)+'_Model', params.main_name + '_model')))
        code_save(params)

        train(params)
    elif args.mode == 'transfer':
        check_dir_or_create(dir=os.path.join(params.base_path, Training_result, 'Tensorboard', str(year)+'0{}'.format(month)+'_Graph', params.main_name))
        check_dir_or_create(dir=os.path.join(os.path.join(params.checkpoint_path, str(year)+'0{}'.format(month)+'_Model', params.main_name + '_model')))
        code_save(params)

        Transfer_Learning(params)
    elif args.mode == 'test':
        test(params)
    elif args.mode == 'fault_test':
        fault_test(params)
    elif args.mode == 'train_val_test':
        train_val_test(params)

if __name__ == '__main__':
    params = JW2MK_parameters(
        width=args.width,
        height=args.height,
        main_name=args.main_name,
        pre_main_name=args.pre_main_name,
        base_path=args.base_path,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        num_threads=args.num_threads,
    )

    now = datetime.now()
    year = now.year
    month = now.month

    Training_result = os.listdir(params.base_path)[1]
    main()