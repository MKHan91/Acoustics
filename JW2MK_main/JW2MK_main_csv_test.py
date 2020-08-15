# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from MK_version.JW2MK_dataloader.JW2MK_dataloader_v3_window import *
from MK_version.JW2MK_model.JW2MK_model_v3_window import *
from scipy import signal
from glob import glob
from PIL import Image
from scipy import misc
from datetime import datetime
from scipy.io import wavfile
import numpy as np
import wave
import csv
from tqdm import tqdm_notebook as tqdm
import argparse
import time
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import img_to_array, array_to_img

parser = argparse.ArgumentParser('LG CNS training and test code of Myungkyu')
parser.add_argument('--mode',               type=str,       help='choosing either training or test', default='csv_test')
parser.add_argument('--test_mode',          type=str,       help='choosing either training or test', default='train_test')
parser.add_argument('--main_name',          type=str,       help='current main file name', default='20190524_JW2MK_mainV3_JK_v2')
parser.add_argument('--base_path',          type=str,       help='path containing LG_CNS files',
                    default='/home/onepredict/Myungkyu/LG_CNS')
parser.add_argument('--test_image_path',    type=str,       help='path containing LG_CNS files',
                    default='D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\20190416_test_3sec_image')
parser.add_argument('--test_result_path',    type=str,       help='path containing LG_CNS files',
                    default='/home/onepredict/Myungkyu/LG_CNS/JW_will/Test_result/20190524_test_NoAug_JK_ep46')
parser.add_argument('--checkpoint_path',    type=str,       help='path containing LG_CNS files',
                    default='D:\\Onepredict_MK\\LG CNS\\Training_result\\TrainedModel')
parser.add_argument('--learning_rate',      type=float,     help='initial learning rate', default=3e-4)
parser.add_argument('--batch_size',         type=int,       help='training image batch size', default=16)
parser.add_argument('--epochs',             type=int,       help='the number of training', default=80)
parser.add_argument('--num_threads',    type=int,       help='the number of cpu core', default=12)
parser.add_argument('--test_csv_file',  type=str,       help='Read LG CNS test csv file',
                    default='D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\20190403_test_csv')
args = parser.parse_args()

# def spectrogram():
#     wav_path = 'D:\\Onepredict_MK\\LG CNS\By_datasetMK\\20190411_normal_signal\\20180919071505_Wave1_2018Y09M19D_07h15m05s.wav'
#     # highpass_wav_path = 'D:\\Onepredict_MK\\LG CNS\\By_datasetMK\\20190411_normal_signal\\20180919071505_Wave1_2018Y09M19D_07h15m05s_highpass.wav'
#
#     fault0_path = 'D:\\Onepredict_MK\\LG CNS\\By_datasetMK\\20190411_fault_signal\\fault0\\20181218213456_Wave1_2018Y12M18D_21h34m56s.wav'
#
#     sr, data = wavfile.read(wav_path)
#     # sr_hp, data_hp = wavfile.read(highpass_wav_path)
#     sr_f0, data_f0 = wavfile.read(fault0_path)
#
#     plt.specgram(x = data, Fs=sr)
#     plt.title('Normal')
#     plt.show()
#
#     plt.specgram(x = data_f0, Fs=sr_f0)
#     plt.title('Fault 0')
#     plt.show()
#
#     f, t, Zxx = signal.stft(data, fs=25600, nperseg=512, noverlap=210)

def csv_test(params):
    # spectrogram()

    loaded_model = resnet_windows()

    model_name = 'model-0068.ckpt'
    best_model_path = os.path.join(params.checkpoint_path, params.main_name + '_model', model_name)
    loaded_model.load_weights(best_model_path)

    test_csv = sorted(glob(args.test_csv_file + '\\*.csv'))

    for num, csv_file in enumerate(test_csv):
        data = np.genfromtxt(csv_file, delimiter=',')
        start = time.time()

        data = data[51200:]

        # wavfile.write(filename='D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\1.wav', data=data, rate=25600)
        # _, data = wavfile.read(filename='D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\1.wav')

        f, t, Zxx = signal.stft(data, fs=25600, nperseg=512, noverlap=210)
        Zxx = Zxx[1:, :]

        Sxx = np.abs(Zxx)
        Sxx = np.maximum(Sxx, 1e-8)
        mel = 20 * np.log10(Sxx)
        img = (mel + 160) / 160

        # img_dummy = img
        # img_zero = np.zeros([256, 256, 3])

        # img = img * 160
        # img = img.astype(np.uint8)
        # img = Image.fromarray(img).convert("RGB")
        # a = img_to_array(img=img, dtype=np.uint8)

        plt.imsave('D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\1.png', img)
        # array_img = misc.imread('D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\1.png')
        # img.save('D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\1.bmp')
        image = np.expand_dims(img, axis=0)
        """Class prediction"""
        # class_pred = loaded_model.predict_classes(x=image[:, :, :, 0:3], batch_size=1)
        class_pred = loaded_model.predict_classes(x=image, batch_size=1)
        time_sofar = time.time() - start
        now = datetime.now()

        print('Current Time: {}-{}-{} {}:{}:{}.{}, Predicted class: {}, Checking test Image: {}/{}, Time elapsed: {:f}s'.format(
            now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond, class_pred, num+1, len(test_csv), time_sofar))

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

    if args.mode == 'csv_test':
        csv_test(params)

if __name__ == '__main__':
    main()