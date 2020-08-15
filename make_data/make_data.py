from JW2MK_model.JW2MK_model_v2_window import *
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from glob import glob
from scipy.io import wavfile
from scipy import signal
from PIL import Image

def make_npy(base_path):
    input_image_path = os.path.join(base_path, 'LG_CNS_data', 'By_datasetMK', '20190424_training_data')
    input_array = np.empty(shape=(9825, 256, 256, 3))

    for folder in sorted(os.listdir(input_image_path)):
        print('====== folder name: ', folder ,'=========')
        input_image_list = sorted(glob(os.path.join(input_image_path, folder, '*.png')))

        """Spend too many time.... ==> make npy file!"""
        for index, image in enumerate(input_image_list):
            img = Image.open(image)
            np_img = np.asarray(img)[:, :, 0:3]
            input_array[index, :, :, :] = np_img
            print('Make array: {}'.format(index))

    input_x = input_array
    save_dir = os.path.join('/', *input_image_path.split('/')[:-1])
    np.save(file=os.path.join(save_dir, 'training_image.npy'), arr=input_x)

def Mando_WAV2IMG(m_base_path):
    for mando_type in sorted(os.listdir(m_base_path)):
        m_wave = sorted(glob(os.path.join(m_base_path, mando_type, "*.wav")))

        total_cutting_line = np.zeros(len(m_wave))
        total_data_len = np.zeros(len(m_wave))

        for num, wave_signal in enumerate(m_wave):
            sr, data = wavfile.read(wave_signal)

            data_len = len(data[:, 0])
            max_value = np.max(data[0:int(data_len / 4), 0])
            max_index = np.where(data[0:int(data_len / 4), 0] == max_value)
            total_cutting_line[num] = np.max(max_index)
            total_data_len[num] = len(data)

        cutting_line = int(np.max(total_cutting_line))

        for num, wave_signal in enumerate(m_wave):
            sr, data = wavfile.read(wave_signal)
            f, t, Zxx = signal.stft(data[cutting_line:(cutting_line + 25000), 0], fs=sr, nperseg=512, noverlap=210)
            # f, t, Zxx = signal.stft(data[cutting_line:len(data), 0], fs=sr, nperseg=512, noverlap=210)
            Zxx = Zxx[:256, :]
            # f, t, Zxx = signal.stft(data, fs=sr, nperseg=512, noverlap=210)
            # Zxx = Zxx[1:, :]

            Sxx = np.abs(Zxx)
            Sxx = np.maximum(Sxx, 1e-8)
            mel = 20 * np.log10(Sxx)
            img = (mel + 160) / 160
            STFT_input = img
            STFT_input = Image.fromarray(STFT_input)
            STFT_input = STFT_input.resize((256, 256))

            plt.imsave(
                fname=os.path.join(m_base_path, mando_type, wave_signal.split('/')[-1][:-4] + '.png'),
                arr=STFT_input)
            print('Save the image:', num + 1)

def Normal_WAV2IMG(base_path):
    wave_path = os.path.join(base_path, 'By_datasetMK', '20190411_normal_signal')
    # wave = glob(os.path.join(wave_path, 'normal', '*.wav'))
    wave = glob(os.path.join(wave_path,'*.wav'))

    for num, wave_signal in enumerate(sorted(wave)):
        # if '_highpass' in wave_signal:
        #     continue
        sr, data = wavfile.read(wave_signal)
        f, t, Zxx = signal.stft(data, fs=sr, nperseg=512, noverlap=210)
        Zxx = Zxx[1:, :]

        Sxx = np.abs(Zxx)
        Sxx = np.maximum(Sxx, 1e-8)
        mel = 20 * np.log10(Sxx)
        img = (mel + 160) / 160

        # plt.imsave(
        #     fname=os.path.join(wave_path, wave_signal.split('/')[-1][:-4] + '.png'),
        #     arr=img
        # )

        plt.imsave(
            fname=os.path.join('/home/onepredict/Myungkyu/LG_CNS/LG_CNS_data/By_datasetMK/20190411_normal_image', wave_signal.split('/')[-1][:-4] + '.png'),
            arr=img
        )

        #plt.imsave(
        #    fname='/home/onepredict/Myungkyu/LG_CNS/LG_CNS_data/By_datasetMK/20190427_temp/1.png',
        #    arr=img,
        #    vmin=0,
        #    vmax=0.5
        # )
        print('Save the image:', num)

def Fault_WAV2IMG(base_path):
    for i in range(6):
        wave_path = os.path.join(base_path, 'By_datasetMK', '20190520_training_data_M3_2', 'fault{}').format(i)
        # wave_path = os.path.join(base_path, 'LG_CNS_data', 'By_datasetMK', '20190430_JW_choice_fault', 'fault{}').format(i)
        wave = glob(os.path.join(wave_path, '*.wav'))
        num2 = 0

        for num, signal_ in enumerate(sorted(wave)):
            # if '_highpass.wav' in signal_:
            #     num2 += 1
            #     continue
            sr, data = wavfile.read(signal_)
            f, t, Zxx = signal.stft(data, fs=sr, nperseg=512, noverlap=210)
            Zxx = Zxx[1:, :]

            Sxx = np.abs(Zxx)
            Sxx = np.maximum(Sxx, 1e-8)
            mel = 20 * np.log10(Sxx)
            img = (mel + 160) / 160

            # sp = wave_path.split('/')[:-1]
            # sp = wave_path.split('/')[:-2]
            # plt.imsave(
            #     fname=os.path.join('/', *sp, '20190411_normal_image', signal_.split('/')[-1])[:-4] + '_nonhighpass.png',
            #     arr=img)

            # if os.path.exists(os.path.join(base_path, 'LG_CNS_data', 'By_datasetMK', '20190411_fault_image', 'fault{}'.format(i))) != True:
            #     os.makedirs(os.path.join(base_path, 'LG_CNS_data', 'By_datasetMK', '20190411_fault_image', 'fault{}'.format(i)))

            plt.imsave(
                fname=os.path.join(base_path, 'By_datasetMK', '20190520_training_data_M3_2', 'fault{}', signal_.split('/')[-1][:-13]+'_fault_highpass.png').format(i),
                arr=img)
            print('Save the image:', (num+1) - num2)
        print('Converting End!!')

# def CAM(ckpt_path, main_name):
#     test_images = sorted(glob(args.data_path + '/*.png'))
#
#     resnet = resnet_windows()
#     best_model_path = os.path.join(ckpt_path, '201906_Model', main_name + '_model', 'model-0014.ckpt')
#     resnet.load_weights(best_model_path)
#
#     for num, img in enumerate(test_images):
#         img_pil = image.load_img(img, target_size=(256, 256))
#         img_input = np.asarray(img_pil, dtype=np.float32)
#         img_input = np.expand_dims(img_input, axis=0)
#         img_input = img_input / 255.0
#
#         activation_layer = resnet.layers[0].get_layer('activation_48')
#         model = Model(inputs=resnet.layers[0].input, outputs=activation_layer.output)
#         final_dense = resnet.get_layer('dense')
#         weight = final_dense.get_weights()[0]  # weights.shape = (2048, 2048)
#
#         fmaps = model.predict(img_input)[0]  # fmaps.shape = (8, 8, 2048)
#         probs = resnet.predict(img_input)
#         pred = np.argmax(probs[0])  # matrix index
#
#         """Extracting weight corresponding to specific class (pred)"""
#         w = weight[:, pred]
#         cam = fmaps.dot(w)
#         camp = ndimage.zoom(cam, (32, 32), order=1)
#
#         plt.imshow(img_pil, alpha=1.0, aspect='auto')
#         plt.imshow(camp, cmap='jet', alpha=0.6)
#         plt.axis('off')

parser = argparse.ArgumentParser('Make data set')
parser.add_argument('--mode',               type=str,   help='Image types to change', default='cns_normal')
parser.add_argument('--base_path',          type=str,   help='Data path',   default='/home/onepredict/Myungkyu/LG_CNS/LG_CNS_data')
parser.add_argument('--mando_base_path',    type=str,  help='only mando path', default='/home/onepredict/JaeKyung/Mando/20190520_test_data')
parser.add_argument('--main_name',          type=str,       help='current main file name', default='20190611_JW2MK_mainV2_M3_2')
parser.add_argument('--checkpoint_path',    type=str,       help='path containing LG_CNS files',
                    default='D:\\Onepredict_MK\\LG CNS\\Training_result\\TrainedModel')
args = parser.parse_args()

if __name__ == '__main__':
    if args.mode == 'cns_normal':
        Normal_WAV2IMG(args.base_path)
    elif args.mode == 'cns_fault':
        Fault_WAV2IMG(args.base_path)
    elif args.mode == 'mando':
        Mando_WAV2IMG(args.mando_base_path)
    # elif args.mode == 'CAM':
    #     CAM(args.checkpoint_path, args.main_name)
