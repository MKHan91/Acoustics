from JW_will.MK_version.JW2MK_model.JW2MK_model_v2_window import *
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import librosa
import pylab
import librosa.display as display
from glob import glob
from scipy.io import wavfile
from scipy import signal
from PIL import Image
from scipy import misc

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

def Test_WAV2IMG():
    test_path = 'D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\20190416_test_3sec_signal'
    wave = sorted(glob(os.path.join(test_path, '*.wav')))

    for num, sig in enumerate(wave):
        sr, data = wavfile.read(sig)
        f, t, Zxx = signal.stft(data, fs=sr, nperseg=512, noverlap=210)
        Zxx = Zxx[1:, :]

        Sxx = np.abs(Zxx)
        Sxx = np.maximum(Sxx, 1e-8)
        mel = 20 * np.log10(Sxx)
        img = (mel + 160) / 160

        plt.pcolormesh(t, f, img)
        plt.xlabel('Time(s)')
        plt.ylabel('Hz')
        plt.show()

def Normal_WAV2IMG(base_path):
    wave_path = os.path.join(base_path, '201904', '20190411_normal_signal')
    save_path = os.path.join(base_path, '201907', '20190717_normal_image')

    wave = glob(wave_path + '\\*.wav')
    for num, wave_signal in enumerate(sorted(wave)):
        if '_highpass' in wave_signal:
            continue
        sr, data = wavfile.read(wave_signal)
        f, t, Zxx = signal.stft(data, fs=sr, nperseg=512, noverlap=210)
        Zxx = Zxx[1:, :]

        Sxx = np.abs(Zxx)
        Sxx = np.maximum(Sxx, 1e-8)
        mel = 20 * np.log10(Sxx)
        img = (mel + 160) / 160

        plt.imsave(
            fname=os.path.join(save_path, wave_signal.split('\\')[-1][:-4] + '.png'),
            arr=img
        )

        #plt.imsave(
        #    fname='/home/onepredict/Myungkyu/LG_CNS/LG_CNS_data/By_datasetMK/20190427_temp/1.png',
        #    arr=img,
        #    vmin=0,
        #    vmax=0.5
        # )
        print('Save the image:{}'.format(num + 1))

def Fault_WAV2IMG(base_path):
    for i in range(6):
        wave_path = os.path.join(base_path, '201904', '20190411_fault_signal', 'fault{}').format(i)
        wave = sorted(glob(os.path.join(wave_path, '*.wav')))
        num2 = 0
        for num, signal_ in enumerate(wave):
            # if not '10h35m42s.wav' in signal_.split('\\')[-1]:
            #     continue
            sr, data = wavfile.read(signal_)
            f, t, Zxx = signal.stft(data, fs=sr, nperseg=512, noverlap=210)
            Zxx = Zxx[1:, :]

            Sxx = np.abs(Zxx)
            Sxx = np.maximum(Sxx, 1e-8)
            mel = 20 * np.log10(Sxx)
            img = (mel + 160) / 160

            plt.pcolormesh(t, f, img)
            plt.xlabel('Time(s)')
            plt.ylabel('Hz')
            # librosa.display.specshow(img, fmax=25600, y_axis='hz', x_axis='s', sr=sr)
            # plt.show()

            print('Save the image:', (num+1) - num2)
        print('Converting End!!')

def Melspectro(data_path, type_):
    class_name = ['fault0', 'fault1', 'fault2', 'fault3', 'fault4', 'fault5']

    if type_ == 'fault':
        for cname in class_name:
            wave_paths = glob(os.path.join(data_path, '201904', '20190411_fault_signal', cname, '*.wav'))

            for num, wave_path in enumerate(sorted(wave_paths)):
                name = wave_path.split('\\')[-1][:-4]
                # if os.path.exists(os.path.join(data_path, '201907', '20190708_fault_image', cname)) != True:
                #     os.makedirs(os.path.join(data_path, '201907', '20190708_fault_image', cname))

                sr, audio = wavfile.read(filename=wave_path)
                # S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=512, hop_length=512)
                S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, fmax=12800)
                db = librosa.power_to_db(S, ref=np.max)
                display.specshow(db, y_axis='mel', x_axis='s', sr=25600)
                plt.xlabel('Time(s)')
                plt.ylabel('mel-scale')
                plt.show()

                pylab.figure(figsize=(3, 3))
                # pylab.axis('off')

                # Remove the white edge, # frameon: 범례에 테투리를 씌울 것인지 결정
                pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
                # plt.imshow(db, cmap='jet')
                # plt.show()
                # plt.imsave(fname='D:\\Onepredict_MK\\LG CNS\\By_datasetMK\\201907\\20190705_fault_image\\{}_nonhighpass'.format(wave_path.split('\\')[-1][:-4]),
                #            arr=db)
                librosa.display.specshow(db, fmax=12800, cmap='jet')
                pylab.savefig(
                    os.path.join(data_path, '201907', '20190708_fault_image', cname, '{}_nonhighpass.png'.format(name)))
                pylab.close()
                print('current processing: {}/{}, fault: {}'.format(num + 1, len(wave_paths), cname))
    else:
        wave_paths = glob(os.path.join(data_path, '20190411_normal_signal', '*.wav'))

        for idx, wave_path in enumerate(sorted(wave_paths)):
            if '_highpass' in wave_path:
                continue
            name = wave_path.split('\\')[-1][:-4]
            sr, audio = wavfile.read(filename=wave_path)
            S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, fmax=12800)
            db = librosa.power_to_db(S, ref=np.max)

            pylab.figure(figsize=(3, 3))
            pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[],yticks=[])
            librosa.display.specshow(db, fmax=12800, cmap='jet')

            pylab.savefig(
                os.path.join(data_path, '201907', '20190708_normal_image','{}_nonhighpass.png'.format(name)))
            pylab.close()
            print('current processing: {}/{}'.format(idx + 1, len(wave_paths)))

def test_Melspectro(data_path):
    save_path = '\\'.join(data_path.split('\\')[:-1]) + '\\20190417_LG_CNS_test_mode\\20190709_test_3sec_image'
    data_path = '\\'.join(data_path.split('\\')[:-1]) + '\\20190417_LG_CNS_test_mode\\20190416_test_3sec_signal'

    sig = glob(data_path+'\\*.wav')
    for num, wave in enumerate(sorted(sig)):
        sr, audio = wavfile.read(filename=wave)
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, fmax=12800)
        db = librosa.power_to_db(S, ref=np.max)

        pylab.figure(figsize=(3, 3))

        # Remove the white edge, # frameon: 범례에 테투리를 씌울 것인지 결정
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        librosa.display.specshow(db, fmax=12800, cmap='jet')

        pylab.savefig(os.path.join(save_path, wave.split('\\')[-1][:-4] + '_nonhighpass.png'))

        pylab.close()
        print('current processing: {}/{}'.format(num + 1, len(sig)))

parser = argparse.ArgumentParser('Make data set')
parser.add_argument('--mode',               type=str,   help='Image types to change', default='test')
parser.add_argument('--type_',               type=str,   help='Image types to change', default='fault')
parser.add_argument('--base_path',          type=str,   help='Data path',   default='D:\\Onepredict_MK\\LG CNS\\By_datasetMK')
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
    elif args.mode == 'Melspectro':
        Melspectro(args.base_path, args.type_)
    elif args.mode == 'test_Melspectro':
        test_Melspectro(args.base_path)
    elif args.mode == 'test':
        Test_WAV2IMG()