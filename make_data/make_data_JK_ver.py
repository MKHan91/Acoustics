import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from PIL import Image

base_path = 'D:/'
listdir = os.listdir(os.path.join(base_path, '03 LG_CNS', 'Mando','fault_data'))

main_path = listdir[2]
print(main_path)
wave_path = os.path.join(base_path, '03 LG_CNS', 'Mando','fault_data', main_path)
print(wave_path)
wave = glob(os.path.join(wave_path, '*.wav'))

total_cutting_line = np.zeros(len(wave))
total_data_len = np.zeros(len(wave))

for num, wave_signal in enumerate(sorted(wave)):
    sr, data = wavfile.read(wave_signal)
    data_len = len(data[:, 0])
    max_value = np.max(data[0:int(data_len/4),0])
    max_index = np.where(data[0:int(data_len / 4), 0] == max_value)
    total_cutting_line[num] = np.max(max_index)
    total_data_len[num] = len(data)

cutting_line = int(np.max(total_cutting_line))
print(cutting_line)
print(np.mean(total_data_len))


for num, wave_signal in enumerate(sorted(wave)):
    sr, data = wavfile.read(wave_signal)
    f, t, Zxx = signal.stft(data[cutting_line:(cutting_line+25000), 0], fs=sr, nperseg=512, noverlap=210)
    # f, t, Zxx = signal.stft(data[cutting_line:len(data), 0], fs=sr, nperseg=512, noverlap=210)
    Zxx = Zxx[:256, :]

    Sxx = np.abs(Zxx)
    Sxx = np.maximum(Sxx, 1e-8)
    mel = 20 * np.log10(Sxx)
    img = (mel + 160) / 160

    STFT_input = img
    STFT_input = Image.fromarray(STFT_input)
    STFT_input = STFT_input.resize((256, 256))
    plt.imsave(
        fname=os.path.join(base_path, wave_signal.split('/')[-1][:-4] + '.png'),
        arr=STFT_input)
    print('Save the image:', num)
#
#
# base_path = 'D:/'
# # listdir = os.listdir(os.path.join(base_path, '03 LG_CNS', 'Mando','fault_data'))
# listdir = os.listdir(os.path.join(base_path, '03 LG_CNS', 'Mando','real0'))
#
# # main_path = listdir[4]
# # print(main_path)
# # wave_path = os.path.join(base_path, '03 LG_CNS', 'Mando','real0', main_path)
# wave_path = os.path.join(base_path, '03 LG_CNS', 'Mando','real0')
# wave = glob(os.path.join(wave_path, '*.wav'))
#
# total_cutting_line = np.zeros(len(wave))
# total_data_len = np.zeros(len(wave))
#
# for num, wave_signal in enumerate(sorted(wave)):
#     sr, data = wavfile.read(wave_signal)
#     data_len = len(data[:])
#     max_value = np.max(data[0:int(data_len/4)])
#     max_index = np.where(data[0:int(data_len / 4)] == max_value)
#     total_cutting_line[num] = np.max(max_index)
#     total_data_len[num] = len(data)
#
# cutting_line = int(np.max(total_cutting_line))
#
# print(cutting_line)
# print(np.mean(total_data_len))
#
# for num, wave_signal in enumerate(sorted(wave)):
#     sr, data = wavfile.read(wave_signal)
#
#     if np.array(data.shape).shape == (2,):
#         f, t, Zxx = signal.stft(data[cutting_line:(cutting_line+25000),0], fs=sr, nperseg=512, noverlap=210)
#         Zxx = Zxx[:256, :]
#     else:
#         f, t, Zxx = signal.stft(data[cutting_line:(cutting_line+25000)], fs=sr, nperseg=512, noverlap=210)
#         Zxx = Zxx[:256, :]
#
#     Sxx = np.abs(Zxx)
#     Sxx = np.maximum(Sxx, 1e-8)
#     mel = 20 * np.log10(Sxx)
#     img = (mel + 160) / 160
#
#     STFT_input = img
#     STFT_input = Image.fromarray(STFT_input)
#     STFT_input = STFT_input.resize((256, 256))
#     save_dir = 'normal'
#     file_name = os.path.splitext(os.path.basename(wave[num]))[0]
#     plt.imsave(
#         fname=os.path.join(base_path, '03 LG_CNS', 'Mando', save_dir, file_name+'.png'),
#         arr=STFT_input)
#     print('Save the image:', num)