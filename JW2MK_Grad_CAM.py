from JW_will.MK_version.JW2MK_model.JW2MK_model_v2_window import *
from Guided_grad_CAM import *
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
from scipy.io import wavfile
from scipy import ndimage
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
import librosa.display as display
import numpy as np
import tensorflow as tf
import argparse
import os
import cv2

parser = argparse.ArgumentParser('CAM code of Myungkyu')
parser.add_argument('--main_name',      type=str,       help='current main file name', default='20190624_JW2MK_mainV4-1_M3_2')
parser.add_argument('--data_path',      type=str,       help='path containing LG_CNS files',
                    default='D:\\Onepredict_MK\\LG CNS\\By_datasetMK\\201904\\20190411_fault_image')
parser.add_argument('--write_mode',     type=bool,      help="writting maximum grad-cam value", default=False)
parser.add_argument('--data_mode',      type=str,      help="writting maximum grad-cam value", default='fault')
# parser.add_argument('--test_result_path',    type=str,       help='path containing LG_CNS files',
#                     default='D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\Test_result\\Grad_CAM_result\\fault_image')
parser.add_argument('--test_result_path',    type=str,       help='path containing LG_CNS files',
                    default='D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\Test_result\\201906\\20190624_NormalData_ep80')
parser.add_argument('--checkpoint_path',    type=str,       help='path containing LG_CNS files',
                    default='D:\\Onepredict_MK\\LG CNS\\Training_result\\TrainedModel')

args = parser.parse_args()

def example():
    image_path = 'D:\\Onepredict_MK\\LG CNS\\cat.jpg'

    img = image.load_img(image_path, target_size=(224, 224))
    img_input = preprocess_input(np.expand_dims(img, 0))

    resnet = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=True)

    probs = resnet.predict(img_input)
    pred = np.argmax(probs[0])

    activation_layer = resnet.layers[-3].name
    inp = resnet.input
    # for idx in range(1000):
    y_c = resnet.output.op.inputs[0][:, pred]
    A_k = resnet.get_layer(activation_layer).output

    grads = K.gradients(y_c, A_k)[0]
    # Model(inputs=[inp], outputs=[A_k, grads, resnet.output])
    get_output = K.function(inputs=[inp], outputs=[A_k, grads, resnet.output])
    [conv_output, grad_val, model_output] = get_output([img_input])

    conv_output = conv_output[0]
    grad_val = grad_val[0]

    weights = np.mean(grad_val, axis=(0, 1))
    grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for k, w in enumerate(weights):
        grad_cam += w * conv_output[:, :, k]
        # RELU
        grad_cam = np.maximum(grad_cam, 0)

    grad_cam = cv2.resize(grad_cam, (224, 224))

    # Guided grad-CAM
    register_gradient()
    guided_model, activation_layer = modify_backprop(resnet, 'GuidedBackProp', args.checkpoint_path, args.main_name)
    saliency_fn = compile_saliency_function(guided_model, activation_layer)
    saliency = saliency_fn([img_input, 0])
    gradcam = saliency[0] * grad_cam[..., np.newaxis]
    gradcam = deprocess_image(gradcam)

    # grad_cam = ndimage.zoom(grad_cam, (32, 32), order=1)
    plt.subplot(1, 2, 1)
    plt.imshow(img, alpha=0.8)
    plt.imshow(grad_cam, cmap='jet', alpha=0.5)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gradcam, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()

def scatter(fpath):
    grad_file = open(fpath, "r+")
    for row in grad_file:
        sent = row.rstrip()
        pos = sent.find(':')
        comma = sent.find(',', pos)
        episode = int(sent[pos+1:comma].rstrip())

        pos = sent.find(':', comma)
        comma = sent.find(',', pos)
        if comma == -1:
            comma = len(sent)
            score = float(sent[pos+1:comma].rstrip())

            plt.plot(episode, score, 'o')
    plt.xlabel('The number of data')
    plt.ylabel('Maximum grad cam')
    plt.show()
    print("Finish")
    grad_file.close()

def get_f_t():
    f_list = []
    t_list = []
    base_path = 'D:\\Onepredict_MK\\LG CNS\\By_datasetMK'
    for i in range(6):
        wave_path = os.path.join(base_path, '201904', '20190411_fault_signal', 'fault{}').format(i)
        wave = sorted(glob(os.path.join(wave_path, '*.wav')))
        num2 = 0
        for num, signal_ in enumerate(wave):
            sr, data = wavfile.read(signal_)
            f, t, _ = signal.stft(data, fs=sr, nperseg=512, noverlap=210)
            f_list.append(f)
            t_list.append(t)

    return f_list, t_list

def make_gard_CAM(ckpt_path, main_name):
    test_images = fault_images
    # test_images = sorted(glob(args.data_path + '/*.png'))
    # test_images = normal_images

    best_model_path = os.path.join(ckpt_path, '201906_Model', main_name + '_model', main_name+'.h5')
    network = load_model(filepath=best_model_path)

    activation_layer = network.layers[-9].name
    inp = network.input
    A_k = network.get_layer(activation_layer).output

    register_gradient()
    guided_model, activation_layer = modify_backprop(network, 'GuidedBackProp', ckpt_path, main_name)
    saliency_fn = compile_saliency_function(guided_model, activation_layer)

    for num, img in enumerate(test_images):
        # if not '10h35m42s' in img.split('\\')[-1]:
        #     continue
        # image = misc.imread(img)
        # image = np.expand_dims(image, axis=0)
        img_pil = image.load_img(img, target_size=(256, 256))
        img_input = np.asarray(img_pil, dtype=np.float32)
        img_input = img_input / 255.0
        img_tensor = np.expand_dims(img_input, axis=0)
        # img_tensor = img_tensor[:, :, :, 0:3]

        probs = network.predict(img_tensor)
        pred = np.argmax(probs[0])

        "SAVE THE IMAGE"
        # if os.path.exists(os.path.join(args.test_result_path, 'Grad_CAM{}'.format(pred))) != True:
        #     os.makedirs(os.path.join(args.test_result_path, 'Grad_CAM{}'.format(pred)))

        y_c = network.output.op.inputs[0][0, pred]

        grads = K.gradients(y_c, A_k)[0]
        get_output = K.function(inputs=[inp], outputs=[A_k, grads, network.output])
        [conv_output, grad_val, model_output] = get_output([img_tensor])

        conv_output = conv_output[0]
        grad_val = grad_val[0]

        weights = np.mean(grad_val, axis=(0, 1))
        grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
        for k, w in enumerate(weights):
            # grad_cam += np.maximum(w * conv_output[:, :, k], 0)
            grad_cam += w * conv_output[:, :, k]
            "RELU"
            grad_cam = np.maximum(grad_cam, 0)
            "NORMALIZATION"
            # if grad_cam.max() == 0:
            #     grad_cam += 1e-18
            # grad_cam = grad_cam / grad_cam.max()

            # grad_cam = cv2.resize(grad_cam, (img_input.shape[0], img_input.shape[1]))
        grad_cam = ndimage.zoom(grad_cam, (32, 32), order=1)

        f_list, t_list = get_f_t()
        plt.pcolormesh(f_list[0], t_list[0], img_input)

        "Guided grad-CAM"
        saliency = saliency_fn([img_tensor, 0])
        gradcam = saliency[0] * grad_cam[..., np.newaxis]
        guided_back_cam = deprocess_image(gradcam)
        # guided_back_cam = guided_back_cam/guided_back_cam.max()

        # plt.imsave(fname=os.path.join(args.test_result_path, 'Grad_CAM{}'.format(pred),
        #                               img.split('\\')[-1][:-4] + '_guided.png'), arr=guided_back_cam)

        "PLOT"
        plt.imshow(img_pil, alpha=0.8)
        plt.imshow(grad_cam, cmap='jet', alpha=0.5)
        plt.colorbar()
        # plt.clim(0, 11)
        plt.axis('off')
        save_path = os.path.join(args.test_result_path, 'Grad_CAM{}'.format(pred), img.split('\\')[-1])
        plt.savefig(save_path)
        plt.close()
        print('processing: {}/{}'.format(num + 1, len(test_images)))

        if args.write_mode == True:
            Write_mode(index=num, grad_cam=grad_cam)

def Write_mode(index, grad_cam):
    "WRITE MAXIMUM VALUE OF GRAD_CAM FOR SCATTER PLOTTING"
    log_grad_file = open(os.path.join(args.test_result_path, "scatter_plot_log.txt"), "w")
    grad_file = open(os.path.join(args.test_result_path, "scatter_plot.txt"), "w")

    mv = grad_cam.max()
    log_mv = np.log10(mv)
    log_grad_file.write('index:{}, log_grad_cam:{} \n'.format(str(index + 1), str(log_mv)))
    grad_file.write('index:{}, grad_cam:{} \n'.format(str(index + 1), str(mv)))

    scatter(fpath=os.path.join(args.test_result_path, "scatter_plot_log.txt"))
    scatter(fpath=os.path.join(args.test_result_path, "scatter_plot.txt"))

    log_grad_file.close()
    grad_file.close()

if __name__ == '__main__':
    # example()
    if args.data_mode == 'fault':
        fault_mode = ['fault0', 'fault1', 'fault2', 'fault3', 'fault4', 'fault5']
        for mode in fault_mode:
            fault_images = sorted(glob(os.path.join(args.data_path, mode, '*.png')))
            make_gard_CAM(args.checkpoint_path, args.main_name)

    elif args.data_mode == 'normal':
        normal_images = sorted(glob(os.path.join(args.data_path, '*.png')))
        make_gard_CAM(args.checkpoint_path, args.main_name)
    elif args.data_mode == 'None':
        make_gard_CAM(args.checkpoint_path, args.main_name)
